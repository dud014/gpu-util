#!/usr/bin/env python3
"""GPU idle manager that runs a dummy CUDA workload when each GPU is idle.

The manager polls `nvidia-smi` on a configurable interval, starting a per-GPU
dummy workload whenever that GPU is idle and stopping it as soon as other
compute processes are detected on the device.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

LOG = logging.getLogger("gpu-idle-manager")


@dataclass
class GPUState:
    """Track metadata, dummy workload, and usage metrics for a single GPU."""

    index: int
    uuid: str
    dummy_proc: Optional[subprocess.Popen] = None
    external_usage_count: int = 0
    has_external_activity: bool = False


def run_command(cmd: List[str]) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except FileNotFoundError as exc:
        LOG.error("Failed to run %s: %s", cmd[0], exc)
        return None


def get_gpu_inventory() -> List[GPUState]:
    """Return the list of GPUs available on the system."""
    result = run_command([
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader",
    ])
    if result is None:
        return []
    if result.returncode != 0:
        LOG.error("nvidia-smi failed: %s", result.stderr.strip())
        return []
    gpus: List[GPUState] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            index_str, uuid = [part.strip() for part in line.split(",", maxsplit=1)]
        except ValueError:
            LOG.debug("Unable to parse GPU info from line: %s", line)
            continue
        try:
            index = int(index_str)
        except ValueError:
            LOG.debug("Unable to parse GPU index from line: %s", line)
            continue
        gpus.append(GPUState(index=index, uuid=uuid))
    return gpus


def get_gpu_compute_processes() -> Dict[str, List[int]]:
    """Return mapping of GPU UUID to compute process PIDs."""
    result = run_command([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader",
    ])
    if result is None:
        return {}
    if result.returncode != 0:
        LOG.error("nvidia-smi failed: %s", result.stderr.strip())
        return {}
    gpu_processes: Dict[str, List[int]] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or "No running processes" in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            LOG.debug("Unable to parse process info from line: %s", line)
            continue
        uuid, pid_str = parts
        try:
            pid = int(pid_str)
        except ValueError:
            LOG.debug("Unable to parse pid from line: %s", line)
            continue
        gpu_processes.setdefault(uuid, []).append(pid)
    return gpu_processes


def build_dummy_executable(src: Path, build_dir: Path) -> Optional[Path]:
    build_dir.mkdir(parents=True, exist_ok=True)
    exe_path = build_dir / "dummy_spin"
    if exe_path.exists():
        src_mtime = src.stat().st_mtime
        exe_mtime = exe_path.stat().st_mtime
        if exe_mtime >= src_mtime:
            return exe_path
    nvcc = os.environ.get("NVCC", "nvcc")
    LOG.info("Compiling dummy CUDA workload using %s", nvcc)
    result = run_command([nvcc, str(src), "-O2", "-o", str(exe_path)])
    if result is None:
        return None
    if result.returncode != 0:
        LOG.error("Failed to compile dummy workload: %s", result.stderr.strip())
        return None
    return exe_path


def start_dummy_process(exe: Path, gpu_index: int) -> subprocess.Popen:
    LOG.info("Starting dummy GPU workload on GPU %s: %s", gpu_index, exe)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return subprocess.Popen([str(exe), "--device", "0"], env=env)


def stop_dummy_process(proc: subprocess.Popen, gpu_index: int) -> None:
    LOG.info("Stopping dummy workload on GPU %s", gpu_index)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        LOG.warning("Dummy workload on GPU %s did not exit gracefully, killing.", gpu_index)
        proc.kill()


def manage_gpu_idle(dummy_exe: Path, poll_interval: float) -> None:
    """Continuously poll GPU activity and manage the dummy workload accordingly."""
    gpu_states = get_gpu_inventory()
    if not gpu_states:
        LOG.error("No GPUs detected. Exiting.")
        return

    stopping = False

    def handle_signal(signum, frame):
        nonlocal stopping
        stopping = True
        LOG.info("Received signal %s, shutting down", signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while not stopping:
        active_processes = get_gpu_compute_processes()

        for gpu in gpu_states:
            proc = gpu.dummy_proc
            if proc and proc.poll() is not None:
                gpu.dummy_proc = None
                proc = None

            gpu_pids = active_processes.get(gpu.uuid, [])
            dummy_pid = proc.pid if proc else None
            external_pids = [pid for pid in gpu_pids if pid != dummy_pid]

            if external_pids:
                if not gpu.has_external_activity:
                    gpu.external_usage_count += 1
                    gpu.has_external_activity = True
                    LOG.info(
                        "External GPU usage detected on GPU %s (events=%s, PIDs: %s).",
                        gpu.index,
                        gpu.external_usage_count,
                        ", ".join(map(str, external_pids)),
                    )
                if proc and proc.poll() is None:
                    stop_dummy_process(proc, gpu.index)
                    gpu.dummy_proc = None
            else:
                if gpu.has_external_activity:
                    gpu.has_external_activity = False
                    LOG.info(
                        "GPU %s returned to idle (external usage events=%s).",
                        gpu.index,
                        gpu.external_usage_count,
                    )
                if proc is None:
                    gpu.dummy_proc = start_dummy_process(dummy_exe, gpu.index)

        time.sleep(poll_interval)

    for gpu in gpu_states:
        proc = gpu.dummy_proc
        if proc and proc.poll() is None:
            stop_dummy_process(proc, gpu.index)
        LOG.info(
            "GPU %s summary: detected %s external usage event(s).",
            gpu.index,
            gpu.external_usage_count,
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="Directory to store compiled dummy workload"
    )
    parser.add_argument(
        "--dummy-src",
        type=Path,
        default=Path(__file__).resolve().parent / "dummy_spin.cu",
        help="Path to the CUDA source file for dummy workload"
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    dummy_exe = build_dummy_executable(args.dummy_src, args.build_dir)
    if not dummy_exe:
        LOG.error("Unable to build dummy workload. Exiting.")
        return 1

    try:
        manage_gpu_idle(dummy_exe, args.poll_interval)
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
