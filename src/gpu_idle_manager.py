#!/usr/bin/env python3
"""GPU idle manager that runs a dummy CUDA workload when the GPU is idle.

The manager polls `nvidia-smi` on a configurable interval, starting the dummy
workload whenever the GPU is idle and stopping it as soon as other compute
processes are detected.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

LOG = logging.getLogger("gpu-idle-manager")


def run_command(cmd: List[str]) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except FileNotFoundError as exc:
        LOG.error("Failed to run %s: %s", cmd[0], exc)
        return None


def get_gpu_compute_pids() -> List[int]:
    """Return list of PIDs currently using the GPU for compute."""
    result = run_command([
        "nvidia-smi",
        "--query-compute-apps=pid",
        "--format=csv,noheader"
    ])
    if result is None:
        return []
    if result.returncode != 0:
        LOG.error("nvidia-smi failed: %s", result.stderr.strip())
        return []
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            LOG.debug("Unable to parse pid from line: %s", line)
    return pids


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


def start_dummy_process(exe: Path) -> subprocess.Popen:
    LOG.info("Starting dummy GPU workload: %s", exe)
    return subprocess.Popen([str(exe)])


def manage_gpu_idle(dummy_exe: Path, poll_interval: float) -> None:
    """Continuously poll GPU activity and manage the dummy workload accordingly."""
    dummy_proc: Optional[subprocess.Popen] = None
    stopping = False

    def handle_signal(signum, frame):
        nonlocal stopping
        stopping = True
        LOG.info("Received signal %s, shutting down", signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while not stopping:
        active_pids = get_gpu_compute_pids()
        dummy_pid = dummy_proc.pid if dummy_proc else None
        external_pids = [pid for pid in active_pids if pid != dummy_pid]

        if external_pids:
            if dummy_proc and dummy_proc.poll() is None:
                LOG.info(
                    "External GPU usage detected (PIDs: %s). Stopping dummy workload.",
                    ", ".join(map(str, external_pids)) or "unknown",
                )
                dummy_proc.terminate()
                try:
                    dummy_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    LOG.warning("Dummy workload did not exit gracefully, killing.")
                    dummy_proc.kill()
                dummy_proc = None
        else:
            if not dummy_proc or dummy_proc.poll() is not None:
                dummy_proc = start_dummy_process(dummy_exe)

        time.sleep(poll_interval)

    if dummy_proc and dummy_proc.poll() is None:
        LOG.info("Shutting down dummy workload")
        dummy_proc.terminate()
        try:
            dummy_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            dummy_proc.kill()


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
