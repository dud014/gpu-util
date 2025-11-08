gpu-util
========

This project provides a GPU idle manager that keeps NVIDIA GPUs busy with a dummy CUDA workload when they would otherwise be idle. The manager runs continuously, polling `nvidia-smi` at a configurable interval, and stops the dummy workload as soon as another process starts using the GPU. When the external workload finishes, the manager automatically relaunches the dummy workload so the GPU utilisation stays elevated during the next idle window.

## Components

- `src/dummy_spin.cu`: CUDA program that launches a simple compute kernel repeatedly to keep the GPU utilized.
- `src/gpu_idle_manager.py`: Python controller that compiles the CUDA program (using `nvcc`) and starts/stops it based on current GPU usage.

## Usage

1. Ensure the CUDA toolkit (`nvcc`) and NVIDIA drivers are installed and that `nvidia-smi` is available in `$PATH`.
2. Run the idle manager:

   ```bash
   python3 src/gpu_idle_manager.py --poll-interval 2.0
   ```

   The script compiles the CUDA dummy workload into `build/dummy_spin` on first run. It then starts the workload whenever the GPU is idle and terminates it when external GPU usage is detected.

You can adjust the polling interval, build directory, and CUDA source path using the command-line options. Use `--log-level DEBUG` for more detailed logging.
