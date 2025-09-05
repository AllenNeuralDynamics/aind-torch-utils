# aind-torch-utils

Generic, queue-based, multi-GPU PyTorch inference pipeline.

## Features
- Block/patch based volume processing with optional overlap
- Multi-threaded preparation & writing stages
- Multi-GPU inference workers
- Flexible blending/trim seam handling
- Queue & system monitoring hooks

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e . --group dev
```
Note: --group flag is available only in pip versions >=25.1

Alternatively, if using `uv`, run
```bash
uv sync
```
