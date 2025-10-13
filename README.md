# VLLM 0.11.0 for Pascal GPUs (CUDA 12.6)

This repository provides instructions and prebuilt wheels for installing **VLLM 0.11.0** with **Pascal GPU support** (e.g., GTX 1060, 1070, 1080, etc.) using **CUDA 12.6**.

---

## ✅ Requirements

- Debian 12 (or compatible)
- NVIDIA GPU with Pascal architecture
- CUDA 12.6 and NVIDIA drivers
- Miniconda or Anaconda
- Python 3.12

---

### 1. Install CUDA 12.6 and NVIDIA drivers

Follow the official guide:  
👉 [CUDA 12.6 Download Archive](https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=12&target_type=deb_network)

Or use this helpful guide for Debian 12:  
👉 [How to Install CUDA on Debian 12](https://greenwebpage.com/community/how-to-install-cuda-on-debian-12/)

### 2. Install Miniconda

👉 [https://www.anaconda.com/docs/getting-started/miniconda/main](https://www.anaconda.com/docs/getting-started/miniconda/main)

### 3. Create Conda Environment
```sh
conda create -n venv -c  conda-forge  git python=3.12

conda activate venv
```
### 4. Install Prebuilt Wheels
```sh
pip install https://github.com/ampir-nn/vllm-pascal/releases/download/wheels/vllm-0.11.0+pascal.cu126-cp312-cp312-linux_x86_64.whl

pip uninstall torch triton -y

pip install https://github.com/ampir-nn/vllm-pascal/releases/download/wheels/triton-3.4.0-cp312-cp312-linux_x86_64.whl

pip install https://github.com/ampir-nn/vllm-pascal/releases/download/wheels/torch-2.8.0a0+gitba56102-cp312-cp312-linux_x86_64.whl
```
At the end of the torch/triton installation, the installer will complain about dependencies — just ignore it.

### 5. Install NCCL Libraries
```sh
sudo apt install libnccl2_2.28.3-1+cuda12.6_amd64 libnccl-dev_2.28.3-1+cuda12.6_amd64
```
### 6. Running Models
### 3-GPU Setup (Pipeline Parallelism)
```sh
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

vllm serve jart25/Qwen3-Coder-30B-A3B-Instruct-Int4-gptq \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 3 \
  --max-num-seqs 1 \
  --max-model-len 4096 \
  --dtype float16 \
  --quantization gptq \
  --gpu-memory-utilization 0.95 \
  --swap-space 0 \
  --cpu-offload-gb 0 \
  --enable-expert-parallel \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```
### 2-GPU Setup (Tensor Parallelism)
```sh
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

vllm serve jart25/Qwen3-Coder-30B-A3B-Instruct-Int4-gptq \
  --tensor-parallel-size 2 \
  --max-num-seqs 1 \
  --max-model-len 4096 \
  --dtype float16 \
  --quantization gptq \
  --gpu-memory-utilization 0.95 \
  --swap-space 0 \
  --cpu-offload-gb 0 \
  --enable-expert-parallel \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```
### GGUF Model on 2 GPUs
```sh
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

vllm serve ./Qwen3-14B-Q5_K_M.gguf \
  --tensor-parallel-size 2 \
  --max-num-seqs 1 \
  --max-model-len 16384 \
  --max-num-batched-tokens 16384 \
  --dtype float16 \
  --quantization gguf \
  --gpu-memory-utilization 0.95 \
  --swap-space 0 \
  --cpu-offload-gb 0
```

### Notes
    This setup is specific to Pascal GPUs and CUDA 12.6
    Do not use with newer GPUs (Turing/Ampere/Ada) — use standard VLLM instead
    Built for Python 3.12 and Debian 12
