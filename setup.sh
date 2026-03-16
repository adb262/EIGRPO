apt-get update
apt-get install -y curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
uv pip install -e .
uv pip install torch==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126

uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"