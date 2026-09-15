#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot environment bootstrap for the FIR-TN capstone.
#
# Runs INSIDE WSL2 (Ubuntu). Rationale (IMPLEMENTATION.md §1): Smart App Control
# on the Windows host blocks pyarrow, which HF `datasets` and `bitsandbytes`
# depend on. A Linux userland sidesteps that entirely while still being 100%
# local -- no hosted APIs anywhere in this project.
#
# The venv lives on WSL's native ext4 filesystem (~/.venvs), NOT on /mnt/c.
# Python imports over the 9p mount are an order of magnitude slower.
#
#   usage:  bash environment/setup_wsl.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${FIR_VENV:-$HOME/.venvs/fir-tn}"
PY_VERSION="3.11"

echo "==> repo:  $REPO_ROOT"
echo "==> venv:  $VENV_DIR"

# --- 1. uv (user-local, no sudo) ------------------------------------------
# Ubuntu 24.04 ships Python 3.12 and this box has no passwordless sudo, so we
# cannot apt-install 3.11. uv fetches a standalone 3.11 build into ~/.local.
if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# --- 2. Python 3.11 + venv -------------------------------------------------
echo "==> provisioning Python $PY_VERSION"
uv python install "$PY_VERSION"
uv venv --python "$PY_VERSION" "$VENV_DIR"

# --- 3. torch, from the CUDA 12.8 index ------------------------------------
# This machine's GPU is an RTX 5050 Laptop (Blackwell, sm_120). Wheels built
# against CUDA 12.1 have no sm_120 kernels and fail at runtime.
echo "==> installing torch (cu128)"
VIRTUAL_ENV="$VENV_DIR" uv pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch torchaudio

# --- 4. everything else ----------------------------------------------------
echo "==> installing project requirements"
VIRTUAL_ENV="$VENV_DIR" uv pip install -r "$REPO_ROOT/environment/requirements.txt"

# --- 5. report -------------------------------------------------------------
echo "==> verifying"
"$VENV_DIR/bin/python" - <<'PY'
import platform, torch
print(f"python       : {platform.python_version()}")
print(f"torch        : {torch.__version__}")
print(f"cuda build   : {torch.version.cuda}")
print(f"cuda avail   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu          : {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
PY

echo
echo "==> done.  activate with:  source $VENV_DIR/bin/activate"
