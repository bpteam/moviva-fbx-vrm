"""SkinTokens / TokenRig (MIT, https://github.com/VAST-AI-Research/SkinTokens): AI skeleton and skin weights on an
NVIDIA GPU, installed on demand into the user's data folder and run as its own processes (Python 3.11, CUDA
PyTorch), never inside this program.

    install   clone the pinned commit, a private venv by ``uv``, CUDA PyTorch, its requirements, a prebuilt
              flash-attn wheel (no CUDA toolkit needed), and its checkpoints (``download.py --model``)
    rig       its ``demo.py --use_transfer`` (its ``bpy_server`` started first): the source mesh, textures and
              scale kept, a skeleton and weights added; the converter maps the bones afterwards
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

COMMIT = "273b691d35989d71cd17ff2895fdc735097b92d1"
REPO = "https://github.com/VAST-AI-Research/SkinTokens"
TORCH = ["torch==2.7.0", "torchvision==0.22.0", "--index-url", "https://download.pytorch.org/whl/cu128"]
FLASH_ATTN = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
              "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl")
EXTRA = ["scipy"]  # imported by its code, missing from its requirements.txt
CHECKPOINTS = ("experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
               "experiments/skin_vae_2_10_32768/last.ckpt")
MIN_VRAM_MB = 14_000
TIMEOUT = 1800
BPY_SERVER = "http://localhost:59876"  # BPY_PORT in its src/server/spec.py at COMMIT


def data_root() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Moviva" / "AI"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Moviva" / "AI"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "moviva" / "ai"


def home() -> Path:
    return data_root() / "engines" / "skintokens"


def python() -> Path:
    return home() / ".venv" / "bin" / "python"


def gpus() -> list[tuple[int, str, int]]:
    """(PCI-order index, name, MiB) of every NVIDIA GPU, from nvidia-smi; [] without one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        i, name, mem = (s.strip() for s in line.split(","))
        rows.append((int(i), name, int(float(mem))))
    return rows


def pick_gpu() -> tuple[int, str, int] | None:
    """The largest GPU, when it has the VRAM SkinTokens needs."""
    big = max(gpus(), key=lambda g: g[2], default=None)
    return big if big and big[2] >= MIN_VRAM_MB else None


def status() -> dict:
    """{"state": READY | NOT_INSTALLED | UNAVAILABLE, "reason", "gpu"}."""
    if not sys.platform.startswith("linux"):
        return {"state": "UNAVAILABLE", "reason": "the prebuilt CUDA wheels used here are Linux x64 only"}
    gpu = pick_gpu()
    if gpu is None:
        return {"state": "UNAVAILABLE", "reason": f"needs an NVIDIA GPU with {MIN_VRAM_MB // 1000} GB VRAM",
                "gpus": gpus()}
    ready = python().is_file() and all((home() / c).is_file() for c in CHECKPOINTS)
    return {"state": "READY" if ready else "NOT_INSTALLED", "gpu": {"index": gpu[0], "name": gpu[1], "mib": gpu[2]}}


def install(log=print) -> None:
    """Everything above, in steps that can be rerun: a finished step is skipped. Needs ``uv`` and ``git``."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is needed to install SkinTokens (https://docs.astral.sh/uv/)")
    h = home()
    h.parent.mkdir(parents=True, exist_ok=True)
    if not (h / "demo.py").is_file():
        log("Downloading SkinTokens source")
        tmp = h.with_name("skintokens.partial")
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(["git", "clone", "-q", REPO, str(tmp)], check=True)
        subprocess.run(["git", "-C", str(tmp), "checkout", "-q", COMMIT], check=True)
        tmp.rename(h)
    py = python()
    pip = [uv, "pip", "install", "--python", str(py)]
    if not py.is_file():
        log("Preparing Python 3.11")
        subprocess.run([uv, "venv", "--python", "3.11", str(h / ".venv")], check=True)
    log("Installing PyTorch (CUDA) and dependencies")
    subprocess.run(pip + TORCH, check=True)
    subprocess.run(pip + ["-r", str(h / "requirements.txt"), *EXTRA, FLASH_ATTN], check=True)
    if not all((h / c).is_file() for c in CHECKPOINTS):
        log("Downloading model weights")
        subprocess.run([str(py), "download.py", "--model"], cwd=h, check=True)
    log("Verifying")
    subprocess.run([str(py), "-c", "import torch, flash_attn; assert torch.cuda.is_available()"], check=True)


def _env(gpu: int) -> dict:
    """Its own Python and libraries only: a Blender or frozen-app parent's PYTHONHOME / PYTHONPATH and library path
    (snap Blender's libusd, PyInstaller's bundle) break the 3.11 venv and its bpy."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "LD_LIBRARY_PATH"}
    if os.environ.get("LD_LIBRARY_PATH_ORIG"):  # PyInstaller keeps the user's own here
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH_ORIG"]
    return dict(env, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))


def _ping() -> bool:
    try:
        urllib.request.urlopen(BPY_SERVER + "/ping", timeout=2)
        return True
    except OSError:
        return False


def _bpy_server(gpu: int) -> subprocess.Popen | None:
    """Its Blender helper, started and awaited here: demo.py starts one too but waits only 30 s, less than its cold
    start (importing torch); the copy demo.py starts then finds the port taken and quits."""
    if _ping():
        return None
    logs = data_root() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / "skintokens-bpy-server.log", "wb") as out:
        proc = subprocess.Popen([str(python()), "bpy_server.py"], cwd=home(), env=_env(gpu), stdout=out,
                                stderr=subprocess.STDOUT)
    for _ in range(300):
        if _ping():
            return proc
        if proc.poll() is not None:
            break
        time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"SkinTokens' bpy_server did not start (log: {logs / 'skintokens-bpy-server.log'})")


def rig(source: Path, output: Path, log=print) -> None:
    """``source`` rigged by SkinTokens into ``output`` (GLB)."""
    st = status()
    if st["state"] != "READY":
        raise RuntimeError(f"SkinTokens is {st['state']}: {st.get('reason', 'run: character-vrm ai install')}")
    gpu = st["gpu"]["index"]
    log(f"SkinTokens on GPU {gpu} ({st['gpu']['name']})")
    server = _bpy_server(gpu)
    try:
        proc = subprocess.run([str(python()), "demo.py", "--input", str(source), "--output", str(output),
                               "--use_transfer"], cwd=home(), env=_env(gpu), capture_output=True, text=True,
                              timeout=TIMEOUT)
    finally:
        if server is not None:
            server.terminate()
    if proc.returncode != 0 or not output.is_file():
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        raise RuntimeError("SkinTokens failed: " + " | ".join(tail))
