#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bootstrap_pyenv_local.py

Create a local venv, install deps (including llama-cpp-python by default), optionally fetch a
Qwen2.5 14B Instruct GGUF model, and provide handy 'create' / 'run' / 'pip' / 'doctor' / 'clean' commands.

Examples
--------
python bootstrap_pyenv_local.py clean
python bootstrap_pyenv_local.py create
python bootstrap_pyenv_local.py create --skip-model
python bootstrap_pyenv_local.py create --model-url https://example/my.gguf
python bootstrap_pyenv_local.py create --model-file D:\downloads\my.gguf
python bootstrap_pyenv_local.py pip install -r requirements.txt
python bootstrap_pyenv_local.py run -- src\load_config.py
python bootstrap_pyenv_local.py run -- -m src.company_index.create_integrated_search_file
python bootstrap_pyenv_local.py run -- -m src.company_index.company_tfidf_api
python bootstrap_pyenv_local.py run -- -m src.query_engine.enhanced_query_with_summary
python bootstrap_pyenv_local.py run -- .\test_pyenv_integrated_query_summary.py
python bootstrap_pyenv_local.py run -- .\test_pyenv_integrated_query_summary.py --query "location of Company_003"

Notes
-----
- No Hugging Face token is used (public downloads only).
- If model fetch fails, we WARN and continue (unless --strict-model).
- llama-cpp-python is installed by default. To override the spec, set env LLAMA_CPP_SPEC
  (e.g., path to a wheel) or place a wheel under ./wheels or ./vendor.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------- Config ----------------

DEFAULT_VENV_DIR = Path(".venv")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_MODEL_QUANT = "Q4_K_M"  # change with --quant

# Base packages (llama-cpp-python handled separately below)
BASE_PKGS = [
    "huggingface_hub>=0.23.0",
    "requests>=2.31.0",
]

# Official GGUF repo and mirrors we’ll search for consolidated single-file GGUFs
PRIMARY_REPO = "Qwen/Qwen2.5-14B-Instruct-GGUF"
CONSOLIDATED_REPOS: List[str] = [
    "Qwen/Qwen2.5-14B-Instruct-GGUF",      # official GGUF repo
    "bartowski/Qwen2.5-14B-Instruct-GGUF", # popular mirror of GGUF builds
    "llmware/qwen2.5-14b-instruct-gguf",   # another mirror
]

# ---------------- Small utils ----------------

def _p(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")

def repo_root() -> Path:
    return Path(__file__).resolve().parent

def venv_python_path(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

def venv_pip_path(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def find_exe(name: str) -> Optional[Path]:
    p = shutil.which(name)
    return Path(p) if p else None

def rmrf(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        _p("WARN", f"Failed to remove {path}: {e}")

# ---------------- Venv ----------------

def create_venv(venv_dir: Path = DEFAULT_VENV_DIR) -> Path:
    if not venv_dir.exists():
        _p("RUN", f'python -m venv "{venv_dir}"')
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        _p("OK", f"created venv at {venv_dir}")
    else:
        _p("INFO", f"venv already exists at {venv_dir}")
    return venv_dir

def pip_install(pip_exe: Path, pkgs: List[str]) -> None:
    if not pkgs:
        _p("INFO", "No base packages requested for install.")
        return
    _p("RUN", f'"{pip_exe}" install ' + " ".join(pkgs))
    subprocess.run([str(pip_exe), "install", *pkgs], check=True)
    _p("OK", "Base packages installed into venv.")

def pip_install_spec(pip_exe: Path, spec: str) -> None:
    spec = spec.strip()
    if spec:
        _p("RUN", f'"{pip_exe}" install {spec}')
        subprocess.run([str(pip_exe), "install", spec], check=True)
        _p("OK", f"Installed: {spec}")

def verify_llama_import(python_exe: Path) -> None:
    _p("RUN", f'"{python_exe}" -c "import llama_cpp,sys;print(\'llama_cpp OK, py=\',sys.version)"')
    subprocess.run([str(python_exe), "-c",
                   "import llama_cpp,sys;print('llama_cpp OK, py=',sys.version)"],
                   check=True)

# ---------------- venv site-packages injection ----------------

def get_venv_site_packages(python_exe: Path) -> Optional[Path]:
    """Ask the venv’s Python where its purelib/site-packages is."""
    try:
        out = subprocess.run(
            [str(python_exe), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            capture_output=True, text=True, check=True
        )
        p = Path(out.stdout.strip())
        return p if p.exists() else None
    except Exception:
        return None

def inject_venv_site_packages(python_exe: Path) -> bool:
    """Put the venv’s site-packages on this process’s sys.path so imports work here."""
    sp = get_venv_site_packages(python_exe)
    if not sp:
        return False
    if str(sp) not in sys.path:
        sys.path.insert(0, str(sp))
    return True

# ---------------- HF access (import on demand, after injection) ----------------

def _get_hf():
    """
    Import huggingface_hub lazily (after we've injected venv site-packages).
    Returns (snapshot_download, hf_hub_download, list_repo_files) or (None, None, None).
    """
    try:
        from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files  # type: ignore
        return snapshot_download, hf_hub_download, list_repo_files
    except Exception:
        return None, None, None

# ---------------- Model helpers ----------------

def quant_to_names(quant: str) -> Tuple[str, str, str]:
    """
    Return:
      include_glob: pattern for snapshot_download (lowercase)
      split_glob:   glob to find split parts (lowercase)
      strict_name:  final merged filename (CamelCase)
    """
    q = quant.strip()
    lc = q.lower()  # q4_k_m
    include_glob = f"qwen2.5-14b-instruct-{lc}*.gguf"
    split_glob = f"qwen2.5-14b-instruct-{lc}-*-of-*.gguf"
    strict = f"Qwen2.5-14B-Instruct-{q}.gguf"
    return include_glob, split_glob, strict

def consolidated_filename(quant: str) -> str:
    return f"Qwen2.5-14B-Instruct-{quant}.gguf"

def try_download_consolidated_from_repos(models_dir: Path, quant: str) -> bool:
    """
    Try to find and download the exact single-file GGUF using list_repo_files + hf_hub_download.
    Search official repo first, then mirrors. Public only; no token.
    """
    snapshot_download, hf_hub_download, list_repo_files = _get_hf()
    if not (hf_hub_download and list_repo_files):
        _p("WARN", "huggingface_hub not available; skipping consolidated file search.")
        return False

    exact_name = consolidated_filename(quant)
    for repo in CONSOLIDATED_REPOS:
        _p("INFO", f"Searching consolidated GGUF in repo: {repo}")
        try:
            files = list_repo_files(repo_id=repo, repo_type="model")
        except Exception as e:
            _p("WARN", f"Could not list files for {repo}: {e}")
            continue

        if not any(f.lower() == exact_name.lower() for f in files):
            _p("WARN", f"No exact file in {repo}: {exact_name}")
            continue

        _p("INFO", f"Found {exact_name} in {repo}; downloading...")
        try:
            local_path = hf_hub_download(
                repo_id=repo,
                filename=exact_name,
                local_dir=str(models_dir),
                local_dir_use_symlinks=False,
                revision="main",
            )
            target = models_dir / exact_name
            Path(local_path).replace(target)
            if target.exists() and target.stat().st_size > 0:
                _p("OK", f"Downloaded consolidated model -> {target}")
                return True
        except Exception as e:
            _p("WARN", f"Download failed from {repo}: {e}")

    return False

def merge_split_parts_with_tool(first_part: Path, merged_out: Path) -> bool:
    tool = find_exe("llama-gguf-split") or find_exe("llama-gguf-split.exe")
    if not tool:
        return False
    _p("INFO", f"Merging split GGUF via tool -> {merged_out.name}")
    subprocess.run([str(tool), "--merge", str(first_part), str(merged_out)], check=True)
    return merged_out.exists() and merged_out.stat().st_size > 0

def merge_split_parts_python(parts: List[Path], merged_out: Path) -> bool:
    """
    Fallback merge by concatenation in correct order of -00001-of-0000N.
    """
    pat = re.compile(r".*-(\d+)-of-(\d+)\.gguf$", re.IGNORECASE)

    def idx(p: Path) -> int:
        m = pat.match(p.name)
        return int(m.group(1)) if m else 10**9

    parts_sorted = sorted(parts, key=idx)
    if not parts_sorted:
        return False

    _p("INFO", f"Merging {len(parts_sorted)} split parts (Python fallback) -> {merged_out.name}")
    with open(merged_out, "wb") as out:
        for shard in parts_sorted:
            with open(shard, "rb") as inp:
                shutil.copyfileobj(inp, out)
    return merged_out.exists() and merged_out.stat().st_size > 0

def try_download_split_and_merge(models_dir: Path, quant: str) -> bool:
    """
    Download split shards via snapshot_download from the official repo,
    then merge them into a single GGUF.
    """
    snapshot_download, hf_hub_download, list_repo_files = _get_hf()
    if not snapshot_download:
        _p("WARN", "huggingface_hub not available; cannot fetch split shards.")
        return False

    include_glob, split_glob, strict_name = quant_to_names(quant)
    target = models_dir / strict_name
    try:
        _p("INFO", f"snapshot_download {PRIMARY_REPO} :: {include_glob}")
        local_root = snapshot_download(
            repo_id=PRIMARY_REPO,
            allow_patterns=[include_glob],
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
            revision="main",
        )
        parts = sorted(Path(local_root).glob(split_glob))
        if not parts:
            _p("WARN", f"No split parts matched: {split_glob}")
            return False
        first = next((p for p in parts if "-00001-of-" in p.name), parts[0])
        if merge_split_parts_with_tool(first, target) or merge_split_parts_python(parts, target):
            _p("OK", f"Merged split parts -> {target}")
            return True
        _p("WARN", "Split parts downloaded but merge step did not produce a file.")
        return False
    except Exception as e:
        _p("WARN", f"snapshot_download failed: {e}")
        return False

def try_download_from_url_or_file(
    models_dir: Path,
    quant: str,
    *,
    model_url: Optional[str],
    model_file: Optional[str | Path],
) -> bool:
    """
    Handle direct URL or local file options. Returns True on success.
    """
    _, _, strict_name = quant_to_names(quant)
    target = models_dir / strict_name

    # Local file copy
    if model_file:
        src = Path(model_file)
        if src.exists() and src.is_file():
            _p("INFO", f"Copying local model file -> {target.name}")
            shutil.copy2(src, target)
            if target.exists() and target.stat().st_size > 0:
                _p("OK", f"Copied local model -> {target}")
                return True
            _p("WARN", "Local model copy did not materialize; continuing...")
        else:
            _p("WARN", f"--model-file not found: {src}")

    # Direct URL
    if model_url:
        try:
            import requests
            _p("INFO", f"Downloading model from URL -> {target.name}")
            with requests.get(model_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            if target.exists() and target.stat().st_size > 0:
                _p("OK", f"Downloaded model from URL -> {target}")
                return True
        except Exception as e:
            _p("WARN", f"Direct URL download failed: {e}")

    return False

def ensure_model(
    models_dir: Path,
    quant: str = DEFAULT_MODEL_QUANT,
    *,
    model_url: Optional[str] = None,
    model_file: Optional[str | Path] = None,
) -> bool:
    """
    Ensure models/Qwen2.5-14B-Instruct-{quant}.gguf exists.

    Priority (public only):
      1) --model-file / --model-url (if provided)
      2) Consolidated single-file GGUF from repos (official first, then mirrors)
      3) Split shards from official → merge

    Returns True if present/created, False if not obtainable.
    """
    ensure_dir(models_dir)
    _, _, strict_name = quant_to_names(quant)
    target = models_dir / strict_name

    if target.exists() and target.stat().st_size > 0:
        _p("OK", f"Model already present: {target}")
        return True

    # 1) explicit sources
    if try_download_from_url_or_file(models_dir, quant, model_url=model_url, model_file=model_file):
        return True

    # 2) consolidated single-file
    if try_download_consolidated_from_repos(models_dir, quant):
        return True

    # 3) split shards + merge
    if try_download_split_and_merge(models_dir, quant):
        return True

    return False

# ---------------- llama-cpp install (default ON) ----------------

def _find_local_llama_wheel(search_roots: List[Path]) -> Optional[Path]:
    """
    Look for a prebuilt llama-cpp-python wheel in common folders (Windows-friendly).
    """
    patterns = [
        "llama_cpp_python-*.whl",
        "llama-cpp-python-*.whl",
    ]
    for root in search_roots:
        if not root or not root.exists():
            continue
        for pat in patterns:
            # prefer the latest by name sort descending
            matches = sorted(root.glob(pat), reverse=True)
            if matches:
                return matches[0]
    return None

def install_llama_default(pip_exe: Path, python_exe: Path, strict: bool = False) -> bool:
    """
    Install llama-cpp-python by default using this priority:
      1) LLAMA_CPP_SPEC env var (pip spec or absolute wheel path)
      2) local wheel under ./wheels or ./vendor or repo root
      3) requirements.txt (repo root)
      4) pip install llama-cpp-python (no pin)
    Returns True if import succeeds, False otherwise (unless strict → raises).
    """
    repo = repo_root()
    env_spec = os.getenv("LLAMA_CPP_SPEC", "").strip()

    # 1) explicit env spec (wheel or pip spec)
    try:
        if env_spec:
            _p("INFO", f"LLAMA_CPP_SPEC set → installing: {env_spec}")
            pip_install_spec(pip_exe, env_spec)
            verify_llama_import(python_exe)
            return True
    except Exception as e:
        _p("WARN", f"Env spec install/import failed: {e}")
        if strict:
            raise

    # 2) local wheel
    try:
        wheel = _find_local_llama_wheel([repo / "wheels", repo / "vendor", repo])
        if wheel:
            _p("INFO", f"Found local llama-cpp-python wheel: {wheel.name}")
            pip_install_spec(pip_exe, str(wheel))
            verify_llama_import(python_exe)
            return True
    except Exception as e:
        _p("WARN", f"Local wheel install/import failed: {e}")
        if strict:
            raise

    # 3) requirements.txt
    try:
        req = repo / "requirements.txt"
        if req.exists():
            _p("INFO", "Installing from requirements.txt")
            pip_install_spec(pip_exe, f"-r {req}")
            verify_llama_import(python_exe)
            return True
    except Exception as e:
        _p("WARN", f"requirements.txt install/import failed: {e}")
        if strict:
            raise

    # 4) plain pip install
    try:
        _p("INFO", "Installing llama-cpp-python from PyPI (no version pin)")
        pip_install_spec(pip_exe, "llama-cpp-python")
        verify_llama_import(python_exe)
        return True
    except Exception as e:
        _p("WARN", f"pip install llama-cpp-python failed or import failed: {e}")
        if strict:
            raise
        return False

# ---------------- Commands ----------------

def cmd_create(args: argparse.Namespace) -> int:
    venv_dir = create_venv(DEFAULT_VENV_DIR)
    pip_exe = venv_pip_path(venv_dir)
    python_exe = venv_python_path(venv_dir)

    # Install base deps first
    try:
        pip_install(pip_exe, BASE_PKGS)
    except subprocess.CalledProcessError as e:
        _p("ERROR", f"Base package install failed: {e}")
        return 1

    # Make the venv’s site-packages importable in THIS process so huggingface_hub is available
    if not inject_venv_site_packages(python_exe):
        _p("WARN", "Could not inject venv site-packages; huggingface downloads may not work.")

    # Install llama-cpp-python by default (unless --no-llama)
    if not args.no_llama:
        ok = install_llama_default(pip_exe, python_exe, strict=args.strict_llama)
        if not ok:
            _p("WARN", "llama-cpp-python was not installed or import failed.")
    else:
        _p("INFO", "Skipping llama-cpp-python install (--no-llama).")

    return 0

    # Model step
    models_dir = ensure_dir(Path(args.models_dir))
    if args.skip_model:
        _p("INFO", "Skipping model download (--skip-model).")
        return 0

    ok = ensure_model(
        models_dir,
        quant=args.quant,
        model_url=args.model_url,
        model_file=args.model_file,
    )

    if not ok:
        msg = f"Model not obtained. Please place it at: {models_dir / consolidated_filename(args.quant)}"
        if args.strict_model:
            _p("ERROR", msg)
            return 2
        else:
            _p("WARN", msg)
            _p("WARN", "Proceeding without model download; install step completed.")
    return 0

def cmd_run(args: argparse.Namespace) -> int:
    venv_dir = DEFAULT_VENV_DIR
    python_exe = venv_python_path(venv_dir)
    if not python_exe.exists():
        _p("ERROR", f"Venv not found at {venv_dir}. Run 'create' first.")
        return 1

    # Ensure repo root on PYTHONPATH so 'import src' works with script paths or -m
    env = os.environ.copy()
    repo = repo_root()
    env["PYTHONPATH"] = (env.get("PYTHONPATH", "") + os.pathsep + str(repo)).strip(os.pathsep)

    forward = args.args or []
    # Allow both `run -- ...` and `run ...`
    if forward and forward[0] == "--":
        forward = forward[1:]
    if not forward:
        _p("ERROR", "Nothing to run. Example: run -- -m src.company_index.create_integrated_search_file")
        return 2

    _p("RUN", f'"{python_exe}" ' + " ".join(forward))
    try:
        subprocess.run([str(python_exe), *forward], cwd=str(repo), env=env, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        return e.returncode

def cmd_pip(args: argparse.Namespace) -> int:
    """Run venv's pip with the provided arguments (after optional `--`)."""
    venv_dir = DEFAULT_VENV_DIR
    pip_exe = venv_pip_path(venv_dir)
    if not pip_exe.exists():
        _p("ERROR", f"Venv not found at {venv_dir}. Run 'create' first.")
        return 1

    forward = args.args or []
    # Allow both `pip -- ...` and `pip ...`
    if forward and forward[0] == "--":
        forward = forward[1:]
    if not forward:
        _p("ERROR", "Nothing to pass to pip. Example: pip install -r requirements.txt")
        return 2

    _p("RUN", f'"{pip_exe}" ' + " ".join(forward))
    try:
        subprocess.run([str(pip_exe), *forward], cwd=str(repo_root()), check=True)
        return 0
    except subprocess.CalledProcessError as e:
        return e.returncode

def cmd_doctor(_: argparse.Namespace) -> int:
    print("=== Doctor ===")
    print(f"Python: {sys.version}")
    print(f"Repo root: {repo_root()}")
    print(f"Venv: {DEFAULT_VENV_DIR}")
    print(f"Models dir: {DEFAULT_MODELS_DIR}")
    py = venv_python_path(DEFAULT_VENV_DIR)
    print(f"Venv python: {py} | exists={py.exists()}")
    return 0

def cmd_clean(args: argparse.Namespace) -> int:
    # by default, wipe venv + models; optional flags to include HF caches
    if args.venv:
        _p("INFO", f"Removing venv: {DEFAULT_VENV_DIR}")
        rmrf(DEFAULT_VENV_DIR)
    if args.models:
        _p("INFO", f"Removing models dir: {DEFAULT_MODELS_DIR}")
        rmrf(DEFAULT_MODELS_DIR)
    if args.hf_cache:
        # common HF cache locations
        home = Path.home()
        candidates = [
            home / ".cache" / "huggingface",
            home / "AppData" / "Local" / "huggingface" if os.name == "nt" else None,
        ]
        for c in filter(None, candidates):
            if c.exists():
                _p("INFO", f"Removing HF cache: {c}")
                rmrf(c)
    return 0

# ---------------- CLI ----------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap local venv + model and run tasks")
    sub = p.add_subparsers(dest="cmd", required=True)

    # create
    pc = sub.add_parser("create", help="Create venv, install deps, install llama-cpp-python, optionally fetch model")
    pc.add_argument("--quant", default=DEFAULT_MODEL_QUANT, help="Quantization (default: Q4_K_M)")
    pc.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="Models directory (default: models)")
    pc.add_argument("--skip-model", action="store_true", help="Do not download a model")
    pc.add_argument("--model-url", default=None, help="Direct URL to a single-file GGUF to download")
    pc.add_argument("--model-file", default=None, help="Path to a local single-file GGUF to copy")
    pc.add_argument("--strict-model", action="store_true", help="Fail if the model cannot be obtained")
    pc.add_argument("--no-llama", action="store_true", help="Skip llama-cpp-python install (default: install)")
    pc.add_argument("--strict-llama", action="store_true", help="Fail if llama-cpp-python install/import fails")
    pc.set_defaults(func=cmd_create)

    # run
    pr = sub.add_parser("run", help="Run a Python command inside venv (adds repo root to PYTHONPATH)")
    pr.add_argument("args", nargs=argparse.REMAINDER, help="Args to pass to python (prefix with --)")
    pr.set_defaults(func=cmd_run)

    # pip
    pp = sub.add_parser("pip", help="Run venv pip with your arguments (prefix with -- optional)")
    pp.add_argument("args", nargs=argparse.REMAINDER, help="Args to pass to pip, e.g. install -r requirements.txt")
    pp.set_defaults(func=cmd_pip)

    # doctor
    pd = sub.add_parser("doctor", help="Print basic env info")
    pd.set_defaults(func=cmd_doctor)

    # clean
    pcl = sub.add_parser("clean", help="Remove venv/models (and optionally HF caches)")
    pcl.add_argument("--venv", action="store_true", default=True, help="Remove the virtualenv (default: true)")
    pcl.add_argument("--models", action="store_true", default=True, help="Remove the models dir (default: true)")
    pcl.add_argument("--hf-cache", action="store_true", default=False, help="Also remove local Hugging Face caches")
    pcl.set_defaults(func=cmd_clean)

    return p.parse_args()

def main() -> int:
    args = _parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())