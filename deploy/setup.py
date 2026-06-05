#!/usr/bin/env python3
"""Interactive one-shot installer for the Lead Priority Engine service.

Walks a fresh-clone reviewer through the four steps that turn this repo into
a running ``http://localhost:8000`` API:

1. Ensure ``.env`` exists (copy from ``.env.example`` if missing).
2. Prompt for ``OPEN_ROUTER_API_KEY`` if it is empty or malformed.
3. Pick a deploy mode (Docker container OR Python virtualenv + uvicorn).
4. Smoke-test ``/healthz`` and ``/score`` with the tracked example payload.

Stdlib-only on purpose: the user has not yet installed our requirements at
the time they run this script. If something fails, the script points at
``docs/5_fastapi_serving_and_deployment.docx §8`` (failure-mode table) so the
user can recover manually instead of debugging the installer.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
EXAMPLE_PAYLOAD_PATH = REPO_ROOT / "examples" / "score_request.json"
DEPLOYMENT_DOC = "docs/5_fastapi_serving_and_deployment.docx"

OPENROUTER_KEYS_URL = "https://openrouter.ai/keys"
OPENROUTER_KEY_PREFIX = "sk-or-"
KEY_PROMPT_MAX_ATTEMPTS = 3

DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
HEALTH_POLL_TIMEOUT_SECONDS = 60
HEALTH_POLL_INTERVAL_SECONDS = 1.0

DOCKER_IMAGE_TAG = "lead-priority:latest"
DOCKER_CONTAINER_NAME = "lead-priority"


# ---------------------------------------------------------------------------
# Printing helpers — small, no colour dependency.
# ---------------------------------------------------------------------------


def info(msg: str) -> None:
    print(f"[setup] {msg}")


def warn(msg: str) -> None:
    print(f"[setup] WARN: {msg}", file=sys.stderr)


def fail(msg: str) -> None:
    print(f"[setup] HATA: {msg}", file=sys.stderr)
    print(f"[setup] Elle kurulum için bakınız: {DEPLOYMENT_DOC} §8.", file=sys.stderr)


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


# ---------------------------------------------------------------------------
# .env management
# ---------------------------------------------------------------------------


def ensure_env_file() -> None:
    """Copy ``.env.example`` → ``.env`` if the latter does not exist."""
    if ENV_PATH.exists():
        info(f".env mevcut: {ENV_PATH}")
        return
    if not ENV_EXAMPLE_PATH.exists():
        fail(f"{ENV_EXAMPLE_PATH} bulunamadı; repo eksik klonlanmış olabilir.")
        sys.exit(2)
    shutil.copyfile(ENV_EXAMPLE_PATH, ENV_PATH)
    info(f".env oluşturuldu (.env.example'dan kopyalandı): {ENV_PATH}")


def read_env(path: Path) -> dict[str, str]:
    """Return parsed KEY=VALUE pairs (ignoring comments / blank lines)."""
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()
    return result


def update_env_value(path: Path, key: str, value: str) -> None:
    """Set ``key=value`` in-place, preserving comments + ordering."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out: list[str] = []
    replaced = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if (
            stripped
            and not stripped.startswith("#")
            and "=" in stripped
            and stripped.split("=", 1)[0].strip() == key
        ):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(raw_line)
    if not replaced:
        out.append(f"{key}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def ensure_openrouter_key() -> None:
    """Validate ``OPEN_ROUTER_API_KEY``; prompt interactively if missing."""
    env = read_env(ENV_PATH)
    current = env.get("OPEN_ROUTER_API_KEY", "").strip()
    if current and current.startswith(OPENROUTER_KEY_PREFIX):
        info(f"OPEN_ROUTER_API_KEY mevcut ({OPENROUTER_KEY_PREFIX}…).")
        return

    section("OpenRouter API key gerekli")
    print(f"  1. Tarayıcıdan {OPENROUTER_KEYS_URL} adresine git.")
    print("  2. Ücretsiz hesap aç (e-posta yeterli, kredi kartı opsiyonel).")
    print("  3. 'Create Key' → key'i kopyala (formatı 'sk-or-...').")
    print("  4. Aşağıya yapıştır. Yazdığın görünmeyecek (gizli giriş).")
    print()

    for attempt in range(1, KEY_PROMPT_MAX_ATTEMPTS + 1):
        try:
            key = getpass.getpass("OPEN_ROUTER_API_KEY: ").strip()
        except (EOFError, KeyboardInterrupt):
            fail("Key girişi iptal edildi.")
            sys.exit(2)
        if not key:
            warn(f"Boş key kabul edilmez. ({attempt}/{KEY_PROMPT_MAX_ATTEMPTS})")
            continue
        if not key.startswith(OPENROUTER_KEY_PREFIX):
            warn(
                f"Key '{OPENROUTER_KEY_PREFIX}' ile başlamalı. "
                f"({attempt}/{KEY_PROMPT_MAX_ATTEMPTS})"
            )
            continue
        update_env_value(ENV_PATH, "OPEN_ROUTER_API_KEY", key)
        info(".env güncellendi.")
        return
    fail("Üç denemede geçerli key alınamadı.")
    sys.exit(2)


# ---------------------------------------------------------------------------
# Mode selection + deploy
# ---------------------------------------------------------------------------


def choose_mode(forced: str | None) -> str:
    """Return ``"docker"`` or ``"venv"``."""
    if forced in {"docker", "venv"}:
        info(f"Mod parametre ile sabitlendi: {forced}")
        return forced
    section("Çalıştırma modu seçimi")
    print("  [D] Docker container (önerilen, izole)")
    print("  [P] Python virtualenv + uvicorn (lokal .venv)")
    while True:
        try:
            choice = input("Seçim (D/P) [D]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            fail("Mod seçimi iptal edildi.")
            sys.exit(2)
        if choice in {"", "d", "docker"}:
            return "docker"
        if choice in {"p", "python", "venv"}:
            return "venv"
        warn("Geçersiz seçim — D veya P giriniz.")


def run_command(cmd: list[str], *, cwd: Path | None = None) -> int:
    info("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=False).returncode


def run_docker() -> None:
    """Build the image and start the container."""
    section("Docker modu")
    if shutil.which("docker") is None:
        fail("`docker` komutu bulunamadı. Docker Desktop kurulu mu?")
        sys.exit(3)
    info("Docker daemon erişilebilir mi kontrol ediliyor…")
    code = subprocess.run(["docker", "info"], capture_output=True, check=False).returncode
    if code != 0:
        fail("Docker daemon yanıt vermiyor. Docker Desktop'ı başlatın.")
        sys.exit(3)

    # Eski container'ı temizle (varsa).
    subprocess.run(
        ["docker", "rm", "-f", DOCKER_CONTAINER_NAME],
        capture_output=True,
        check=False,
    )

    code = run_command(["docker", "build", "-t", DOCKER_IMAGE_TAG, "."], cwd=REPO_ROOT)
    if code != 0:
        fail("docker build başarısız.")
        sys.exit(4)

    code = run_command(
        [
            "docker",
            "run",
            "-d",
            "--name",
            DOCKER_CONTAINER_NAME,
            "--env-file",
            str(ENV_PATH),
            "-p",
            f"{DEFAULT_API_PORT}:{DEFAULT_API_PORT}",
            DOCKER_IMAGE_TAG,
        ],
        cwd=REPO_ROOT,
    )
    if code != 0:
        fail("docker run başarısız.")
        sys.exit(4)
    info(f"Container çalışıyor: {DOCKER_CONTAINER_NAME}")


def run_venv() -> None:
    """Create .venv, install deps, launch uvicorn in the background."""
    section("Python virtualenv modu")
    python = shutil.which("python3.12") or shutil.which("python3")
    if python is None:
        fail("python3.12 veya python3 PATH'te bulunamadı.")
        sys.exit(3)

    venv_dir = REPO_ROOT / ".venv"
    if not venv_dir.exists():
        code = run_command([python, "-m", "venv", str(venv_dir)])
        if code != 0:
            fail("virtualenv oluşturulamadı.")
            sys.exit(4)

    pip = venv_dir / "bin" / "pip"
    uvicorn = venv_dir / "bin" / "uvicorn"
    code = run_command([str(pip), "install", "-q", "-r", "requirements.txt"], cwd=REPO_ROOT)
    if code != 0:
        fail("pip install başarısız.")
        sys.exit(4)
    code = run_command([str(pip), "install", "-q", "-e", "."], cwd=REPO_ROOT)
    if code != 0:
        fail("pip install -e . başarısız.")
        sys.exit(4)

    log_path = REPO_ROOT / "uvicorn.log"
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            [
                str(uvicorn),
                "lead_priority.api.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(DEFAULT_API_PORT),
            ],
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    info(f"uvicorn arka planda çalışıyor (PID {process.pid}). Log: {log_path}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def wait_for_health() -> bool:
    """Poll ``/healthz`` until 200 or timeout."""
    url = f"http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}/healthz"
    deadline = time.monotonic() + HEALTH_POLL_TIMEOUT_SECONDS
    info(f"/healthz bekleniyor ({HEALTH_POLL_TIMEOUT_SECONDS}s timeout)…")
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(HEALTH_POLL_INTERVAL_SECONDS)
    return False


def post_example_score() -> bool:
    """POST ``examples/score_request.json`` to /score and pretty-print."""
    if not EXAMPLE_PAYLOAD_PATH.exists():
        warn(f"Örnek payload bulunamadı: {EXAMPLE_PAYLOAD_PATH}")
        return False
    payload = EXAMPLE_PAYLOAD_PATH.read_bytes()
    req = urllib.request.Request(
        f"http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}/score",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        warn(f"POST /score {exc.code}: {exc.read()[:200]!r}")
        return False
    except (urllib.error.URLError, OSError) as exc:
        warn(f"POST /score başarısız: {exc!r}")
        return False
    sentiment = body.get("sentiment") or {}
    sentiment_unavailable = bool(sentiment.get("sentiment_unavailable"))
    fallback_reason = sentiment.get("fallback_reason")
    info("POST /score 200 OK. Özet:")
    print(f"  p_conversion       : {body.get('p_conversion')}")
    print(f"  predicted_attitude : {sentiment.get('predicted_attitude')}")
    print(f"  sentiment_unavailable: {sentiment_unavailable}")
    print(f"  priority           : {body.get('priority')}")
    if sentiment_unavailable:
        warn(
            f"Sentiment fallback aktif (reason={fallback_reason}). "
            f"OPEN_ROUTER_API_KEY .env'de geçerli mi kontrol et veya "
            f":free model günlük quota dolmuş olabilir."
        )
        return False
    return True


def run_smoke_test() -> bool:
    section("Smoke test")
    if not wait_for_health():
        fail("/healthz timeout. Container/process loglarına bakın.")
        return False
    info("/healthz 200 OK.")
    if not post_example_score():
        return False
    info(f"✅ Hazır: http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}/docs")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("docker", "venv"),
        help="İnteraktif sormadan modu sabitle.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Smoke test çalıştırma (CI / debug).",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    section("Lead Priority Engine kurulum başlıyor")
    info(f"Repo: {REPO_ROOT}")

    ensure_env_file()
    ensure_openrouter_key()
    mode = choose_mode(args.mode)
    if mode == "docker":
        run_docker()
    else:
        run_venv()

    if args.skip_smoke:
        info("Smoke test atlandı (--skip-smoke).")
        return 0
    return 0 if run_smoke_test() else 5


if __name__ == "__main__":
    sys.exit(main())
