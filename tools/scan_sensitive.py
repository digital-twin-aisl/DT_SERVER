# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Fail if a tree contains credentials, private addresses, media or weights.

    python tools/scan_sensitive.py .                  # git-tracked + untracked, unignored
    python tools/scan_sensitive.py DIR --deny-file F  # plus literal strings from F

Run by CI and by the release export. Extra deny strings (for example a password
known to have leaked) belong in a local, git-ignored file, never in this script.
"""

import argparse
import fnmatch
import ipaddress
from pathlib import Path
import re
import subprocess
import sys

MAX_BYTES = 5 * 1024 * 1024
LARGE_ALLOWED = {"apps/frontend_api/assets/map.glb"}

MEDIA = {".mp4", ".mkv", ".avi", ".mov", ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
MEDIA_ALLOWED = ["docs/**"]
WEIGHTS = {".pt", ".pth", ".tar", ".engine", ".onnx", ".ckpt", ".safetensors", ".pkl"}
OTHER_BLOCKED = {".pdf", ".zip", ".dtframe"}
NPZ_ALLOWED = ["apps/deployments/cache/*.npz"]
JSONL_ALLOWED = ["examples/**"]
TEXT_SKIP = ["LICENSE", "LICENSES/*", "**/package-lock.json", "tools/scan_sensitive.py"]

RTSP_CREDENTIAL = re.compile(r"rtsps?://([^\s/:@'\"<>]+):([^\s@'\"<>]+)@", re.I)
PLACEHOLDER = re.compile(r"^(\{.*\}|user(name)?|password|pass|secret|\*+|<.*>|\$\{?\w+\}?)$", re.I)
IPV4 = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])")
HOME_PATH = re.compile(r"/(?:home|Users)/(?!user\b|runner\b|<)[A-Za-z0-9_.-]+/")
# Common TLDs only: Python's matrix operator (a@np.linalg.inv) is not an address.
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*"
                   r"\.(?:com|net|org|edu|gov|io|ai|dev|kr|jp|cn|de|uk|me|co)\b", re.I)
EMAIL_ALLOWED = re.compile(r"(noreply@|@example\.(com|org|net)$|^git@github\.com$)", re.I)
DRIVE_ID = re.compile(r"drive\.google\.com/[^\s'\"<>]*(?:id=|/d/)(?!<)[A-Za-z0-9_-]{20,}")
DOC_NETS = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]
PRIVATE_NETS = [ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "169.254.0.0/16",
)]


def matches(path, patterns):
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def list_files(root):
    if (root / ".git").exists():
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"],
            check=True, capture_output=True,
        ).stdout.decode()
        names = [n for n in out.split("\0") if n]
    else:
        names = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
                 and ".git" not in p.parts and "node_modules" not in p.parts]
    return sorted(n for n in names if (root / n).is_file() and not (root / n).is_symlink())


def private_ip(text):
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    if any(address in net for net in DOC_NETS):
        return False
    return any(address in net for net in PRIVATE_NETS)


def scan_text(name, text, deny):
    for number, line in enumerate(text.splitlines(), 1):
        for match in RTSP_CREDENTIAL.finditer(line):
            user, password = match.groups()
            if not (PLACEHOLDER.match(user) and PLACEHOLDER.match(password)):
                yield number, "RTSP URL with credentials"
        for match in IPV4.finditer(line):
            if private_ip(match.group(1)):
                yield number, f"private IP address {match.group(1)}"
        if HOME_PATH.search(line):
            yield number, "absolute home directory path"
        for match in EMAIL.finditer(line):
            if not EMAIL_ALLOWED.search(match.group(0)):
                yield number, f"e-mail address {match.group(0)}"
        if DRIVE_ID.search(line):
            yield number, "Google Drive file link"
        for word in deny:
            if word in line:
                yield number, "deny-listed string"


def scan(root, deny):
    findings = []
    for name in list_files(root):
        path = root / name
        suffix = path.suffix.lower()
        size = path.stat().st_size
        if size > MAX_BYTES and name not in LARGE_ALLOWED:
            findings.append((name, 0, f"large file ({size / 1e6:.1f} MB)"))
        if suffix in MEDIA and not matches(name, MEDIA_ALLOWED):
            findings.append((name, 0, "image/video file"))
        if suffix in WEIGHTS:
            findings.append((name, 0, "model weights or pickle"))
        if suffix in OTHER_BLOCKED:
            findings.append((name, 0, f"{suffix} file"))
        if suffix == ".npz" and not matches(name, NPZ_ALLOWED):
            findings.append((name, 0, "numpy archive outside the ground cache"))
        if suffix == ".jsonl" and not matches(name, JSONL_ALLOWED):
            findings.append((name, 0, "recorded scene JSONL"))
        if matches(name, TEXT_SKIP) or size > MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        findings.extend((name, n, why) for n, why in scan_text(name, text, deny))
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--deny-file", type=Path,
                        help="file with one literal deny string per line")
    args = parser.parse_args(argv)
    deny = []
    if args.deny_file and args.deny_file.is_file():
        deny = [w.strip() for w in args.deny_file.read_text().splitlines()
                if w.strip() and not w.startswith("#")]
    findings = scan(args.root.resolve(), deny)
    for name, number, why in findings:
        print(f"{name}:{number}: {why}" if number else f"{name}: {why}")
    print(f"{len(findings)} finding(s)", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
