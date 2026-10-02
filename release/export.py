# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Export a committed dev revision into the history-free public release branch.

    python release/export.py                      # stage HEAD into ../DT_SERVER-oss, show diff
    python release/export.py --commit             # ... and commit it on release/oss
    python release/export.py --source-ref v0.3.0  # export another commit or tag

Only committed content is exported (uncommitted edits are ignored), paths in
release/oss-exclude.txt are dropped, submodule pointers are kept, and
tools/scan_sensitive.py must report nothing before anything is staged.
The release branch never shares history with dev, so old dev commits (and
anything sensitive in them) cannot leak through it.
"""

import argparse
import fnmatch
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[1]
EXCLUDE_FILE = REPO / "release/oss-exclude.txt"
DENY_FILE = REPO / "data/local/release-denylist.txt"  # git-ignored, optional


def git(*args, cwd=REPO, capture=True):
    result = subprocess.run(["git", *args], cwd=cwd, check=True,
                            capture_output=capture, text=True)
    return result.stdout.strip() if capture else ""


def excluded(path, patterns):
    for pattern in patterns:
        if pattern.endswith("/"):
            if path == pattern[:-1] or path.startswith(pattern):
                return True
        elif fnmatch.fnmatch(path, pattern):
            return True
    return False


def read_patterns():
    return [line.strip() for line in EXCLUDE_FILE.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def submodules(ref):
    """(path, commit) of every gitlink in the exported tree."""
    out = git("ls-tree", "-r", ref)
    links = []
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind == "commit":
            links.append((path, sha))
    return links


def prepare_worktree(worktree, branch):
    if (worktree / ".git").exists():
        current = git("rev-parse", "--abbrev-ref", "HEAD", cwd=worktree)
        if current != branch:
            sys.exit(f"{worktree} is on {current}, expected {branch}")
        if git("status", "--porcelain", cwd=worktree):
            sys.exit(f"{worktree} has uncommitted changes; commit or discard them first")
        return
    if worktree.exists() and any(worktree.iterdir()):
        sys.exit(f"{worktree} exists and is not a worktree of {branch}")
    branches = git("branch", "--list", branch)
    if branches:
        git("worktree", "add", str(worktree), branch, capture=False)
    else:
        git("worktree", "add", "--orphan", "-b", branch, str(worktree), capture=False)


def replace_tree(worktree, ref, patterns, links):
    for child in worktree.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    with tempfile.TemporaryDirectory() as folder:
        archive = Path(folder) / "source.tar"
        git("archive", "--format=tar", "-o", str(archive), ref)
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if not excluded(m.name, patterns)]
            tar.extractall(worktree, members=members)
    for path, _ in links:
        (worktree / path).mkdir(parents=True, exist_ok=True)  # uninitialized submodule


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--branch", default="release/oss")
    parser.add_argument("--worktree", type=Path, default=REPO.parent / "DT_SERVER-oss")
    parser.add_argument("--commit", action="store_true", help="commit the staged export")
    parser.add_argument("--message", help="first line of the release commit message")
    args = parser.parse_args(argv)

    source = git("rev-parse", "--verify", f"{args.source_ref}^{{commit}}")
    if args.source_ref == "HEAD" and git("status", "--porcelain", "--untracked-files=no"):
        print("note: uncommitted changes in the dev tree are NOT exported", file=sys.stderr)
    patterns = read_patterns()
    links = [(p, s) for p, s in submodules(source) if not excluded(p, patterns)]
    worktree = args.worktree.resolve()

    prepare_worktree(worktree, args.branch)
    replace_tree(worktree, source, patterns, links)

    scan = [sys.executable, str(worktree / "tools/scan_sensitive.py"), str(worktree)]
    if DENY_FILE.is_file():
        scan += ["--deny-file", str(DENY_FILE)]
    if subprocess.run(scan).returncode:
        sys.exit("sensitive-data scan failed; fix the dev branch and export again")

    git("add", "-A", cwd=worktree)
    for path, sha in links:
        git("update-index", "--add", "--cacheinfo", f"160000,{sha},{path}", cwd=worktree)
    print(git("diff", "--cached", "--stat", cwd=worktree) or "no changes")

    if not args.commit:
        print(f"\nStaged in {worktree}. Review, then rerun with --commit.")
        return
    if not git("diff", "--cached", "--name-only", cwd=worktree) and git(
        "rev-parse", "--verify", "--quiet", "HEAD", cwd=worktree
    ):
        print("Release branch already matches the source; nothing to commit.")
        return
    title = args.message or f"Release sync from dev {source[:12]}"
    git("commit", "-q", "-m", title, "-m", f"Source-Commit: {source}", cwd=worktree)
    print(f"Committed {git('rev-parse', '--short', 'HEAD', cwd=worktree)} on {args.branch}")


if __name__ == "__main__":
    main()
