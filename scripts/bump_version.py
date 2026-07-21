#!/usr/bin/env python3
"""Bump package version, commit, tag, and push in one non-interactive flow."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.dev\d+)?$")
PYPROJECT_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', flags=re.MULTILINE)
INIT_VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"', flags=re.MULTILINE)
DOCTOR_SAMPLE_RE = re.compile(r"loctran-doctor v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?")


def run(cmd: list[str], cwd: Path, capture: bool = False) -> str:
    """Run a shell command and fail fast with context."""
    print("+", " ".join(cmd))
    result = subprocess.run(
        cmd, cwd=cwd, check=True, capture_output=capture, text=capture
    )
    return result.stdout.strip() if capture else ""


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def update_single_match(
    path: Path, pattern: re.Pattern[str], replacement: str
) -> tuple[bool, str | None]:
    """Replace one pattern match in a file and return (changed, previous_value)."""
    content = read_text(path)
    match = pattern.search(content)
    if not match:
        raise RuntimeError(f"Expected version field not found in {path}")
    previous = match.group(1)
    updated = pattern.sub(replacement, content, count=1)
    changed = updated != content
    if changed:
        write_text(path, updated)
    return changed, previous


def update_docs(repo_root: Path, old_version: str, new_version: str) -> list[Path]:
    """Update version references in docs and README.

    Only replaces version strings that appear in pip install commands
    or loctran-doctor sample output — not arbitrary occurrences.
    """
    changed_files: list[Path] = []
    doc_files = [repo_root / "README.md", *sorted((repo_root / "docs").rglob("*.md"))]

    pip_re = re.compile(r"(pip install\s+loctran==)" + re.escape(old_version))
    badge_re = re.compile(r"(loctran[/-])" + re.escape(old_version))

    for path in doc_files:
        if not path.exists():
            continue
        content = read_text(path)
        updated = pip_re.sub(rf"\g<1>{new_version}", content)
        updated = badge_re.sub(rf"\g<1>{new_version}", updated)
        updated = DOCTOR_SAMPLE_RE.sub(f"loctran-doctor v{new_version}", updated)
        if updated != content:
            write_text(path, updated)
            changed_files.append(path)
    return changed_files


def preflight_checks(repo_root: Path, branch: str) -> None:
    """Abort early if the tree is dirty or we're on the wrong branch."""
    status = run(["git", "status", "--porcelain"], cwd=repo_root, capture=True)
    if status:
        raise RuntimeError(
            f"Working tree is not clean. Commit or stash changes first.\n{status}"
        )

    current_branch = run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, capture=True
    )
    if current_branch != branch:
        raise RuntimeError(
            f"Currently on branch '{current_branch}', expected '{branch}'. "
            f"Switch branches or pass --branch {current_branch}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bump loctran version and release tag."
    )
    parser.add_argument("new_version", help="New version, e.g. 0.1.1b8")
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to push after commit (default: main)",
    )
    return parser.parse_args()


def bump_version(new_version: str, branch: str) -> None:
    if not VERSION_RE.fullmatch(new_version):
        raise ValueError(
            f"Invalid version format '{new_version}'. "
            f"Expected like 0.1.2, 0.1.2a1, 0.1.2b1, or 0.1.2rc1"
        )

    repo_root = Path(__file__).resolve().parents[1]

    preflight_checks(repo_root, branch)

    pyproject_path = repo_root / "pyproject.toml"
    init_path = repo_root / "loctran" / "__init__.py"

    _, current_version = update_single_match(
        pyproject_path,
        PYPROJECT_VERSION_RE,
        f'version = "{new_version}"',
    )
    if current_version is None:
        raise RuntimeError("Could not determine current version from pyproject.toml")

    update_single_match(
        init_path,
        INIT_VERSION_RE,
        f'__version__ = "{new_version}"',
    )
    changed_docs = update_docs(repo_root, current_version, new_version)

    files_to_stage = [
        pyproject_path,
        init_path,
        *changed_docs,
    ]
    files_to_stage = sorted(set(files_to_stage))

    if not files_to_stage:
        raise RuntimeError("No files changed. Aborting release bump.")

    run(
        ["git", "add", *[str(path.relative_to(repo_root)) for path in files_to_stage]],
        cwd=repo_root,
    )
    run(["git", "commit", "-m", f"chore: bump version to {new_version}"], cwd=repo_root)

    tag_name = f"v{new_version}"
    run(["git", "push", "origin", branch], cwd=repo_root)
    run(["git", "tag", tag_name], cwd=repo_root)
    run(["git", "push", "origin", tag_name], cwd=repo_root)

    print(f"Version bump complete: {current_version} -> {new_version}")


def main() -> int:
    args = parse_args()
    try:
        bump_version(args.new_version, args.branch)
    except subprocess.CalledProcessError as err:
        print(f"Command failed with exit code {err.returncode}")
        return err.returncode
    except Exception as err:  # noqa: BLE001
        print(f"Error: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
