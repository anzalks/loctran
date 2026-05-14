#!/usr/bin/env python3
import sys
import subprocess
import re
from pathlib import Path

def bump_version(new_version):
    if not re.match(r"^\d+\.\d+\.\d+([a-zA-Z0-9]+)?$", new_version):
        print(f"Error: Invalid version format '{new_version}'. Should be like 0.1.2 or 0.1.2b1")
        sys.exit(1)
        
    root_dir = Path(__file__).parent.parent
    
    # 1. Update pyproject.toml
    pyproject_path = root_dir / "pyproject.toml"
    with open(pyproject_path, "r") as f:
        pyproject_content = f.read()
    
    pyproject_content = re.sub(
        r'^version\s*=\s*".*"',
        f'version = "{new_version}"',
        pyproject_content,
        flags=re.MULTILINE
    )
    
    with open(pyproject_path, "w") as f:
        f.write(pyproject_content)
        
    # 2. Update loctran/__init__.py
    init_path = root_dir / "loctran" / "__init__.py"
    with open(init_path, "r") as f:
        init_content = f.read()
        
    init_content = re.sub(
        r'^__version__\s*=\s*".*"',
        f'__version__ = "{new_version}"',
        init_content,
        flags=re.MULTILINE
    )
    
    with open(init_path, "w") as f:
        f.write(init_content)
        
    print(f"Updated version to {new_version} in pyproject.toml and loctran/__init__.py")
    
    # 3. Git commit and tag
    try:
        subprocess.run(["git", "add", "pyproject.toml", "loctran/__init__.py"], cwd=root_dir, check=True)
        subprocess.run(["git", "commit", "-m", f"chore: bump version to {new_version}"], cwd=root_dir, check=True)
        subprocess.run(["git", "tag", f"v{new_version}"], cwd=root_dir, check=True)
        print(f"Created commit and tag v{new_version}")
        
        # Ask before pushing
        print("Pushing to main...")
        subprocess.run(["git", "push", "origin", "main"], cwd=root_dir, check=True)
        subprocess.run(["git", "push", "origin", f"v{new_version}"], cwd=root_dir, check=True)
        print("Successfully pushed commit and tag to origin.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during git operations: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py <new_version>")
        sys.exit(1)
        
    bump_version(sys.argv[1])
