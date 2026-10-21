"""Make the TRELLIS Mac generation entry point portable.

Usage: python prepare_trellis_paths.py /path/to/trellis-mac-mps
Only scripts/run_real_pipeline.py is changed. The original gets a
.before-mac-setup backup. No models or dependencies are downloaded.
"""
import argparse
import ast
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    root = args.repo.expanduser().resolve()
    if not (root / "cli.py").is_file():
        parser.error("Expected the trellis-mac-mps checkout containing cli.py")
    path = root / "scripts" / "run_real_pipeline.py"
    source = path.read_text(encoding="utf-8")
    old_setup = (
        "sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')\n"
        "os.chdir('/Users/vinayak/Downloads/TRELLIS-main')"
    )
    new_setup = (
        "from pathlib import Path\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
        "sys.path.insert(0, str(REPO_ROOT))\n"
        "os.chdir(REPO_ROOT)"
    )
    old_output = 'default="/Users/vinayak/Downloads/TRELLIS-main/outputs"'
    new_output = 'default=str(REPO_ROOT / "outputs")'
    if new_setup in source and new_output in source:
        print("Already prepared; 0 files changed.")
        return
    if old_setup not in source or old_output not in source:
        parser.error("Upstream runner changed; inspect scripts/run_real_pipeline.py")
    updated = source.replace(old_setup, new_setup).replace(old_output, new_output)
    ast.parse(updated, filename=str(path))
    backup = path.with_name(path.name + ".before-mac-setup")
    if backup.exists():
        parser.error(f"Backup already exists; inspect before retrying: {backup}")
    backup.write_text(source, encoding="utf-8")
    path.write_text(updated, encoding="utf-8")
    print(f"Updated {path.relative_to(root)}; backup: {backup.name}")


if __name__ == "__main__":
    main()
