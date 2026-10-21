"""Fix the Hunyuan Mac port's Gradio 5 theme argument location.

Usage: python prepare_hunyuan_gradio.py /path/to/hunyuan3d-2.1-mac-rocm
The source is backed up to gradio_app.py.before-mac-setup before editing.
"""
import argparse
import ast
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    path = args.repo.expanduser().resolve() / "gradio_app.py"
    if not path.is_file():
        parser.error("Expected the Hunyuan Mac checkout containing gradio_app.py")
    source = path.read_text(encoding="utf-8")
    old_blocks = 'with gr.Blocks(title="Hunyuan3D 2.1") as demo:'
    new_blocks = 'with gr.Blocks(title="Hunyuan3D 2.1", theme=gr.themes.Soft()) as demo:'
    old_launch = '        show_error=True,\n        theme=gr.themes.Soft(),'
    new_launch = '        show_error=True,'
    if new_blocks in source and old_launch not in source:
        print("Already prepared; 0 files changed.")
        return
    if old_blocks not in source or old_launch not in source:
        parser.error("Upstream Gradio entry point changed; inspect gradio_app.py")
    updated = source.replace(old_blocks, new_blocks).replace(old_launch, new_launch)
    ast.parse(updated, filename=str(path))
    backup = path.with_name(path.name + ".before-mac-setup")
    if backup.exists():
        parser.error(f"Backup already exists; inspect before retrying: {backup}")
    backup.write_text(source, encoding="utf-8")
    path.write_text(updated, encoding="utf-8")
    print(f"Updated {path}; backup: {backup.name}")


if __name__ == "__main__":
    main()
