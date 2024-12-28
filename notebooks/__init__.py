from pathlib import Path
import shutil

notebook_dir = Path(__file__).parent


def copy_notebooks(dst: Path, *, model_name: str) -> None:
    for entry in notebook_dir.iterdir():
        if entry.is_file():
            if entry.name.endswith(".ipynb"):
                shutil.copy(entry, dst / entry.name)
        else:  # entry.is_dir()
            if entry.name != model_name:
                continue
            for sub_entry in entry.iterdir():
                if sub_entry.is_file() and sub_entry.name.endswith(".ipynb"):
                    shutil.copy(sub_entry, dst / sub_entry.name)
