import os
import subprocess

def git_path() -> str:
    git_root = (
        subprocess.Popen(
            ["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE
        )
        .communicate()[0]
        .rstrip()
        .decode("utf-8")
    )

    return os.path.abspath(os.path.join(git_root))

OUT_DIR = f"{git_path()}/out"

os.makedirs(OUT_DIR, exist_ok=True)