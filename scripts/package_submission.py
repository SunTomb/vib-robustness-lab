import argparse
from pathlib import Path

EXCLUDED_PARTS = {
    ".git",
    ".claude",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "dist",
    ".venv",
    "venv",
    "data",
    "build",
}


def _is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_PARTS:
        return True
    normalized = path.as_posix()
    return normalized.startswith("artifacts/dev/")


def build_submission_file_list(root: Path) -> list[str]:
    files: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _is_excluded(relative):
            continue
        files.append(relative.as_posix())
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print VIB project submission file list.")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    for item in build_submission_file_list(args.root):
        print(item)


if __name__ == "__main__":
    main()
