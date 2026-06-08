import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_FILES = ["metrics.json", "curves.json", "latent_pca.json", "confusion_matrix.json"]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifact_root(root: Path) -> dict[str, Any]:
    index = _read_json(root / "index.json")
    missing: list[str] = []
    for experiment in index["experiments"]:
        experiment_dir = root / experiment["path"]
        for filename in REQUIRED_FILES:
            if not (experiment_dir / filename).exists():
                missing.append(f"{experiment['id']}/{filename}")
    return {
        "datasets": index["datasets"],
        "experiment_count": len(index["experiments"]),
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VIB experiment artifacts.")
    parser.add_argument("root", type=Path, nargs="?", default=Path("artifacts/full"))
    args = parser.parse_args()
    report = validate_artifact_root(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["missing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
