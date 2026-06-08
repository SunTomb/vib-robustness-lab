import argparse
import json
from pathlib import Path

from vib_project.artifacts import write_experiment_artifacts
from vib_project.config import ExperimentConfig
from vib_project.phase_artifacts import write_phase_experiment, write_phase_index
from vib_project.phase_runner import run_phase_experiment
from vib_project.train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VIB robust generalization experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--dataset", choices=["mnist", "fashion_mnist", "cifar10"], required=True)
    run.add_argument("--model", choices=["cnn", "vib"], required=True)
    run.add_argument("--beta", type=float, default=0.0)
    run.add_argument("--epochs", type=int, default=5)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--latent-dim", type=int, default=32)
    run.add_argument("--learning-rate", type=float, default=1e-3)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--limit-train", type=int, default=None)
    run.add_argument("--limit-test", type=int, default=None)
    run.add_argument("--output-dir", type=Path, default=Path("artifacts/dev"))

    phase_run = subparsers.add_parser("phase-run")
    phase_run.add_argument("--dataset", choices=["mnist", "fashion_mnist", "cifar10"], required=True)
    phase_run.add_argument("--model", choices=["cnn", "vib"], required=True)
    phase_run.add_argument("--beta", type=float, default=0.0)
    phase_run.add_argument("--epochs", type=int, default=5)
    phase_run.add_argument("--batch-size", type=int, default=128)
    phase_run.add_argument("--latent-dim", type=int, default=32)
    phase_run.add_argument("--learning-rate", type=float, default=1e-3)
    phase_run.add_argument("--seed", type=int, default=7)
    phase_run.add_argument("--limit-train", type=int, default=None)
    phase_run.add_argument("--limit-test", type=int, default=None)
    phase_run.add_argument("--output-dir", type=Path, default=Path("artifacts/phase"))
    phase_run.add_argument("--corruptions", nargs="+", default=["gaussian", "salt_pepper", "blur", "contrast"])
    phase_run.add_argument("--severities", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4])
    phase_run.add_argument("--max-diagnostic-points", type=int, default=1000)
    return parser


def _update_phase_index(output_dir: Path, experiment: dict[str, object]) -> None:
    index_path = output_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"datasets": [], "experiments": []}
    datasets = list(index.get("datasets", []))
    if experiment["dataset"] not in datasets:
        datasets.append(experiment["dataset"])
    experiments = [item for item in index.get("experiments", []) if item["id"] != experiment["id"]]
    experiments.append(experiment)
    experiments = sorted(experiments, key=lambda item: (item["dataset"], item["model"], float(item["beta"])))
    write_phase_index(output_dir, datasets=datasets, experiments=experiments)


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        dataset=args.dataset,
        model=args.model,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    if args.command == "phase-run":
        payload = run_phase_experiment(config, args.corruptions, args.severities, args.max_diagnostic_points)
        write_phase_experiment(config.output_dir, config.experiment_id, payload)
        _update_phase_index(
            config.output_dir,
            {
                "id": config.experiment_id,
                "dataset": config.dataset,
                "model": config.model,
                "beta": config.beta,
                "path": config.experiment_id,
            },
        )
        print(f"Wrote phase artifacts for {config.experiment_id} to {config.output_dir}")
        return

    metrics = run_training(config)
    write_experiment_artifacts(config.output_dir, config.experiment_id, metrics)
    print(f"Wrote artifacts for {config.experiment_id} to {config.output_dir}")


if __name__ == "__main__":
    main()
