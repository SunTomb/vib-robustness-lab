from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    model: str
    beta: float
    epochs: int
    batch_size: int
    latent_dim: int
    learning_rate: float
    seed: int
    output_dir: Path
    limit_train: int | None = None
    limit_test: int | None = None

    @property
    def experiment_id(self) -> str:
        if self.model == "cnn":
            return f"{self.dataset}_cnn_baseline"
        beta_text = f"{self.beta:g}".replace(".", "_").replace("-", "m")
        return f"{self.dataset}_{self.model}_beta_{beta_text}"
