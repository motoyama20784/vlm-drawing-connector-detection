import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path
    inputs_dir: Path
    original_dir: Path   # data/inputs/original/
    masked_dir: Path     # data/inputs/masking/
    masking_dir: Path    # data/masking/  (bbox JSON)
    ollama_base_url: str
    model: str
    prompt_file: str


def get_config() -> Config:
    data_dir = Path(os.getenv("DATA_DIR", "/app/data"))
    inputs_dir = data_dir / "inputs"
    return Config(
        data_dir=data_dir,
        inputs_dir=inputs_dir,
        original_dir=inputs_dir / "original",
        masked_dir=inputs_dir / "masking",
        masking_dir=data_dir / "masking",
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        model=os.getenv("MODEL", "gemma4:26b"),
        prompt_file=os.getenv("ANNOTATOR_PROMPT_FILE", "prompts/annotator/v1.txt"),
    )
