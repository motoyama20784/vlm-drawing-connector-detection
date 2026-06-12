import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path
    ollama_base_url: str
    model: str
    prompt_file: str


def get_config() -> Config:
    return Config(
        data_dir=Path(os.getenv("DATA_DIR", "/app/data")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        model=os.getenv("MODEL", "gemma4:26b"),
        prompt_file=os.getenv("ANNOTATOR_PROMPT_FILE", "prompts/annotator/v1.txt"),
    )
