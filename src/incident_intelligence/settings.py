"""
Settings management for the incident intelligence project.

This module defines a Settings dataclass that loads project-wide configuration such as directory paths. 
It also includes utility functions for loading JSON configuration files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def _project_root() -> Path:
    # If installed as a package, this file is in: src/incident_intelligence/settings.py
    # project root is 3 levels up.
    return Path(__file__).resolve().parents[2]


"""
Settings dataclass for project-wide configuration. This includes paths to 
key directories like data, models, and generator specifications.
"""
@dataclass(frozen=True)
class Settings:
    project_root: Path
    generator_spec_dir: Path
    data_dir: Path
    models_dir: Path

    @classmethod
    def load(cls) -> "Settings":
        root = Path(os.getenv("PROJECT_ROOT", _project_root()))
        return cls(
            project_root=root,
            generator_spec_dir=root / "generator_spec",
            data_dir=root / "data",
            models_dir=root / "models",
        )


SETTINGS = Settings.load()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_class_config() -> Dict[str, Any]:
    return load_json(SETTINGS.generator_spec_dir / "class_config.json")