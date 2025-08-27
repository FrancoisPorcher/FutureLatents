from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf


def _merge_with_conflict(base: OmegaConf, override: OmegaConf) -> OmegaConf:
    """Merge two ``OmegaConf`` objects reporting conflicting keys."""
    for key in override.keys():
        if key in base and base[key] != override[key]:
            print(f"Overriding key '{key}': {base[key]} -> {override[key]}")
    return OmegaConf.merge(base, override)


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration with support for simple file inheritance.

    The YAML file may contain an ``inherits`` list of relative file paths. These
    referenced configs are loaded first and merged sequentially. Later values
    override earlier ones with a notice printed for any conflicting keys.
    """
    cfg = OmegaConf.load(path)
    inherits = cfg.pop("inherits", [])
    merged = OmegaConf.create()
    for rel in inherits:
        sub_path = (path.parent / rel).resolve()
        sub_cfg = OmegaConf.create(load_config(sub_path))
        merged = _merge_with_conflict(merged, sub_cfg)
    merged = _merge_with_conflict(merged, cfg)
    return OmegaConf.to_container(merged, resolve=True)


def print_config(config: Dict[str, Any]) -> None:
    """Print a resolved configuration in YAML format.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary as returned by :func:`load_config`.
    """

    conf = OmegaConf.create(config)
    print(OmegaConf.to_yaml(conf))
