from pathlib import Path
import os

from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def _merge_with_conflict(base: OmegaConf, override: OmegaConf) -> OmegaConf:
    """Merge two ``OmegaConf`` objects reporting conflicting keys."""
    for key in override.keys():
        if key in base and base[key] != override[key]:
            if os.environ.get("RANK", "0") == "0":
                print(f"Overriding key '{key}': {base[key]} -> {override[key]}")
    return OmegaConf.merge(base, override)


def _uppercase_keys(cfg: Any) -> Any:
    """Recursively convert all mapping keys to upper case."""
    if isinstance(cfg, DictConfig):
        # ``items_ex(resolve=False)`` preserves unresolved interpolations so that
        # cross-file references remain intact after merging. Using ``items()``
        # would attempt to resolve them immediately, which fails when the
        # referenced keys are defined in a later config.
        data = {k.upper(): _uppercase_keys(v) for k, v in cfg.items_ex(resolve=False)}
        return OmegaConf.create(data)
    if isinstance(cfg, ListConfig):
        return OmegaConf.create([_uppercase_keys(v) for v in cfg])
    return cfg


def load_config(path: Path) -> DictConfig:
    """Load YAML configuration with support for simple file inheritance.

    The YAML file may contain an ``inherits`` list of relative file paths. These
    referenced configs are loaded first and merged sequentially. Later values
    override earlier ones with a notice printed for any conflicting keys.
    """
    cfg = OmegaConf.load(path)
    inherits = cfg.pop("inherits", [])
    # Uppercase keys early so that later merges use consistent casing.
    # Otherwise, merging a lowercase override (e.g. ``trainer``) with an
    # uppercase base config (``TRAINER``) results in duplicate keys where the
    # override replaces the entire subtree when converted to uppercase at the
    # end. This led to required defaults like ``GRADIENT_CHECKPOINTING`` being
    # dropped from ``TRAINER.TRAINING``.
    cfg = _uppercase_keys(cfg)

    merged = OmegaConf.create()
    for rel in inherits:
        sub_path = (path.parent / rel).resolve()
        sub_cfg = load_config(sub_path)
        merged = _merge_with_conflict(merged, sub_cfg)
    merged = _merge_with_conflict(merged, cfg)
    return merged


def print_config(config: DictConfig) -> str:
    """Return the resolved configuration in YAML format (no printing)."""
    yaml_str = OmegaConf.to_yaml(config)
    return yaml_str
