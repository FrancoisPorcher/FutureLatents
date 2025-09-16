# Coding Guidelines for Codex

- Access configuration parameters directly; do not rely on helper methods such as `dict.get` that inject default values when keys are missing.
- Trust the existing configuration typing and avoid wrapping values in constructors like `str`, `int`, or `float`.
- Allow errors to surface naturally; skip defensive warnings or `None` guards so failures are obvious.
- Prefer concise solutions that achieve the goal with the fewest necessary lines of code.
