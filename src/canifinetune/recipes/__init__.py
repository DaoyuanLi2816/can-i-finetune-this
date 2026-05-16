"""Generate ready-to-run training recipes (train.py, config.yaml, README, ...)."""

from .generator import RecipeRequest, generate_recipe

__all__ = ["RecipeRequest", "generate_recipe"]
