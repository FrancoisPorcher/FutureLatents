# utils/parser.py

"""Command-line argument parsing utilities."""

import argparse


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for the project.
    """

    parser = argparse.ArgumentParser(
        description="Command-line interface for FutureLatents training",
    )

    parser.add_argument(
        "--config_path",
        required=True,
        help="Path to the config file.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode without Weights & Biases logging.",
    )

    return parser

