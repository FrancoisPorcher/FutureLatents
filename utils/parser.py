# utils/parser.py

import argparse

    parser = argparse.ArgumentParser(
        description="Command‑line interface for FutureLatents training"
    )

    parser.add_argument("--config_path",
                        required=True,
                        help="Path to the config file.")


    return parser
