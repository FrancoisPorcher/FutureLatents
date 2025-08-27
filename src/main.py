from utils.parser import create_parser


def main() -> None:
    """Entry point for the FutureLatents application."""
    parser = create_parser()
    args = parser.parse_args()
    breakpoint()


if __name__ == "__main__":
    # run with "python -m src.main --config_path /private/home/francoisporcher/FutureLatents/configs/default.yaml"
    main()
