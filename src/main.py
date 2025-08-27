from utils.parser import create_parser


def main() -> None:
    """Entry point for the FutureLatents application."""
    parser = create_parser()
    parser.parse_args()


if __name__ == "__main__":
    main()
