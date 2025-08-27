from utils.parser import create_parser


def main() -> None:
    """Entry point for the FutureLatents application."""
    parser = create_parser()
    args = parser.parse_args()
    # Placeholder for using args in future development
    _ = args


if __name__ == "__main__":
    main()
