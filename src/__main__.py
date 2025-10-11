"""Main module entry point."""

from dotenv import load_dotenv

load_dotenv()

from src.brain_brr.cli.cli import main

if __name__ == "__main__":
    exit(main())
