"""Main module entry point."""

from dotenv import load_dotenv

from src.brain_brr.cli.cli import main

load_dotenv()

if __name__ == "__main__":
    exit(main())
