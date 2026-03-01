"""logging_config.py — Standard logging setup for the service."""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
