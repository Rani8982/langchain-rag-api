"""
Structured logging with structlog.
Outputs JSON in production, pretty-printed in dev/debug.
"""

import logging
import sys

import structlog

from app.core.config import settings


def setup_logging() -> None:
    """Configure structlog + stdlib logging."""

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.DEBUG:
        # Human-readable coloured output for local dev
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # Machine-readable JSON for production log aggregators
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(settings.LOG_LEVEL)

    # Silence noisy libs
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# Run once on import
setup_logging()


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
