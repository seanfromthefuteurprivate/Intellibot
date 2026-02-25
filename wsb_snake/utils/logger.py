import logging
import sys

# Track if root logger is configured to prevent duplicate handlers
_root_configured = False


def setup_logger(name: str = "wsb_snake", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a centralized logger with consistent formatting.

    === FIX 4: Prevent duplicate log lines ===
    - Only the root "wsb_snake" logger gets a handler
    - Child loggers (wsb_snake.collectors.foo) propagate to root
    - No duplicate handlers means no duplicate log lines
    """
    global _root_configured

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Only add handler to root wsb_snake logger, not child loggers
    # Child loggers propagate to parent by default
    is_root = (name == "wsb_snake")

    if is_root and not _root_configured:
        # Prevent adding handlers multiple times
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)

            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        _root_configured = True

    return logger


# Global logger instance (also configures root)
log = setup_logger()


def get_logger(name: str = "wsb_snake") -> logging.Logger:
    """
    Get a named logger.

    Child loggers (e.g., "wsb_snake.collectors.reddit") inherit from root
    and propagate their logs up. No handler is added to child loggers.
    """
    # Ensure root is configured first
    if not _root_configured:
        setup_logger("wsb_snake")

    return logging.getLogger(name)
