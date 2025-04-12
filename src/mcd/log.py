"""Logger module."""

import logging


def setup_logger(name: str) -> logging.Logger:
    """Set up a logger.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logging.Logger
        The logger.
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    for handler in logger.handlers:
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(console_handler)

    return logger
