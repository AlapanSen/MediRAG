"""
src/__init__.py — Package initializer and logging setup.
Runs once on first `import src`. Sets up logging from config.yaml.
(SRS Section 13)
"""
import logging
import os


def _setup_logging() -> None:
    """Configure root logger. No-op if handlers already exist."""
    os.makedirs("logs", exist_ok=True)

    log_level = logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_file = "logs/medirag.log"

    # Try to load level from config.yaml
    try:
        import yaml
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        level_str = cfg.get("logging", {}).get("level", "INFO")
        log_level = getattr(logging, level_str.upper(), logging.INFO)
        log_file = cfg.get("logging", {}).get("file", log_file)
        log_format = cfg.get("logging", {}).get("format", log_format)
    except Exception:
        pass  # Use defaults if config not found (e.g., during tests)

    root = logging.getLogger()
    if root.handlers:
        return  # Already configured — don't add duplicate handlers

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    except Exception:
        pass  # File logging optional — don't fail on permission errors

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)


_setup_logging()
