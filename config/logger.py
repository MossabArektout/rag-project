from loguru import logger
import sys
from config.settings import settings
import os


def setup_logging():
    """Configure application logging"""
    
    # Remove default handler
    logger.remove()
    
    # Console handler (colorized for development)
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="DEBUG" if settings.debug else "INFO"
    )
    
    # Ensure logs directory exists
    os.makedirs(settings.log_path, exist_ok=True)
    
    # File handler (all logs)
    logger.add(
        f"{settings.log_path}/app.log",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="DEBUG"
    )
    
    # Error file handler (errors only)
    logger.add(
        f"{settings.log_path}/errors.log",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="ERROR"
    )
    
    logger.info(f"Logging configured - Debug mode: {settings.debug}")
    
    return logger