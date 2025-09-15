"""
Structured logging configuration
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level
from structlog.stdlib import ProcessorFormatter

from backend.core.config import settings


def setup_logging(log_level: str = "INFO"):
    """
    Configure structured logging for the application
    """
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.dict_tracebacks,
            JSONRenderer() if settings.LOG_FORMAT == "json" else structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Configure file logging if enabled
    if settings.LOG_FILE:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=settings.LOG_MAX_SIZE_MB * 1024 * 1024,
            backupCount=settings.LOG_BACKUP_COUNT,
        )
        file_handler.setFormatter(
            ProcessorFormatter(
                processor=JSONRenderer() if settings.LOG_FORMAT == "json" else structlog.dev.ConsoleRenderer(colors=False)
            )
        )
        logging.getLogger().addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("aioredis").setLevel(logging.WARNING)
    
    # Log startup message
    logger = structlog.get_logger()
    logger.info(
        "Logging configured",
        level=log_level,
        format=settings.LOG_FORMAT,
        file=str(settings.LOG_FILE) if settings.LOG_FILE else None,
    )


class LoggerAdapter:
    """
    Logger adapter for WebSocket broadcast
    """
    
    def __init__(self, ws_manager=None):
        self.logger = structlog.get_logger()
        self.ws_manager = ws_manager
    
    async def log(self, level: str, message: str, **kwargs):
        """
        Log message and broadcast to WebSocket
        """
        # Log using structlog
        log_method = getattr(self.logger, level.lower())
        log_method(message, **kwargs)
        
        # Broadcast to WebSocket if available
        if self.ws_manager:
            await self.ws_manager.broadcast_log(level, message, kwargs)
    
    async def debug(self, message: str, **kwargs):
        await self.log("debug", message, **kwargs)
    
    async def info(self, message: str, **kwargs):
        await self.log("info", message, **kwargs)
    
    async def warning(self, message: str, **kwargs):
        await self.log("warning", message, **kwargs)
    
    async def error(self, message: str, **kwargs):
        await self.log("error", message, **kwargs)
    
    async def critical(self, message: str, **kwargs):
        await self.log("critical", message, **kwargs)


class RequestLogger:
    """
    HTTP request/response logger
    """
    
    def __init__(self):
        self.logger = structlog.get_logger()
    
    def log_request(self, request: Any, **kwargs):
        """
        Log incoming request
        """
        self.logger.info(
            "Request received",
            method=request.method,
            path=request.url.path,
            query=dict(request.query_params),
            client=request.client.host if request.client else None,
            **kwargs
        )
    
    def log_response(self, status_code: int, duration_ms: float, **kwargs):
        """
        Log outgoing response
        """
        level = "info" if status_code < 400 else "warning" if status_code < 500 else "error"
        log_method = getattr(self.logger, level)
        
        log_method(
            "Response sent",
            status_code=status_code,
            duration_ms=duration_ms,
            **kwargs
        )


class SystemLogger:
    """
    System event logger
    """
    
    def __init__(self):
        self.logger = structlog.get_logger()
    
    def log_startup(self, **kwargs):
        """
        Log system startup
        """
        self.logger.info("System starting up", **kwargs)
    
    def log_shutdown(self, **kwargs):
        """
        Log system shutdown
        """
        self.logger.info("System shutting down", **kwargs)
    
    def log_door_event(self, door_id: str, event_type: str, **kwargs):
        """
        Log door-related events
        """
        self.logger.info(
            "Door event",
            door_id=door_id,
            event_type=event_type,
            **kwargs
        )
    
    def log_person_event(self, person_id: str, event_type: str, **kwargs):
        """
        Log person-related events
        """
        self.logger.info(
            "Person event",
            person_id=person_id,
            event_type=event_type,
            **kwargs
        )
    
    def log_detection(self, detection_type: str, count: int, **kwargs):
        """
        Log detection results
        """
        self.logger.debug(
            "Detection completed",
            detection_type=detection_type,
            count=count,
            **kwargs
        )
    
    def log_performance(self, metric: str, value: float, **kwargs):
        """
        Log performance metrics
        """
        self.logger.debug(
            "Performance metric",
            metric=metric,
            value=value,
            **kwargs
        )
    
    def log_error(self, error: str, **kwargs):
        """
        Log errors with context
        """
        self.logger.error(
            "Error occurred",
            error=error,
            **kwargs
        )