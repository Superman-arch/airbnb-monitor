"""
Production-Ready Security Monitoring System
Built for Jetson Orin Nano Super
"""

import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import structlog
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from starlette.exceptions import HTTPException as StarletteHTTPException

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api import (
    auth,
    doors,
    events,
    health,
    people,
    settings,
    stats,
    streams,
    webhooks,
    zones
)
from backend.core.config import settings as app_settings
from backend.core.database import init_db, close_db
from backend.core.logging import setup_logging
from backend.core.redis_client import init_redis, close_redis
from backend.core.security import get_current_user
from backend.services.monitoring import MonitoringService
from backend.websocket.manager import WebSocketManager

# Initialize structured logging
logger = structlog.get_logger()

# Global services
monitoring_service: MonitoringService = None
ws_manager: WebSocketManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle
    """
    global monitoring_service, ws_manager
    
    try:
        logger.info("Starting Security Monitoring System", 
                   version=app_settings.VERSION,
                   environment=app_settings.ENVIRONMENT)
        
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Initialize Redis
        await init_redis()
        logger.info("Redis cache initialized")
        
        # Initialize WebSocket manager
        ws_manager = WebSocketManager()
        app.state.ws_manager = ws_manager
        logger.info("WebSocket manager initialized")
        
        # Initialize monitoring service
        monitoring_service = MonitoringService(app_settings, ws_manager)
        await monitoring_service.start()
        app.state.monitoring = monitoring_service
        logger.info("Monitoring service started")
        
        # Register signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda s, f: asyncio.create_task(shutdown()))
        
        logger.info("System ready", 
                   api_url=f"http://{app_settings.HOST}:{app_settings.PORT}",
                   docs_url=f"http://{app_settings.HOST}:{app_settings.PORT}/docs")
        
        yield
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down Security Monitoring System")
        
        if monitoring_service:
            await monitoring_service.stop()
            
        await close_redis()
        await close_db()
        
        logger.info("Shutdown complete")


async def shutdown():
    """
    Graceful shutdown handler
    """
    logger.info("Received shutdown signal")
    if monitoring_service:
        await monitoring_service.stop()
    sys.exit(0)


# Create FastAPI app
app = FastAPI(
    title="Security Monitoring System",
    description="Production-ready security monitoring for Jetson Orin Nano Super",
    version=app_settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if app_settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if app_settings.ENVIRONMENT != "production" else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

if app_settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=app_settings.ALLOWED_HOSTS
    )

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "details": exc.errors(),
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", 
                error=str(exc),
                path=str(request.url),
                method=request.method)
    
    if app_settings.ENVIRONMENT == "production":
        error_message = "Internal server error"
    else:
        error_message = str(exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(doors.router, prefix="/api/doors", tags=["Doors"])
app.include_router(people.router, prefix="/api/people", tags=["People"])
app.include_router(zones.router, prefix="/api/zones", tags=["Zones"])
app.include_router(events.router, prefix="/api/events", tags=["Events"])
app.include_router(stats.router, prefix="/api/stats", tags=["Statistics"])
app.include_router(streams.router, prefix="/api/streams", tags=["Video Streams"])
app.include_router(webhooks.router, prefix="/api/webhooks", tags=["Webhooks"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "Security Monitoring System",
        "version": app_settings.VERSION,
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    # Setup logging
    setup_logging(app_settings.LOG_LEVEL)
    
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host=app_settings.HOST,
        port=app_settings.PORT,
        reload=app_settings.ENVIRONMENT == "development",
        log_config=None,  # Use our custom logging
        access_log=False,  # Disable default access logs
        workers=1 if app_settings.ENVIRONMENT == "development" else app_settings.WORKERS,
        loop="uvloop",  # Use uvloop for better performance
        server_header=False,  # Don't expose server info
        date_header=False,  # Minimal headers
    )