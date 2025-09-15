"""
Application configuration using Pydantic Settings
"""

import os
from typing import List, Optional
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with validation
    """
    
    # Application
    APP_NAME: str = "Security Monitoring System"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="production", pattern="^(development|staging|production)$")
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False
    
    # Security
    SECRET_KEY: str = Field(default="change-this-in-production-to-a-long-random-string")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://security:security@localhost:5432/security_monitoring"
    )
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_ECHO: bool = False
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = 50
    REDIS_DECODE_RESPONSES: bool = True
    CACHE_TTL: int = 300  # 5 minutes default
    
    # Video Settings
    VIDEO_RESOLUTION: tuple = (1920, 1080)  # Full HD for Jetson Orin
    VIDEO_FPS: int = 30  # Jetson Orin can handle 30fps
    VIDEO_CODEC: str = "h264"
    VIDEO_BITRATE: int = 4000000  # 4 Mbps
    VIDEO_QUALITY: int = 23  # Lower is better (0-51)
    
    # ML/Detection Settings
    YOLO_MODEL: str = "yolov8m.pt"  # Medium model for Orin
    DETECTION_CONFIDENCE: float = 0.45
    NMS_THRESHOLD: float = 0.45
    USE_TENSORRT: bool = True  # TensorRT optimization for Jetson
    USE_GPU: bool = True
    GPU_DEVICE: int = 0
    
    # Door Detection
    DOOR_DETECTION_ENABLED: bool = True
    DOOR_INFERENCE_URL: str = "http://localhost:9001"
    DOOR_MODEL_ID: str = "door-detection/1"
    DOOR_CONFIDENCE_THRESHOLD: float = 0.6
    DOOR_STATE_CONFIRMATION_FRAMES: int = 2
    DOOR_MIN_SECONDS_BETWEEN_CHANGES: float = 1.0
    
    # Performance
    PERSON_DETECT_INTERVAL: int = 1  # Every frame for Orin
    ZONE_DETECT_INTERVAL: int = 10
    DOOR_DETECT_INTERVAL: int = 1
    MAX_FRAME_QUEUE_SIZE: int = 30
    FRAME_SKIP_THRESHOLD: int = 60  # Skip frames if queue > this
    
    # Storage
    STORAGE_PATH: Path = Path("./storage")
    VIDEO_RETENTION_HOURS: int = 72  # 3 days
    SNAPSHOT_RETENTION_DAYS: int = 30
    LOG_RETENTION_DAYS: int = 90
    DATABASE_BACKUP_DAYS: int = 7
    MAX_STORAGE_GB: int = 500  # Maximum storage usage
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: Path = Path("./logs/security_monitoring.log")
    LOG_MAX_SIZE_MB: int = 100
    LOG_BACKUP_COUNT: int = 10
    LOG_TO_CONSOLE: bool = True
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    HEALTH_CHECK_INTERVAL: int = 30
    ENABLE_SENTRY: bool = False
    SENTRY_DSN: Optional[str] = None
    
    # Webhooks
    WEBHOOK_ENABLED: bool = True
    WEBHOOK_TIMEOUT: int = 10
    WEBHOOK_RETRY_COUNT: int = 3
    WEBHOOK_RETRY_DELAY: int = 5
    DEFAULT_WEBHOOK_URL: Optional[str] = None
    PERSON_WEBHOOK_URL: Optional[str] = None
    DOOR_WEBHOOK_URL: Optional[str] = None
    
    # Alerts
    DOOR_LEFT_OPEN_SECONDS: int = 300  # 5 minutes
    RAPID_DOOR_CHANGES_COUNT: int = 5
    RAPID_DOOR_CHANGES_WINDOW: int = 60  # 1 minute
    MAX_PEOPLE_ALERT: int = 20
    UNAUTHORIZED_ACCESS_ALERT: bool = True
    
    # Jetson Specific
    JETSON_MODE: bool = True
    USE_NVMM_BUFFERS: bool = True
    ENABLE_JETSON_CLOCKS: bool = True
    FAN_MODE: str = "max"
    POWER_MODE: str = "MAXN"
    
    # Authentication
    ENABLE_AUTH: bool = True
    DEFAULT_ADMIN_EMAIL: str = "admin@security.local"
    DEFAULT_ADMIN_PASSWORD: str = "change-this-password"
    ENABLE_2FA: bool = False
    SESSION_TIMEOUT_MINUTES: int = 60
    
    # API Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    @field_validator("STORAGE_PATH", "LOG_FILE")
    def create_paths(cls, v):
        """Ensure directories exist"""
        if isinstance(v, Path):
            v.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("DATABASE_URL")
    def validate_database_url(cls, v, values):
        """Add async support to database URL"""
        if "postgresql://" in v and "+asyncpg" not in v:
            v = v.replace("postgresql://", "postgresql+asyncpg://")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# Create global settings instance
settings = Settings()

# Create necessary directories
settings.STORAGE_PATH.mkdir(parents=True, exist_ok=True)
(settings.STORAGE_PATH / "videos").mkdir(exist_ok=True)
(settings.STORAGE_PATH / "snapshots").mkdir(exist_ok=True)
(settings.STORAGE_PATH / "exports").mkdir(exist_ok=True)
settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)