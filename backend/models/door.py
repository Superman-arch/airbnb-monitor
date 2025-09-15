"""
Door database models
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json
import uuid

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, Text, JSON, DateTime,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.hybrid import hybrid_property

from .base import Base


class Door(Base):
    """
    Door entity with state tracking and metadata
    """
    
    # Identification
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=True)
    spatial_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=True)
    zone_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("zones.id"), nullable=True)
    
    # Physical properties
    bbox: Mapped[List[int]] = mapped_column(ARRAY(Integer), nullable=True)  # [x, y, width, height]
    position_x: Mapped[int] = mapped_column(Integer, nullable=True)
    position_y: Mapped[int] = mapped_column(Integer, nullable=True)
    width: Mapped[int] = mapped_column(Integer, nullable=True)
    height: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # State tracking
    current_state: Mapped[str] = mapped_column(
        String(20), 
        default="unknown",
        nullable=False
    )  # open, closed, unknown, transitioning
    previous_state: Mapped[str] = mapped_column(String(20), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    last_state_change: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Detection settings
    detection_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_calibrate: Mapped[bool] = mapped_column(Boolean, default=True)
    calibration_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)
    
    # Statistics
    total_opens: Mapped[int] = mapped_column(Integer, default=0)
    total_closes: Mapped[int] = mapped_column(Integer, default=0)
    total_time_open: Mapped[int] = mapped_column(Integer, default=0)  # seconds
    average_open_duration: Mapped[int] = mapped_column(Integer, default=0)  # seconds
    
    # Alert settings
    alert_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    max_open_duration: Mapped[int] = mapped_column(Integer, default=300)  # seconds
    rapid_change_threshold: Mapped[int] = mapped_column(Integer, default=5)
    rapid_change_window: Mapped[int] = mapped_column(Integer, default=60)  # seconds
    
    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    camera_id: Mapped[str] = mapped_column(String(100), nullable=True)
    floor: Mapped[str] = mapped_column(String(50), nullable=True)
    building: Mapped[str] = mapped_column(String(100), nullable=True)
    room_number: Mapped[str] = mapped_column(String(50), nullable=True)
    door_type: Mapped[str] = mapped_column(String(50), default="standard")  # standard, emergency, main, service
    
    # Relationships
    zone = relationship("Zone", back_populates="doors", lazy="selectin")
    events = relationship("DoorEvent", back_populates="door", lazy="dynamic", cascade="all, delete-orphan")
    access_logs = relationship("AccessLog", back_populates="door", lazy="dynamic", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_door_state", "current_state"),
        Index("idx_door_zone", "zone_id"),
        Index("idx_door_camera", "camera_id"),
        Index("idx_door_spatial", "spatial_hash"),
        Index("idx_door_last_seen", "last_seen"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="check_confidence_range"),
        CheckConstraint("current_state IN ('open', 'closed', 'unknown', 'transitioning')", name="check_state_values"),
    )
    
    @hybrid_property
    def is_open(self) -> bool:
        """Check if door is currently open"""
        return self.current_state == "open"
    
    @hybrid_property
    def is_closed(self) -> bool:
        """Check if door is currently closed"""
        return self.current_state == "closed"
    
    @hybrid_property
    def current_open_duration(self) -> Optional[timedelta]:
        """Get current open duration if door is open"""
        if self.is_open and self.last_state_change:
            return datetime.utcnow() - self.last_state_change
        return None
    
    def update_state(self, new_state: str, confidence: float = 1.0) -> bool:
        """
        Update door state with validation
        """
        valid_states = ["open", "closed", "unknown", "transitioning"]
        if new_state not in valid_states:
            return False
        
        if new_state != self.current_state:
            self.previous_state = self.current_state
            self.current_state = new_state
            self.last_state_change = datetime.utcnow()
            
            # Update statistics
            if new_state == "open":
                self.total_opens += 1
            elif new_state == "closed":
                self.total_closes += 1
                
                # Calculate open duration
                if self.previous_state == "open" and self.last_state_change:
                    duration = (datetime.utcnow() - self.last_state_change).total_seconds()
                    self.total_time_open += int(duration)
                    
                    # Update average
                    if self.total_opens > 0:
                        self.average_open_duration = self.total_time_open // self.total_opens
        
        self.confidence = confidence
        self.last_seen = datetime.utcnow()
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed properties"""
        data = super().dict()
        data.update({
            "is_open": self.is_open,
            "is_closed": self.is_closed,
            "zone_name": self.zone.name if self.zone else None,
            "current_open_duration": str(self.current_open_duration) if self.current_open_duration else None,
            "display_name": self.display_name or self.name,
        })
        return data


class DoorEvent(Base):
    """
    Door state change events
    """
    
    # Relations
    door_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("doors.id"), nullable=False)
    
    # Event details
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)  # opened, closed, discovered, alert
    previous_state: Mapped[str] = mapped_column(String(20), nullable=True)
    new_state: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    
    # Timing
    duration: Mapped[int] = mapped_column(Integer, nullable=True)  # seconds (for close events)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Context
    triggered_by: Mapped[str] = mapped_column(String(50), default="system")  # system, manual, schedule
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    snapshot_url: Mapped[str] = mapped_column(String(500), nullable=True)
    
    # Alert info
    is_alert: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_type: Mapped[str] = mapped_column(String(50), nullable=True)  # left_open, rapid_changes, unauthorized
    alert_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    door = relationship("Door", back_populates="events")
    
    # Indexes
    __table_args__ = (
        Index("idx_door_event_door", "door_id"),
        Index("idx_door_event_type", "event_type"),
        Index("idx_door_event_timestamp", "timestamp"),
        Index("idx_door_event_alert", "is_alert", "alert_sent"),
    )


class AccessLog(Base):
    """
    Door access logs for people tracking
    """
    
    # Relations
    door_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("doors.id"), nullable=False)
    person_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("people.id"), nullable=True)
    
    # Access details
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # in, out
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    
    # Person details (denormalized for quick access)
    person_track_id: Mapped[str] = mapped_column(String(50), nullable=True)
    person_name: Mapped[str] = mapped_column(String(100), nullable=True)
    
    # Context
    triggered_state: Mapped[str] = mapped_column(String(20), nullable=True)  # Door state that triggered this
    zone_from: Mapped[str] = mapped_column(String(100), nullable=True)
    zone_to: Mapped[str] = mapped_column(String(100), nullable=True)
    
    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    snapshot_url: Mapped[str] = mapped_column(String(500), nullable=True)
    
    # Authorization
    authorized: Mapped[bool] = mapped_column(Boolean, default=True)
    authorization_method: Mapped[str] = mapped_column(String(50), nullable=True)  # badge, face, manual
    
    # Relationships
    door = relationship("Door", back_populates="access_logs")
    person = relationship("Person", back_populates="access_logs")
    
    # Indexes
    __table_args__ = (
        Index("idx_access_door", "door_id"),
        Index("idx_access_person", "person_id"),
        Index("idx_access_timestamp", "timestamp"),
        Index("idx_access_direction", "direction"),
        Index("idx_access_authorized", "authorized"),
    )