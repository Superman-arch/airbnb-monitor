"""
Person tracking database models
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import uuid

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, Text, JSON, DateTime,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.hybrid import hybrid_property

from .base import Base


class Person(Base):
    """
    Tracked person with journey history
    """
    
    # Identification
    track_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=True)
    
    # Tracking data
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_position_x: Mapped[float] = mapped_column(Float, nullable=True)
    last_position_y: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Current state
    current_zone_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("zones.id"), nullable=True)
    current_camera_id: Mapped[str] = mapped_column(String(100), nullable=True)
    is_inside: Mapped[bool] = mapped_column(Boolean, default=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    
    # Journey statistics
    total_zones_visited: Mapped[int] = mapped_column(Integer, default=0)
    total_doors_passed: Mapped[int] = mapped_column(Integer, default=0)
    total_time_tracked: Mapped[int] = mapped_column(Integer, default=0)  # seconds
    zones_visited: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    
    # Appearance features for re-identification
    appearance_features: Mapped[List[float]] = mapped_column(ARRAY(Float), nullable=True)
    appearance_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    meta_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    
    # Relationships
    current_zone = relationship("Zone", foreign_keys=[current_zone_id], lazy="selectin")
    journey_points = relationship("JourneyPoint", back_populates="person", lazy="dynamic", cascade="all, delete-orphan")
    access_logs = relationship("AccessLog", back_populates="person", lazy="dynamic")
    events = relationship("PersonEvent", back_populates="person", lazy="dynamic", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_person_track", "track_id"),
        Index("idx_person_zone", "current_zone_id"),
        Index("idx_person_camera", "current_camera_id"),
        Index("idx_person_last_seen", "last_seen"),
        Index("idx_person_is_inside", "is_inside"),
    )
    
    @hybrid_property
    def duration_tracked(self) -> timedelta:
        """Get total duration person has been tracked"""
        return self.last_seen - self.first_seen
    
    @hybrid_property
    def inactive_duration(self) -> timedelta:
        """Get duration since last seen"""
        return datetime.utcnow() - self.last_seen
    
    def update_position(self, x: float, y: float, camera_id: str, zone_id: Optional[uuid.UUID] = None):
        """Update person's current position"""
        self.last_position_x = x
        self.last_position_y = y
        self.current_camera_id = camera_id
        self.last_seen = datetime.utcnow()
        
        if zone_id and zone_id != self.current_zone_id:
            self.current_zone_id = zone_id
            self.total_zones_visited += 1
            
            # Track unique zones
            zone_str = str(zone_id)
            if zone_str not in self.zones_visited:
                self.zones_visited = self.zones_visited + [zone_str]
    
    def mark_exited(self):
        """Mark person as having exited"""
        self.is_inside = False
        self.current_zone_id = None
        self.last_seen = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed properties"""
        data = super().dict()
        data.update({
            "duration_tracked": str(self.duration_tracked),
            "inactive_duration": str(self.inactive_duration),
            "current_zone_name": self.current_zone.name if self.current_zone else None,
            "display_name": self.display_name or f"Person {self.track_id[:8]}",
        })
        return data


class JourneyPoint(Base):
    """
    Single point in a person's journey
    """
    
    # Relations
    person_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("people.id"), nullable=False)
    zone_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("zones.id"), nullable=True)
    
    # Location data
    camera_id: Mapped[str] = mapped_column(String(100), nullable=False)
    position_x: Mapped[float] = mapped_column(Float, nullable=False)
    position_y: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Movement data
    action: Mapped[str] = mapped_column(String(50), default="move")  # move, enter, exit, wait
    direction: Mapped[str] = mapped_column(String(50), nullable=True)  # north, south, east, west
    speed: Mapped[float] = mapped_column(Float, nullable=True)  # pixels/second
    
    # Context
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    meta_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Relationships
    person = relationship("Person", back_populates="journey_points")
    zone = relationship("Zone")
    
    # Indexes
    __table_args__ = (
        Index("idx_journey_person", "person_id"),
        Index("idx_journey_zone", "zone_id"),
        Index("idx_journey_timestamp", "timestamp"),
        Index("idx_journey_camera", "camera_id"),
    )


class PersonEvent(Base):
    """
    Person-related events
    """
    
    # Relations
    person_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("people.id"), nullable=False)
    zone_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("zones.id"), nullable=True)
    door_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("doors.id"), nullable=True)
    
    # Event details
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)  # entry, exit, zone_change, door_pass
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    
    # Location context
    camera_id: Mapped[str] = mapped_column(String(100), nullable=True)
    zone_from: Mapped[str] = mapped_column(String(100), nullable=True)
    zone_to: Mapped[str] = mapped_column(String(100), nullable=True)
    
    # Additional data
    duration: Mapped[int] = mapped_column(Integer, nullable=True)  # seconds (for zone dwell time)
    meta_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    snapshot_url: Mapped[str] = mapped_column(String(500), nullable=True)
    
    # Alert info
    is_alert: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_type: Mapped[str] = mapped_column(String(50), nullable=True)  # unauthorized, loitering, crowd
    
    # Relationships
    person = relationship("Person", back_populates="events")
    zone = relationship("Zone")
    door = relationship("Door")
    
    # Indexes
    __table_args__ = (
        Index("idx_person_event_person", "person_id"),
        Index("idx_person_event_type", "event_type"),
        Index("idx_person_event_timestamp", "timestamp"),
        Index("idx_person_event_zone", "zone_id"),
        Index("idx_person_event_door", "door_id"),
    )


class PersonIdentity(Base):
    """
    Optional identity information for known persons
    """
    
    # Relations
    person_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("people.id"), unique=True, nullable=False)
    
    # Identity
    name: Mapped[str] = mapped_column(String(200), nullable=True)
    email: Mapped[str] = mapped_column(String(200), nullable=True)
    phone: Mapped[str] = mapped_column(String(50), nullable=True)
    employee_id: Mapped[str] = mapped_column(String(100), nullable=True)
    
    # Access control
    access_level: Mapped[str] = mapped_column(String(50), default="guest")  # guest, resident, staff, admin
    badge_id: Mapped[str] = mapped_column(String(100), nullable=True)
    face_encoding: Mapped[List[float]] = mapped_column(ARRAY(Float), nullable=True)
    
    # Authorization
    authorized_zones: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    authorized_doors: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    access_schedule: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=True)  # Time-based access rules
    
    # Status
    is_authorized: Mapped[bool] = mapped_column(Boolean, default=True)
    is_blocked: Mapped[bool] = mapped_column(Boolean, default=False)
    block_reason: Mapped[str] = mapped_column(String(500), nullable=True)
    
    # Metadata
    meta_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    notes: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Relationships
    person = relationship("Person", backref="identity", uselist=False)
    
    # Indexes
    __table_args__ = (
        Index("idx_identity_email", "email"),
        Index("idx_identity_badge", "badge_id"),
        Index("idx_identity_employee", "employee_id"),
        Index("idx_identity_access_level", "access_level"),
    )