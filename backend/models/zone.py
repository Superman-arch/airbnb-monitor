"""
Zone database models
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, Text, DateTime,
    ForeignKey, Index, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.hybrid import hybrid_property

from .base import Base


class Zone(Base):
    """
    Monitored zones/areas
    """
    
    # Identification
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=True)
    zone_type: Mapped[str] = mapped_column(String(50), default="room")  # room, hallway, entrance, outdoor, restricted
    
    # Physical properties
    coordinates: Mapped[List[List[float]]] = mapped_column(JSONB, nullable=False)  # Polygon coordinates
    area: Mapped[float] = mapped_column(Float, nullable=True)  # Square meters
    height: Mapped[float] = mapped_column(Float, nullable=True)  # Meters
    
    # Location
    floor: Mapped[str] = mapped_column(String(50), nullable=True)
    building: Mapped[str] = mapped_column(String(100), nullable=True)
    camera_id: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Monitoring settings
    monitoring_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    motion_detection: Mapped[bool] = mapped_column(Boolean, default=True)
    person_tracking: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Occupancy
    current_occupancy: Mapped[int] = mapped_column(Integer, default=0)
    max_occupancy: Mapped[int] = mapped_column(Integer, nullable=True)
    average_occupancy: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Access control
    access_level: Mapped[str] = mapped_column(String(50), default="public")  # public, restricted, private, emergency
    authorization_required: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Alert settings
    alert_on_entry: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_on_exit: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_on_loitering: Mapped[bool] = mapped_column(Boolean, default=False)
    loitering_threshold: Mapped[int] = mapped_column(Integer, default=300)  # seconds
    alert_on_crowding: Mapped[bool] = mapped_column(Boolean, default=False)
    crowding_threshold: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # Statistics
    total_entries: Mapped[int] = mapped_column(Integer, default=0)
    total_exits: Mapped[int] = mapped_column(Integer, default=0)
    total_time_occupied: Mapped[int] = mapped_column(Integer, default=0)  # seconds
    last_occupied: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    color: Mapped[str] = mapped_column(String(7), default="#3498db")  # Hex color for UI
    
    # Relationships
    doors = relationship("Door", back_populates="zone", lazy="selectin")
    events = relationship("ZoneEvent", back_populates="zone", lazy="dynamic", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_zone_name", "name"),
        Index("idx_zone_type", "zone_type"),
        Index("idx_zone_camera", "camera_id"),
        Index("idx_zone_floor", "floor"),
        Index("idx_zone_access", "access_level"),
        CheckConstraint("current_occupancy >= 0", name="check_occupancy_positive"),
    )
    
    @hybrid_property
    def is_occupied(self) -> bool:
        """Check if zone is currently occupied"""
        return self.current_occupancy > 0
    
    @hybrid_property
    def is_crowded(self) -> bool:
        """Check if zone is overcrowded"""
        if self.crowding_threshold:
            return self.current_occupancy >= self.crowding_threshold
        if self.max_occupancy:
            return self.current_occupancy >= self.max_occupancy
        return False
    
    @hybrid_property
    def occupancy_percentage(self) -> Optional[float]:
        """Get occupancy as percentage of max"""
        if self.max_occupancy and self.max_occupancy > 0:
            return (self.current_occupancy / self.max_occupancy) * 100
        return None
    
    def update_occupancy(self, delta: int):
        """Update zone occupancy"""
        new_occupancy = max(0, self.current_occupancy + delta)
        self.current_occupancy = new_occupancy
        
        if new_occupancy > 0:
            self.last_occupied = datetime.utcnow()
        
        # Update statistics
        if delta > 0:
            self.total_entries += delta
        else:
            self.total_exits += abs(delta)
        
        # Update average (simple moving average)
        self.average_occupancy = (self.average_occupancy * 0.95) + (new_occupancy * 0.05)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed properties"""
        data = super().dict()
        data.update({
            "is_occupied": self.is_occupied,
            "is_crowded": self.is_crowded,
            "occupancy_percentage": self.occupancy_percentage,
            "door_count": len(self.doors) if self.doors else 0,
            "display_name": self.display_name or self.name,
        })
        return data


class ZoneEvent(Base):
    """
    Zone-related events
    """
    
    # Relations
    zone_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("zones.id"), nullable=False)
    person_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("people.id"), nullable=True)
    
    # Event details
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)  # entry, exit, loitering, crowding
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Occupancy at time of event
    occupancy_before: Mapped[int] = mapped_column(Integer, nullable=True)
    occupancy_after: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # Context
    duration: Mapped[int] = mapped_column(Integer, nullable=True)  # seconds (for loitering)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    camera_id: Mapped[str] = mapped_column(String(100), nullable=True)
    
    # Additional data
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    snapshot_url: Mapped[str] = mapped_column(String(500), nullable=True)
    
    # Alert info
    is_alert: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    zone = relationship("Zone", back_populates="events")
    person = relationship("Person")
    
    # Indexes
    __table_args__ = (
        Index("idx_zone_event_zone", "zone_id"),
        Index("idx_zone_event_person", "person_id"),
        Index("idx_zone_event_type", "event_type"),
        Index("idx_zone_event_timestamp", "timestamp"),
        Index("idx_zone_event_alert", "is_alert"),
    )


class ZoneConnection(Base):
    """
    Connections between zones (for pathfinding and flow analysis)
    """
    
    # Connected zones
    zone_from_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("zones.id"), nullable=False)
    zone_to_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("zones.id"), nullable=False)
    
    # Connection properties
    connection_type: Mapped[str] = mapped_column(String(50), default="direct")  # direct, door, hallway
    door_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("doors.id"), nullable=True)
    distance: Mapped[float] = mapped_column(Float, nullable=True)  # meters
    
    # Traffic statistics
    total_transitions: Mapped[int] = mapped_column(Integer, default=0)
    average_transition_time: Mapped[float] = mapped_column(Float, default=0.0)  # seconds
    
    # Access control
    bidirectional: Mapped[bool] = mapped_column(Boolean, default=True)
    access_restricted: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Relationships
    zone_from = relationship("Zone", foreign_keys=[zone_from_id])
    zone_to = relationship("Zone", foreign_keys=[zone_to_id])
    door = relationship("Door")
    
    # Indexes
    __table_args__ = (
        Index("idx_connection_from", "zone_from_id"),
        Index("idx_connection_to", "zone_to_id"),
        Index("idx_connection_door", "door_id"),
    )