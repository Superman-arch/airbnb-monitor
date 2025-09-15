"""
Base database models and mixins
"""

from datetime import datetime
from typing import Any
import uuid

from sqlalchemy import Column, DateTime, String, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr, declarative_base
from sqlalchemy.orm import Mapped, mapped_column


class BaseModel:
    """
    Base model with common fields
    """
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name"""
        name = cls.__name__
        # Convert CamelCase to snake_case
        result = [name[0].lower()]
        for char in name[1:]:
            if char.isupper():
                result.append('_')
            result.append(char.lower())
        return ''.join(result) + 's'
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    
    def dict(self) -> dict:
        """Convert model to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update(self, **kwargs):
        """Update model attributes"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()
        return self
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.id})>"


# Create declarative base
Base = declarative_base(cls=BaseModel)