"""
Door management API endpoints
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func

from backend.core.database import get_db
from backend.core.redis_client import Cache, CacheKeys
from backend.models.door import Door, DoorEvent, AccessLog
from backend.core.config import settings

router = APIRouter()


@router.get("/", response_model=List[Dict[str, Any]])
async def get_doors(
    state: Optional[str] = Query(None, description="Filter by state (open/closed/unknown)"),
    active_only: bool = Query(True, description="Only show active doors"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all doors with optional filtering
    """
    # Try cache first
    cache_key = f"doors:list:{state}:{active_only}"
    cached = await Cache.get(cache_key)
    if cached:
        return cached
    
    # Build query
    query = select(Door)
    
    if active_only:
        query = query.where(Door.is_active == True)
    
    if state:
        query = query.where(Door.current_state == state)
    
    # Execute query
    result = await db.execute(query.order_by(Door.name))
    doors = result.scalars().all()
    
    # Convert to dict
    door_list = [door.to_dict() for door in doors]
    
    # Cache result
    await Cache.set(cache_key, door_list, ttl=5)
    
    return door_list


@router.get("/{door_id}", response_model=Dict[str, Any])
async def get_door(
    door_id: UUID,
    include_events: bool = Query(False, description="Include recent events"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get specific door details
    """
    # Try cache first
    cache_key = CacheKeys.door(str(door_id))
    cached = await Cache.get(cache_key)
    if cached and not include_events:
        return cached
    
    # Get door
    result = await db.execute(
        select(Door).where(Door.id == door_id)
    )
    door = result.scalar_one_or_none()
    
    if not door:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Door not found"
        )
    
    door_data = door.to_dict()
    
    # Include events if requested
    if include_events:
        events_result = await db.execute(
            select(DoorEvent)
            .where(DoorEvent.door_id == door_id)
            .order_by(DoorEvent.timestamp.desc())
            .limit(20)
        )
        events = events_result.scalars().all()
        door_data['recent_events'] = [
            {
                'id': str(event.id),
                'event_type': event.event_type,
                'timestamp': event.timestamp.isoformat(),
                'new_state': event.new_state,
                'confidence': event.confidence
            }
            for event in events
        ]
    
    # Cache result
    await Cache.set(cache_key, door_data, ttl=10)
    
    return door_data


@router.post("/", response_model=Dict[str, Any])
async def create_door(
    door_data: Dict[str, Any] = Body(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new door
    """
    # Create door instance
    door = Door(
        name=door_data['name'],
        display_name=door_data.get('display_name'),
        bbox=door_data.get('bbox'),
        current_state=door_data.get('current_state', 'unknown'),
        camera_id=door_data.get('camera_id', 'camera_1'),
        floor=door_data.get('floor'),
        room_number=door_data.get('room_number'),
        door_type=door_data.get('door_type', 'standard'),
        metadata=door_data.get('metadata', {})
    )
    
    # Save to database
    db.add(door)
    await db.commit()
    await db.refresh(door)
    
    # Clear cache
    await Cache.clear_pattern("doors:*")
    
    return door.to_dict()


@router.put("/{door_id}", response_model=Dict[str, Any])
async def update_door(
    door_id: UUID,
    updates: Dict[str, Any] = Body(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Update door properties
    """
    # Get door
    result = await db.execute(
        select(Door).where(Door.id == door_id)
    )
    door = result.scalar_one_or_none()
    
    if not door:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Door not found"
        )
    
    # Update fields
    for field, value in updates.items():
        if hasattr(door, field) and field not in ['id', 'created_at']:
            setattr(door, field, value)
    
    door.updated_at = datetime.utcnow()
    
    # Save changes
    await db.commit()
    await db.refresh(door)
    
    # Clear cache
    await Cache.delete(CacheKeys.door(str(door_id)))
    await Cache.clear_pattern("doors:*")
    
    return door.to_dict()


@router.post("/{door_id}/state", response_model=Dict[str, Any])
async def update_door_state(
    door_id: UUID,
    state_data: Dict[str, Any] = Body(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Update door state (open/closed/unknown)
    """
    # Get door
    result = await db.execute(
        select(Door).where(Door.id == door_id)
    )
    door = result.scalar_one_or_none()
    
    if not door:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Door not found"
        )
    
    new_state = state_data['state']
    confidence = state_data.get('confidence', 1.0)
    
    # Update state
    old_state = door.current_state
    success = door.update_state(new_state, confidence)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid state"
        )
    
    # Create event
    event = DoorEvent(
        door_id=door_id,
        event_type=f"door_{new_state}",
        previous_state=old_state,
        new_state=new_state,
        confidence=confidence,
        timestamp=datetime.utcnow(),
        metadata=state_data.get('metadata', {})
    )
    
    db.add(event)
    await db.commit()
    
    # Clear cache
    await Cache.delete(CacheKeys.door(str(door_id)))
    await Cache.clear_pattern("doors:*")
    
    # Return updated door
    await db.refresh(door)
    return door.to_dict()


@router.post("/{door_id}/calibrate", response_model=Dict[str, Any])
async def calibrate_door(
    door_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger door calibration
    """
    from backend.main import monitoring_service
    
    if not monitoring_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Monitoring service not available"
        )
    
    # Get current frame
    frame = await monitoring_service.get_current_frame()
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No video frame available"
        )
    
    # Calibrate
    result = await monitoring_service.calibrate_doors(frame)
    
    if not result.get('success'):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get('error', 'Calibration failed')
        )
    
    return result


@router.get("/{door_id}/events", response_model=List[Dict[str, Any]])
async def get_door_events(
    door_id: UUID,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    event_type: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Get door events with filtering
    """
    # Build query
    query = select(DoorEvent).where(DoorEvent.door_id == door_id)
    
    if event_type:
        query = query.where(DoorEvent.event_type == event_type)
    
    if start_time:
        query = query.where(DoorEvent.timestamp >= start_time)
    
    if end_time:
        query = query.where(DoorEvent.timestamp <= end_time)
    
    # Execute with pagination
    query = query.order_by(DoorEvent.timestamp.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    events = result.scalars().all()
    
    return [
        {
            'id': str(event.id),
            'event_type': event.event_type,
            'timestamp': event.timestamp.isoformat(),
            'previous_state': event.previous_state,
            'new_state': event.new_state,
            'confidence': event.confidence,
            'duration': event.duration,
            'is_alert': event.is_alert,
            'metadata': event.metadata
        }
        for event in events
    ]


@router.get("/{door_id}/access-logs", response_model=List[Dict[str, Any]])
async def get_door_access_logs(
    door_id: UUID,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    direction: Optional[str] = Query(None, regex="^(in|out)$"),
    authorized_only: bool = Query(False),
    db: AsyncSession = Depends(get_db)
):
    """
    Get door access logs
    """
    # Build query
    query = select(AccessLog).where(AccessLog.door_id == door_id)
    
    if direction:
        query = query.where(AccessLog.direction == direction)
    
    if authorized_only:
        query = query.where(AccessLog.authorized == True)
    
    # Execute with pagination
    query = query.order_by(AccessLog.timestamp.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    logs = result.scalars().all()
    
    return [
        {
            'id': str(log.id),
            'person_id': str(log.person_id) if log.person_id else None,
            'person_track_id': log.person_track_id,
            'direction': log.direction,
            'timestamp': log.timestamp.isoformat(),
            'authorized': log.authorized,
            'confidence': log.confidence,
            'metadata': log.metadata
        }
        for log in logs
    ]


@router.delete("/{door_id}")
async def delete_door(
    door_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a door (soft delete)
    """
    # Get door
    result = await db.execute(
        select(Door).where(Door.id == door_id)
    )
    door = result.scalar_one_or_none()
    
    if not door:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Door not found"
        )
    
    # Soft delete
    door.is_active = False
    door.updated_at = datetime.utcnow()
    
    await db.commit()
    
    # Clear cache
    await Cache.delete(CacheKeys.door(str(door_id)))
    await Cache.clear_pattern("doors:*")
    
    return {"message": "Door deleted successfully"}


@router.get("/stats/summary", response_model=Dict[str, Any])
async def get_door_stats(
    db: AsyncSession = Depends(get_db)
):
    """
    Get door statistics summary
    """
    # Get counts by state
    result = await db.execute(
        select(
            Door.current_state,
            func.count(Door.id).label('count')
        )
        .where(Door.is_active == True)
        .group_by(Door.current_state)
    )
    
    state_counts = {row.current_state: row.count for row in result}
    
    # Get total events today
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    events_result = await db.execute(
        select(func.count(DoorEvent.id))
        .where(DoorEvent.timestamp >= today_start)
    )
    events_today = events_result.scalar()
    
    # Get doors left open
    open_threshold = datetime.utcnow() - timedelta(seconds=settings.DOOR_LEFT_OPEN_SECONDS)
    long_open_result = await db.execute(
        select(func.count(Door.id))
        .where(
            and_(
                Door.current_state == 'open',
                Door.last_state_change <= open_threshold,
                Door.is_active == True
            )
        )
    )
    doors_left_open = long_open_result.scalar()
    
    return {
        'total_doors': sum(state_counts.values()),
        'doors_open': state_counts.get('open', 0),
        'doors_closed': state_counts.get('closed', 0),
        'doors_unknown': state_counts.get('unknown', 0),
        'doors_left_open': doors_left_open,
        'events_today': events_today,
        'timestamp': datetime.utcnow().isoformat()
    }