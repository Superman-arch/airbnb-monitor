"""Statistics API endpoints"""
from fastapi import APIRouter
from datetime import datetime
router = APIRouter()

@router.get("/")
async def get_stats():
    return {
        "fps": 30.0,
        "doors": 0,
        "persons": 0,
        "zones": 0,
        "memory": 45.0,
        "gpu": 35.0,
        "timestamp": datetime.utcnow().isoformat()
    }