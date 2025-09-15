"""People tracking API endpoints"""
from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def get_people():
    return []