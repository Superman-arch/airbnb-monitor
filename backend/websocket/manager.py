"""
WebSocket connection manager for real-time updates
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Set, Any, Optional
from uuid import uuid4

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = structlog.get_logger()


class ConnectionManager:
    """
    Manages WebSocket connections for a specific channel
    """
    
    def __init__(self, channel: str):
        self.channel = channel
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """
        Accept a new WebSocket connection
        """
        await websocket.accept()
        
        if not client_id:
            client_id = str(uuid4())
        
        async with self.lock:
            self.active_connections[client_id] = websocket
            self.connection_metadata[client_id] = {
                "connected_at": datetime.utcnow(),
                "channel": self.channel,
                "last_ping": datetime.utcnow()
            }
        
        logger.info(f"Client connected", client_id=client_id, channel=self.channel)
        return client_id
    
    async def disconnect(self, client_id: str):
        """
        Remove a WebSocket connection
        """
        async with self.lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
                del self.connection_metadata[client_id]
                logger.info(f"Client disconnected", client_id=client_id, channel=self.channel)
    
    async def send_personal_message(self, message: Any, client_id: str):
        """
        Send a message to a specific client
        """
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    if isinstance(message, dict):
                        await websocket.send_json(message)
                    else:
                        await websocket.send_text(str(message))
            except Exception as e:
                logger.error(f"Error sending message to client", 
                           client_id=client_id, error=str(e))
                await self.disconnect(client_id)
    
    async def broadcast(self, message: Any, exclude: Optional[Set[str]] = None):
        """
        Broadcast a message to all connected clients
        """
        exclude = exclude or set()
        disconnected = []
        
        for client_id, websocket in self.active_connections.items():
            if client_id in exclude:
                continue
            
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    if isinstance(message, dict):
                        await websocket.send_json(message)
                    else:
                        await websocket.send_text(str(message))
            except Exception as e:
                logger.error(f"Error broadcasting to client", 
                           client_id=client_id, error=str(e))
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_connections_info(self) -> List[Dict[str, Any]]:
        """Get information about all connections"""
        return [
            {
                "client_id": client_id,
                **metadata
            }
            for client_id, metadata in self.connection_metadata.items()
        ]


class WebSocketManager:
    """
    Main WebSocket manager handling multiple channels
    """
    
    def __init__(self):
        self.channels: Dict[str, ConnectionManager] = {}
        self.client_channels: Dict[str, Set[str]] = {}  # client_id -> set of channels
        self.heartbeat_task = None
        self._running = False
    
    def get_channel(self, channel: str) -> ConnectionManager:
        """
        Get or create a channel manager
        """
        if channel not in self.channels:
            self.channels[channel] = ConnectionManager(channel)
        return self.channels[channel]
    
    async def connect(self, websocket: WebSocket, channel: str = "default", 
                     client_id: Optional[str] = None) -> str:
        """
        Connect a client to a channel
        """
        manager = self.get_channel(channel)
        client_id = await manager.connect(websocket, client_id)
        
        # Track client's channels
        if client_id not in self.client_channels:
            self.client_channels[client_id] = set()
        self.client_channels[client_id].add(channel)
        
        # Send welcome message
        await manager.send_personal_message({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)
        
        return client_id
    
    async def disconnect(self, client_id: str, channel: Optional[str] = None):
        """
        Disconnect a client from one or all channels
        """
        if channel:
            # Disconnect from specific channel
            if channel in self.channels:
                await self.channels[channel].disconnect(client_id)
                if client_id in self.client_channels:
                    self.client_channels[client_id].discard(channel)
        else:
            # Disconnect from all channels
            if client_id in self.client_channels:
                for ch in list(self.client_channels[client_id]):
                    if ch in self.channels:
                        await self.channels[ch].disconnect(client_id)
                del self.client_channels[client_id]
    
    async def send_to_client(self, client_id: str, message: Any, channel: str = "default"):
        """
        Send a message to a specific client on a channel
        """
        if channel in self.channels:
            await self.channels[channel].send_personal_message(message, client_id)
    
    async def broadcast_to_channel(self, channel: str, message: Any, 
                                  exclude: Optional[Set[str]] = None):
        """
        Broadcast a message to all clients in a channel
        """
        if channel in self.channels:
            await self.channels[channel].broadcast(message, exclude)
    
    async def broadcast_event(self, event_type: str, data: Any, 
                            channels: Optional[List[str]] = None):
        """
        Broadcast an event to specified channels
        """
        message = {
            "type": "event",
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        channels = channels or list(self.channels.keys())
        for channel in channels:
            if channel in self.channels:
                await self.channels[channel].broadcast(message)
    
    async def broadcast_stats(self, stats: Dict[str, Any]):
        """
        Broadcast system statistics
        """
        message = {
            "type": "stats",
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast to stats channel
        if "stats" in self.channels:
            await self.channels["stats"].broadcast(message)
        
        # Also broadcast to default channel
        if "default" in self.channels:
            await self.channels["default"].broadcast(message)
    
    async def broadcast_log(self, level: str, message: str, metadata: Optional[Dict] = None):
        """
        Broadcast a log message
        """
        log_message = {
            "type": "log",
            "level": level,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast to logs channel
        if "logs" in self.channels:
            await self.channels["logs"].broadcast(log_message)
    
    async def broadcast_door_update(self, door_data: Dict[str, Any]):
        """
        Broadcast door state update
        """
        message = {
            "type": "door_update",
            "data": door_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast to doors and default channels
        for channel in ["doors", "default"]:
            if channel in self.channels:
                await self.channels[channel].broadcast(message)
    
    async def broadcast_person_update(self, person_data: Dict[str, Any]):
        """
        Broadcast person tracking update
        """
        message = {
            "type": "person_update",
            "data": person_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast to people and default channels
        for channel in ["people", "default"]:
            if channel in self.channels:
                await self.channels[channel].broadcast(message)
    
    async def broadcast_zone_update(self, zone_data: Dict[str, Any]):
        """
        Broadcast zone update
        """
        message = {
            "type": "zone_update",
            "data": zone_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast to zones and default channels
        for channel in ["zones", "default"]:
            if channel in self.channels:
                await self.channels[channel].broadcast(message)
    
    async def send_heartbeat(self):
        """
        Send periodic heartbeat to all connections
        """
        while self._running:
            try:
                message = {
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                for channel in self.channels.values():
                    await channel.broadcast(message)
                
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except Exception as e:
                logger.error(f"Error sending heartbeat", error=str(e))
                await asyncio.sleep(30)
    
    async def start(self):
        """
        Start the WebSocket manager
        """
        self._running = True
        self.heartbeat_task = asyncio.create_task(self.send_heartbeat())
        logger.info("WebSocket manager started")
    
    async def stop(self):
        """
        Stop the WebSocket manager
        """
        self._running = False
        
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all clients
        all_clients = list(self.client_channels.keys())
        for client_id in all_clients:
            await self.disconnect(client_id)
        
        logger.info("WebSocket manager stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get WebSocket statistics
        """
        return {
            "total_channels": len(self.channels),
            "total_clients": len(self.client_channels),
            "channels": {
                channel: {
                    "connections": manager.get_connection_count(),
                    "clients": manager.get_connections_info()
                }
                for channel, manager in self.channels.items()
            }
        }