"""
WebSocket Connection Manager for Real-Time Dashboard Updates

Provides real-time streaming of:
- Position updates
- Trade executions
- System events
- P&L updates
- Heartbeat to maintain connections
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of WebSocket messages."""
    POSITION_UPDATE = "position_update"
    TRADE_EXECUTED = "trade_executed"
    EVENT = "event"
    PNL_UPDATE = "pnl_update"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    CONNECTED = "connected"
    SUBSCRIPTION = "subscription"


class EventSeverity(str, Enum):
    """Severity levels for events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class WebSocketMessage:
    """Standard WebSocket message format."""
    type: MessageType
    data: Dict[str, Any]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps({
            "type": self.type.value if isinstance(self.type, MessageType) else self.type,
            "data": self.data,
            "timestamp": self.timestamp
        })


@dataclass
class ClientInfo:
    """Information about a connected client."""
    websocket: WebSocket
    client_id: str
    connected_at: datetime
    subscriptions: Set[str]
    last_heartbeat: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding websocket)."""
        return {
            "client_id": self.client_id,
            "connected_at": self.connected_at.isoformat(),
            "subscriptions": list(self.subscriptions),
            "last_heartbeat": self.last_heartbeat.isoformat()
        }


class ConnectionManager:
    """
    Manages WebSocket connections for real-time dashboard updates.

    Features:
    - Multiple client support
    - Subscription-based message filtering
    - Automatic heartbeat to keep connections alive
    - Graceful disconnect handling
    - JSON message serialization
    """

    def __init__(self, heartbeat_interval: int = 30):
        """
        Initialize the connection manager.

        Args:
            heartbeat_interval: Seconds between heartbeat messages
        """
        self.active_connections: Dict[str, ClientInfo] = {}
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._client_counter = 0
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        client_id: Optional[str] = None,
        subscriptions: Optional[List[str]] = None
    ) -> str:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            client_id: Optional client identifier
            subscriptions: Optional list of message types to subscribe to

        Returns:
            The assigned client ID
        """
        await websocket.accept()

        async with self._lock:
            self._client_counter += 1
            if client_id is None:
                client_id = f"client_{self._client_counter}"

            # Default subscriptions to all message types
            if subscriptions is None:
                subscriptions = [mt.value for mt in MessageType]

            client_info = ClientInfo(
                websocket=websocket,
                client_id=client_id,
                connected_at=datetime.utcnow(),
                subscriptions=set(subscriptions),
                last_heartbeat=datetime.utcnow()
            )

            self.active_connections[client_id] = client_info

            # Start heartbeat if this is the first connection
            if len(self.active_connections) == 1:
                self._start_heartbeat()

        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

        # Send connection confirmation
        await self.send_personal_message(
            client_id,
            WebSocketMessage(
                type=MessageType.CONNECTED,
                data={
                    "client_id": client_id,
                    "subscriptions": list(client_info.subscriptions),
                    "message": "Connected to Intellibot WebSocket"
                }
            )
        )

        return client_id

    async def disconnect(self, client_id: str) -> None:
        """
        Remove a client connection.

        Args:
            client_id: The client to disconnect
        """
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
                logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

                # Stop heartbeat if no connections remain
                if len(self.active_connections) == 0:
                    self._stop_heartbeat()

    async def send_personal_message(
        self,
        client_id: str,
        message: WebSocketMessage
    ) -> bool:
        """
        Send a message to a specific client.

        Args:
            client_id: The target client
            message: The message to send

        Returns:
            True if sent successfully, False otherwise
        """
        if client_id not in self.active_connections:
            return False

        client = self.active_connections[client_id]

        # Check if client is subscribed to this message type
        msg_type = message.type.value if isinstance(message.type, MessageType) else message.type
        if msg_type not in client.subscriptions and msg_type != MessageType.HEARTBEAT.value:
            return False

        try:
            await client.websocket.send_text(message.to_json())
            return True
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {e}")
            await self.disconnect(client_id)
            return False

    async def broadcast(
        self,
        message: WebSocketMessage,
        exclude: Optional[List[str]] = None
    ) -> int:
        """
        Broadcast a message to all connected clients.

        Args:
            message: The message to broadcast
            exclude: Optional list of client IDs to exclude

        Returns:
            Number of clients that received the message
        """
        if exclude is None:
            exclude = []

        sent_count = 0
        disconnected = []

        for client_id, client in list(self.active_connections.items()):
            if client_id in exclude:
                continue

            # Check subscription
            msg_type = message.type.value if isinstance(message.type, MessageType) else message.type
            if msg_type not in client.subscriptions and msg_type != MessageType.HEARTBEAT.value:
                continue

            try:
                await client.websocket.send_text(message.to_json())
                sent_count += 1
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)

        return sent_count

    async def broadcast_position_update(
        self,
        positions: List[Dict[str, Any]]
    ) -> int:
        """
        Broadcast position updates to all clients.

        Args:
            positions: List of position data

        Returns:
            Number of clients that received the message
        """
        message = WebSocketMessage(
            type=MessageType.POSITION_UPDATE,
            data={"positions": positions}
        )
        return await self.broadcast(message)

    async def broadcast_trade(
        self,
        trade: Dict[str, Any]
    ) -> int:
        """
        Broadcast a trade execution to all clients.

        Args:
            trade: Trade execution data

        Returns:
            Number of clients that received the message
        """
        message = WebSocketMessage(
            type=MessageType.TRADE_EXECUTED,
            data={"trade": trade}
        )
        return await self.broadcast(message)

    async def broadcast_event(
        self,
        event_type: str,
        description: str,
        severity: EventSeverity = EventSeverity.INFO,
        details: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Broadcast a system event to all clients.

        Args:
            event_type: Type of event
            description: Event description
            severity: Event severity level
            details: Optional additional details

        Returns:
            Number of clients that received the message
        """
        message = WebSocketMessage(
            type=MessageType.EVENT,
            data={
                "event_type": event_type,
                "description": description,
                "severity": severity.value,
                "details": details or {}
            }
        )
        return await self.broadcast(message)

    async def broadcast_pnl_update(
        self,
        total_pnl: float,
        daily_pnl: float,
        unrealized_pnl: float,
        realized_pnl: float,
        positions_pnl: Optional[Dict[str, float]] = None
    ) -> int:
        """
        Broadcast P&L updates to all clients.

        Args:
            total_pnl: Total P&L
            daily_pnl: Today's P&L
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L
            positions_pnl: Optional per-position P&L breakdown

        Returns:
            Number of clients that received the message
        """
        message = WebSocketMessage(
            type=MessageType.PNL_UPDATE,
            data={
                "total_pnl": total_pnl,
                "daily_pnl": daily_pnl,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_pnl,
                "positions_pnl": positions_pnl or {}
            }
        )
        return await self.broadcast(message)

    async def update_subscription(
        self,
        client_id: str,
        subscriptions: List[str],
        add: bool = True
    ) -> bool:
        """
        Update a client's subscriptions.

        Args:
            client_id: The client to update
            subscriptions: List of message types
            add: True to add subscriptions, False to remove

        Returns:
            True if updated successfully
        """
        if client_id not in self.active_connections:
            return False

        client = self.active_connections[client_id]

        if add:
            client.subscriptions.update(subscriptions)
        else:
            client.subscriptions.difference_update(subscriptions)

        # Notify client of subscription change
        await self.send_personal_message(
            client_id,
            WebSocketMessage(
                type=MessageType.SUBSCRIPTION,
                data={
                    "action": "add" if add else "remove",
                    "subscriptions": subscriptions,
                    "current_subscriptions": list(client.subscriptions)
                }
            )
        )

        return True

    def _start_heartbeat(self) -> None:
        """Start the heartbeat background task."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("Heartbeat task started")

    def _stop_heartbeat(self) -> None:
        """Stop the heartbeat background task."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            logger.info("Heartbeat task stopped")

    async def _heartbeat_loop(self) -> None:
        """Background loop that sends heartbeat messages."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                if not self.active_connections:
                    continue

                heartbeat_msg = WebSocketMessage(
                    type=MessageType.HEARTBEAT,
                    data={
                        "server_time": datetime.utcnow().isoformat() + "Z",
                        "connected_clients": len(self.active_connections)
                    }
                )

                disconnected = []
                for client_id, client in list(self.active_connections.items()):
                    try:
                        await client.websocket.send_text(heartbeat_msg.to_json())
                        client.last_heartbeat = datetime.utcnow()
                    except Exception:
                        disconnected.append(client_id)

                # Clean up disconnected clients
                for client_id in disconnected:
                    await self.disconnect(client_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific client."""
        if client_id in self.active_connections:
            return self.active_connections[client_id].to_dict()
        return None

    def get_all_clients(self) -> List[Dict[str, Any]]:
        """Get information about all connected clients."""
        return [client.to_dict() for client in self.active_connections.values()]


# Global connection manager instance
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, client_id: Optional[str] = None):
    """
    WebSocket endpoint handler for FastAPI.

    Usage in FastAPI:
        @app.websocket("/ws")
        async def websocket_route(websocket: WebSocket):
            await websocket_endpoint(websocket)

    Or with client ID:
        @app.websocket("/ws/{client_id}")
        async def websocket_route(websocket: WebSocket, client_id: str):
            await websocket_endpoint(websocket, client_id)
    """
    assigned_id = await manager.connect(websocket, client_id)

    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                # Handle subscription updates
                if message.get("action") == "subscribe":
                    await manager.update_subscription(
                        assigned_id,
                        message.get("types", []),
                        add=True
                    )
                elif message.get("action") == "unsubscribe":
                    await manager.update_subscription(
                        assigned_id,
                        message.get("types", []),
                        add=False
                    )
                elif message.get("action") == "ping":
                    # Respond to client ping
                    await manager.send_personal_message(
                        assigned_id,
                        WebSocketMessage(
                            type=MessageType.HEARTBEAT,
                            data={"pong": True}
                        )
                    )

            except json.JSONDecodeError:
                await manager.send_personal_message(
                    assigned_id,
                    WebSocketMessage(
                        type=MessageType.ERROR,
                        data={"error": "Invalid JSON message"}
                    )
                )

    except WebSocketDisconnect:
        await manager.disconnect(assigned_id)
    except Exception as e:
        logger.error(f"WebSocket error for {assigned_id}: {e}")
        await manager.disconnect(assigned_id)
