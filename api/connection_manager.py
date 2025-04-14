import logging

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logging.info(f"Client {client_id} connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logging.info(f"Client {client_id} disconnected. Total clients: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                 self.disconnect(client_id)
            except Exception as e:
                 logging.error(f"Error sending message to {client_id}: {e}")
                 self.disconnect(client_id)

    def add_streaming_task(self, client_id: str, task: asyncio.Task):
        if client_id in self.streaming_tasks:
            logging.warning(f"Warning: Cancelling previous streaming task for client {client_id}")
            self.streaming_tasks[client_id].cancel()
        self.streaming_tasks[client_id] = task

    def remove_streaming_task(self, client_id: str):
        if client_id in self.streaming_tasks:
            del self.streaming_tasks[client_id]

    ## Method to broadcast messages to all connected clients (Not used in the original code)
    async def broadcast(self, message: str):
        disconnected_clients = []
        for client_id, websocket in self.active_connections.items():
             try:
                 await websocket.send_text(message)
             except WebSocketDisconnect:
                 disconnected_clients.append(client_id)
             except Exception as e:
                 logging.error(f"Error broadcasting to {client_id}: {e}")
                 disconnected_clients.append(client_id)
        for client_id in disconnected_clients:
             self.disconnect(client_id)