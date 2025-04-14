import configparser
import logging

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks

from .connection_manager import ConnectionManager
from llm import Service
from logger.config import setup_logging
import asyncio
import websockets

setup_logging()
llm_service = Service()
manager = ConnectionManager()
app = FastAPI(title="AI Server",
              description="AI Server for context generation and streaming.",
              version="1.0.0")

config = configparser.ConfigParser()
config.read('config.ini')
SEARCHER_SERVICE = config.get('SEARCHER', 'URL')

@app.get("/")
async def get():
    return {"message": "Context Server is running."}

# --- Async Streaming Consumer ---
async def stream_tokens_to_client(client_id: str, queue: asyncio.Queue):
    """Consumes from the queue and sends tokens to the specific WebSocket client."""
    logging.info(f"Starting token streaming consumer for {client_id}...")
    try:
        while True:
            token_or_signal: str = await queue.get()
            if token_or_signal is None:
                logging.info(f"End signal received internally for {client_id}. Sending [DONE] signal.")
                try:
                    await manager.send_personal_message("[DONE]", client_id)
                except Exception as send_error:
                    logging.error(f"Error sending [DONE] signal to {client_id}: {send_error}")
                break
            if token_or_signal.__contains__("<|eot_id|>"):
                await manager.send_personal_message(token_or_signal.replace("<|eot_id|>", ""), client_id)
            # print(f"Sending token to {client_id}: {token_or_signal[:30]}...") # Debug
            else: await manager.send_personal_message(token_or_signal, client_id)
            queue.task_done()
    except asyncio.CancelledError:
         logging.error(f"Streaming task for {client_id} was cancelled.")
    except Exception as e:
         logging.error(f"Error in streaming consumer for {client_id}: {e}")
         await manager.send_personal_message(f"ERROR: Streaming failed - {e}", client_id)
    finally:
         logging.info(f"Token streaming consumer for {client_id} finished.")
         manager.remove_streaming_task(client_id)

# --- WebSocket Endpoint ---
@app.websocket("/api/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "search":
                user_query = data.get("query")
                collection_name = data.get("collection_name")

                if not user_query or not isinstance(user_query, str):
                    await manager.send_personal_message("ERROR: Invalid 'request' for query.", client_id)
                    continue

                logging.info(f"Processing query for collection '{collection_name}': {user_query}")

                qdrant_data = None

                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        response = await client.post(SEARCHER_SERVICE, json={"query": user_query, "collection_name": collection_name})
                        response.raise_for_status() # Raise an error for bad responses
                        qdrant_data = response.json()
                        logging.info(f"HTTP {client_id}: Received response from Searcher - {len(qdrant_data)} results.")

                except httpx.RequestError as exc:
                    logging.error(f"WS {client_id}: Could not connect to Searcher Service: {exc}")
                    await manager.send_personal_message("ERROR: Failed to contact search service.", client_id)
                    continue # wait for next message

                except httpx.HTTPStatusError as exc:
                    logging.error(
                        f"WS {client_id}: Error from Searcher Service: {exc.response.status_code} - {exc.response.text}")
                    err_detail = exc.response.json().get("detail", exc.response.text) if exc.response.content else exc.response.text
                    await manager.send_personal_message(f"ERROR: Search service error - {err_detail}", client_id)
                    continue

                except Exception as e:
                    logging.error(f"WS {client_id}: Unexpected error calling Searcher Service: {e}")
                    await manager.send_personal_message("ERROR: Internal error during search phase.", client_id)
                    continue

                if qdrant_data:
                    item = qdrant_data[0]
                    summary = item.get("summary", "")
                    url = item.get("url", "")
                    max_length = 512

                    queue = asyncio.Queue()

                    consumer_task = asyncio.create_task(stream_tokens_to_client(client_id, queue))
                    manager.add_streaming_task(client_id, consumer_task)

                    logging.info(f"WS {client_id}: Dispatching LLM generation for query.")

                    asyncio.create_task(llm_service.run_async_stream(summary, url, max_length, queue))
                else:
                    logging.warning(f"WS {client_id}: No data received from Searcher, but no error reported.")
                    await manager.send_personal_message("INFO: No specific data found for query.", client_id)

            else:
                logging.warning(f"WS {client_id}: Received unknown message type: {data.get('type')}")

    except WebSocketDisconnect:
        logging.info(f"WebSocket disconnected for client {client_id}.")
        manager.disconnect(client_id)
    except websockets.exceptions.ConnectionClosedOK:
        logging.info(f"WebSocket connection closed normally for {client_id}.")
        manager.disconnect(client_id)
    except Exception as e:
        logging.error(f"Unexpected error in WebSocket endpoint for {client_id}: {e}")
        manager.disconnect(client_id)
        try:
            await websocket.close(code=1011, reason=f"Internal server error: {e}")
        except:
            pass
