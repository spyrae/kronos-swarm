"""WebSocket handlers for live logs and agent status."""

import asyncio
import logging

from fastapi import WebSocket, WebSocketDisconnect

from kronos.logging import add_pii_filter

# Connected WebSocket clients for log streaming
log_clients: set[WebSocket] = set()


class WebSocketLogHandler(logging.Handler):
    """Logging handler that broadcasts to connected WebSocket clients."""

    def __init__(self):
        super().__init__()
        self.clients = log_clients  # reference to module-level set

    def emit(self, record: logging.LogRecord) -> None:
        if not self.clients:
            return
        msg = self.format(record)
        disconnected = set()
        for ws in self.clients:
            try:
                asyncio.ensure_future(ws.send_text(msg))
            except Exception:
                disconnected.add(ws)
        self.clients -= disconnected


def install_log_handler() -> None:
    """Install WebSocket log handler on root logger."""
    handler = WebSocketLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    add_pii_filter(handler)
    logging.getLogger().addHandler(handler)


async def ws_logs(websocket: WebSocket) -> None:
    """WebSocket endpoint for live log streaming."""
    await websocket.accept()
    log_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        log_clients.discard(websocket)
