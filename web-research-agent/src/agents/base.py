import abc
import uuid
import logging
import threading
from typing import Dict, Any
from src.schemas import Message
from src.message_bus import MessageBus

class BaseAgent(abc.ABC):
    def __init__(self, name: str, message_bus: MessageBus):
        self.name = name
        self.worker_id = f"{name}_{uuid.uuid4().hex[:8]}"
        self.bus = message_bus
        self.logger = logging.getLogger(self.worker_id)
        self._stop_event = threading.Event()

    def start(self):
        """Start the agent's background message consumer."""
        self.logger.info(f"Starting agent: {self.worker_id}")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self.bus.subscribe(self.worker_id, self.name, self.handle_message)

    def stop(self):
        """Stop the agent."""
        self.logger.info(f"Stopping agent: {self.worker_id}")
        self._stop_event.set()

    def send_message(self, request_id: str, recipient: str, msg_type: str, payload: Dict[str, Any]):
        """Helper to construct and send a message."""
        msg = Message(
            request_id=request_id,
            sender=self.name,
            recipient=recipient,
            msg_type=msg_type,
            payload=payload
        )
        self.bus.publish(msg)

    @abc.abstractmethod
    def handle_message(self, message: Message):
        """Process incoming messages."""
        pass
