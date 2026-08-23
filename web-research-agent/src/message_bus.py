import json
import logging
import redis
from redis.connection import BlockingConnectionPool
from typing import Optional, Callable
from src.schemas import Message

logger = logging.getLogger(__name__)

class MessageBus:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        # Using connection pooling for better throughput
        self.pool = BlockingConnectionPool(
            max_connections=50, 
            timeout=20, 
            host=host, 
            port=port, 
            db=db, 
            decode_responses=True
        )
        self.redis = redis.Redis(connection_pool=self.pool)
        self.stream_name = "agent_messages"

    def publish(self, message: Message):
        """Publish a message to the recipient's Redis stream."""
        try:
            data = {"payload": message.json()}
            stream_name = f"agent_stream_{message.recipient}"
            self.redis.xadd(stream_name, data)
            logger.debug(f"Published message {message.msg_type} from {message.sender} to {message.recipient}")
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")

    def subscribe(self, worker_id: str, agent_role: str, callback: Callable[[Message], None]):
        """
        Subscribe to the agent's specific stream using a consumer group.
        """
        stream_name = f"agent_stream_{agent_role}"
        group_name = f"{agent_role}_group"
        
        try:
            self.redis.xgroup_create(stream_name, group_name, mkstream=True, id="0")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP Consumer Group name already exists" not in str(e):
                logger.warning(f"Error creating group: {e}")

        while True:
            try:
                # Read from stream
                messages = self.redis.xreadgroup(group_name, worker_id, {stream_name: ">"}, count=10, block=2000)
                if messages:
                    for stream, msgs in messages:
                        for msg_id, data in msgs:
                            raw_payload = data.get("payload")
                            if raw_payload:
                                message = Message.parse_raw(raw_payload)
                                callback(message)
                            self.redis.xack(stream_name, group_name, msg_id)
            except Exception as e:
                logger.error(f"Error consuming stream for {worker_id}: {e}")
                import time
                time.sleep(1)

    def close(self):
        self.redis.close()
