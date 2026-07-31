import asyncio
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AsyncMessageBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, topic: str, callback: Callable) -> None:
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    async def publish(self, topic: str, data: Any = None) -> None:
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                asyncio.create_task(callback(data))
        logger.debug(f"Published to topic: {topic}")
