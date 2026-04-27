from collections import deque
from typing import Deque, Generic, List, Optional, TypeVar

T = TypeVar("T")


class SimQueue(Generic[T]):
    """A small fixed-capacity queue backed by a deque."""

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        self._max_size = max_size
        self._items: Deque[T] = deque()

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def items(self) -> List[T]:
        # return a shallow copy to avoid external mutation
        return list(self._items)

    @property
    def _raw_items(self) -> Deque[T]:
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def is_full(self) -> bool:
        return len(self._items) >= self._max_size

    def enqueue(self, item: T) -> bool:
        """Try to add item to the back of the queue.
        Returns True on success, False if capacity would be exceeded.
        """
        if len(self._items) + 1 > self._max_size:
            return False
        self._items.append(item)
        return True

    def dequeue(self) -> Optional[T]:
        """Remove and return the front item, or None if empty."""
        if self.is_empty():
            return None
        return self._items.popleft()

    def peek(self) -> Optional[T]:
        """Return front item without removing it, or None if empty."""
        return self._items[0] if not self.is_empty() else None

    def clear(self) -> None:
        self._items.clear()

    def remove(self, item: T) -> bool:
        try:
            self._items.remove(item)
        except ValueError:
            return False
        return True