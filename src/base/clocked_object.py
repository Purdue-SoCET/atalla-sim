from typing import Tuple
from base.debug import dprintf

Time = float
EventHandle = Tuple[Time, int]

class Clocked:
    def tick(self, time: Time) -> None:
        dprintf("Clocked", f"{self.__class__.__name__} tick at {time}")

    def __Cycles(self, time: Time)-> None:
        pass

    def __update(self, time: Time)-> None:
        pass
