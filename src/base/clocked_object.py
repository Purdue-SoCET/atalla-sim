from dataclasses import dataclass
from typing import Tuple

Time = float
EventHandle = Tuple[Time, int]

@dataclass
class Clocked:
    def __Tick(self, time: Time)-> None:
        pass

    def __Cycles(self, time: Time)-> None:
        pass

    def __update(self, time: Time)-> None:
        pass