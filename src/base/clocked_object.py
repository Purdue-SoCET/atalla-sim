from typing import Tuple

Time = float
EventHandle = Tuple[Time, int]

class Clocked:
    def _Tick(self, time: Time) -> None:
        print(f"{self.__class__.__name__} tick at {time}")

    def __Cycles(self, time: Time)-> None:
        pass

    def __update(self, time: Time)-> None:
        pass