from base.eventq import EventQueue
from base.core import Core
from base.debug import close_debug, configure_debug, dprintf
from typing import Iterable, Optional, Tuple

Time = float
EventHandle = Tuple[Time, int]

class Sim:
    def __init__(self) -> None:
        self.event_queue = None
        self.core = None

    def init(self, event_queue: EventQueue, core: Core) -> None:
        self.event_queue = event_queue
        self.core = core

    def run(self, until: Optional[Time] = None) -> None:
        dprintf("Sim", f"run(start, until={until})")
        if until is None:
            self.event_queue.run_all()
        else:
            self.event_queue.run_until(until)
        dprintf("Sim", "run(done)")

    def stop(self) -> None:
        dprintf("Sim", "stop()")

    def configure_debug(
        self,
        flags: Optional[Iterable[str]] = None,
        log_dir: str = "logs",
        append: bool = False,
        also_stdout: bool = False,
    ) -> None:
        configure_debug(
            enabled=True,
            flags=flags,
            log_dir=log_dir,
            append=append,
            also_stdout=also_stdout,
        )

    def close_debug(self) -> None:
        close_debug()
