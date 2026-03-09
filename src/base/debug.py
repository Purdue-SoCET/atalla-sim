from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, Optional, TextIO


class _DebugState:
    def __init__(self) -> None:
        self.enabled = False
        self.flags = set()
        self.log_dir = Path("logs")
        self.append = False
        self.also_stdout = False
        self._files: Dict[str, TextIO] = {}
        self._lock = Lock()

    def configure(
        self,
        *,
        enabled: bool = True,
        flags: Optional[Iterable[str]] = None,
        log_dir: str = "logs",
        append: bool = False,
        also_stdout: bool = False,
    ) -> None:
        with self._lock:
            self.enabled = bool(enabled)
            self.flags = set(flags or [])
            self.log_dir = Path(log_dir)
            self.append = bool(append)
            self.also_stdout = bool(also_stdout)
            self._close_files()
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def is_enabled(self, flag: str) -> bool:
        if not self.enabled:
            return False
        if not self.flags:
            return True
        return flag in self.flags

    def dprintf(self, flag: str, msg: str) -> None:
        if not self.is_enabled(flag):
            return
        line = f"[{flag}] {msg}\n"
        with self._lock:
            fp = self._files.get(flag)
            if fp is None:
                mode = "a" if self.append else "w"
                path = self.log_dir / f"{flag}.log"
                fp = path.open(mode=mode, encoding="utf-8")
                self._files[flag] = fp
            fp.write(line)
            fp.flush()
        if self.also_stdout:
            print(line, end="")

    def close(self) -> None:
        with self._lock:
            self._close_files()

    def _close_files(self) -> None:
        for fp in self._files.values():
            fp.close()
        self._files.clear()


_STATE = _DebugState()


def configure_debug(
    *,
    enabled: bool = True,
    flags: Optional[Iterable[str]] = None,
    log_dir: str = "logs",
    append: bool = False,
    also_stdout: bool = False,
) -> None:
    _STATE.configure(
        enabled=enabled,
        flags=flags,
        log_dir=log_dir,
        append=append,
        also_stdout=also_stdout,
    )


def dprintf(flag: str, msg: str) -> None:
    _STATE.dprintf(flag, msg)


def close_debug() -> None:
    _STATE.close()

