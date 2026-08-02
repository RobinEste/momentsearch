"""The background loop shape shared by the dispatcher and the reaper.

Both are the same thing: a daemon thread that pokes the manifest every few
seconds, must never die on an error, and can be switched off by a config flag.
Collapsed at the second caller rather than the third, because every loop-level
improvement they will want — jitter so N workers do not tick in lockstep on the
same rows, backoff after repeated database failures, a stop event for clean
shutdown — is otherwise two edits that have to agree.

What stays in each module is its own `*_once()`: the tick is the interesting
part, the loop around it is not.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable


def start(name: str, tick: Callable[[], object], interval_s: float,
          *, enabled: bool, on_start: str, on_disabled: str) -> None:
    """Run `tick` every `interval_s` in a daemon thread, forever."""
    if not enabled:
        print(on_disabled)
        return
    print(on_start)

    def _loop() -> None:
        while True:
            try:
                tick()
            except Exception as exc:  # noqa: BLE001 — the thread must outlive it
                print(f"[{name}] error: {type(exc).__name__}: {exc}")
            time.sleep(interval_s)

    threading.Thread(target=_loop, daemon=True, name=name).start()
