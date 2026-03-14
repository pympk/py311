from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskResult:
    ok: bool
    msg: str = ""
    val: Any = None


#
