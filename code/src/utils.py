"""通用工具：UTF-8 控制台、计时、日志小工具。"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager


def enable_utf8_stdout() -> None:
    """
    让 stdout/stderr 以 UTF-8 输出，避免 Windows 控制台中文乱码。

    脚本入口处调用一次即可。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def banner(title: str, char: str = "=", width: int = 80) -> None:
    """打印分节标题。"""
    print("\n" + char * width)
    print(title)
    print(char * width)


@contextmanager
def timer(label: str):
    """计时上下文管理器：with timer('训练'): ..."""
    start = time.perf_counter()
    print(f"[开始] {label} ...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[完成] {label}，耗时 {elapsed:.1f} 秒")
