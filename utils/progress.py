import sys
import time
import asyncio
from termcolor import cprint
from typing import Any, Awaitable, List


def _format_seconds(secs: float) -> str:
    """将秒数格式化为 tqdm 风格的时间字符串（HH:MM:SS 或 MM:SS）"""
    secs = max(0, int(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


async def run_tasks_with_progress(
    tasks: List[Awaitable[Any]],
    desc: str = "Progress",
    max_number_of_print: int = 20
) -> List[Any]:
    """以 tqdm 风格在终端打印进度，并返回所有任务结果。

    Args:
        tasks: 协程对象列表（例如 [fn(x) for x in xs]）
        desc: 描述性文字，会显示在进度条前，如 "Evaluation Progress"
        max_number_of_print: 最大打印次数

    Returns:
        与输入 tasks 等长的结果列表，保持顺序。
        如果某个任务抛异常，则对应位置会是该异常对象。
    """
    import shutil
    import signal

    total = len(tasks)
    if total == 0:
        return []

    results: List[Any] = [None] * total
    completed = 0
    start_time = time.time()

    # bar_len 由 get_bar_len 获取，并在 SIGWINCH 时自动更新
    def get_bar_len() -> int:
        term_size = shutil.get_terminal_size(fallback=(80, 20))
        cols = term_size.columns
        usable_len = cols - (len(desc) + 3 + 2 + 80)
        return max(5, usable_len)

    bar_len = get_bar_len()

    # 信号处理器，用于检测终端宽度变化
    def handle_winch(signum, frame):
        nonlocal bar_len
        bar_len = get_bar_len()
    try:
        signal.signal(signal.SIGWINCH, handle_winch)
    except Exception:
        # 不支持 SIGWINCH 的平台（如 Windows）忽略
        pass

    def print_progress() -> None:
        nonlocal completed
        if total == 0:
            return

        now = time.time()
        elapsed = now - start_time
        percent = (completed / total) * 100
        filled_len = int(bar_len * completed // total)
        # 使用填充矩形块表示进度条
        filled_block = "█" * filled_len
        empty_block = " " * (bar_len - filled_len)
        bar = filled_block + empty_block

        extra = ""
        if completed > 0 and elapsed > 0:
            rate = completed / elapsed  # it/s
            remaining = (total - completed) / rate if rate > 0 else 0.0
            finish_ts = now + remaining

            elapsed_str = _format_seconds(elapsed)
            remaining_str = _format_seconds(remaining)
            finish_time_str = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(finish_ts))

            # 尽量模仿 tqdm 的风格：[elapsed<remaining, it/s, ETA datetime]
            extra = f" [{elapsed_str}<{remaining_str}, {rate:.2f} it/s, ETA {finish_time_str}]"

        cprint(
            f"\r[{desc}] |{bar}| {percent:.1f}% ({completed}/{total}){extra}",
            end="",
            color="green",
        )

    async def wrap_task(i: int, coro: Awaitable[Any]) -> Any:
        nonlocal completed
        try:
            result = await coro
        except Exception as e:
            result = e
        results[i] = result
        completed += 1
        print_progress()
        if completed % (max(1, total // max_number_of_print)) == 0:
            cprint("")
        return result

    wrapped_tasks = [wrap_task(i, task) for i, task in enumerate(tasks)]
    print_progress()
    cprint("")
    await asyncio.gather(*wrapped_tasks)

    return results
