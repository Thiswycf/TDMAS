import sys
import time
import asyncio
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
    max_number_of_print: int = 1000
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

    total = len(tasks)
    if total == 0:
        return []

    results: List[Any] = [None] * total
    completed = 0
    start_time = time.time()

    # 自适应 bar_len
    term_size = shutil.get_terminal_size(fallback=(80, 20))
    cols = term_size.columns
    # desc部分: [desc] + "|", 进度数字、百分数等，以及附加说明的长度估算
    # |========| 100.0% (10/100) [00:10<00:20, 1.00 it/s, ETA ...]
    # desc部分：len(desc) + 3(左右括号、空格)，进度条符号部分2个("|"和"|")，数字部分(>=20)，附加部分(>=25)
    # 这里给附加及右边预留80列，根据终端宽度动态确定bar_len
    usable_len = cols - (len(desc) + 3 + 2 + 80)
    bar_len = max(10, usable_len)

    def print_progress() -> None:
        nonlocal completed
        if total == 0:
            return

        now = time.time()
        elapsed = now - start_time
        percent = (completed / total) * 100
        filled_len = int(bar_len * completed // total)
        bar = "=" * filled_len + "-" * (bar_len - filled_len)

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

        sys.stdout.write(
            f"\r[{desc}] |{bar}| {percent:.1f}% ({completed}/{total}){extra}"
        )
        sys.stdout.flush()

    async def wrap_task(i: int, coro: Awaitable[Any]) -> Any:
        nonlocal completed
        try:
            result = await coro
        except Exception as e:
            result = e
        results[i] = result
        completed += 1
        if completed % (max(1, total // max_number_of_print)) == 0:
            print_progress()
        return result

    wrapped_tasks = [wrap_task(i, task) for i, task in enumerate(tasks)]
    print_progress()
    sys.stdout.write("\n")
    sys.stdout.flush()
    await asyncio.gather(*wrapped_tasks)

    return results
