"""
Code execution module
Execute Python code returned by agents and capture results
"""

import ast
import io
import re
import sys
import asyncio
import traceback
from typing import Optional, Tuple
from metagpt.logs import logger
import multiprocessing

# 全局的代码执行并发控制信号量
_execution_semaphore: Optional[asyncio.Semaphore] = None


def _get_execution_semaphore(max_concurrent: int = 128) -> asyncio.Semaphore:
    """获取或创建代码执行并发控制信号量
    
    Args:
        max_concurrent: 最大并发执行数量
        
    Returns:
        共享的 Semaphore 对象
        
    Note:
        如果 Semaphore 已存在，将使用现有的 Semaphore（即使并发数不同）。
        要更改并发数，需要先调用 reset_execution_semaphore()。
    """
    global _execution_semaphore
    if _execution_semaphore is None:
        _execution_semaphore = asyncio.Semaphore(max_concurrent)
    return _execution_semaphore
    

def extract_python_code(text: str) -> Optional[str]:
    """Extract Python code from text (may be wrapped in code blocks or plain text)

    Args:
        text: Text that may contain Python code

    Returns:
        Extracted Python code string, or None if no valid code found
    """
    # Try to extract code from markdown code blocks
    code_patterns = [
        r'```python\s*\n(.*?)```',
        r'```\s*\n(.*?)```',
        r'<code>(.*?)</code>',
    ]

    for pattern in code_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return code

    # If no code block found, try to parse the entire text as Python code
    text = text.strip()
    if not text:
        return None

    # Try to determine if it's valid Python code
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        # If parsing fails, return None
        return None


def _exec_code_in_process(code: str, result_queue: multiprocessing.Queue):
    """在独立进程中执行代码的辅助函数"""
    # 创建独立的执行命名空间
    namespace = {}
    # 重定向 stdout 和 stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        exec(code, namespace)
        output = stdout_capture.getvalue()
        error = stderr_capture.getvalue()
        if error:
            result_queue.put((None, error, False))
        else:
            result_queue.put((output, None, True))
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        result_queue.put((None, error_msg, False))
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


async def execute_python_code(code: str, timeout: int = 30) -> Tuple[Optional[str], Optional[str], bool]:
    """Execute Python code and capture output

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds

    Returns:
        (output_text, error_text, success_flag)
        - output_text: Standard output if successful
        - error_text: Error message if failed
        - success_flag: True if execution succeeded
    """
    # 使用多进程来执行代码，这样可以真正地强制终止超时的进程
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_exec_code_in_process,
        args=(code, result_queue)
    )
    
    process.start()
    
    def _wait_process():
        """在后台线程中等待进程完成"""
        process.join(timeout=timeout)
        return process.is_alive()
    
    try:
        # 在后台线程中等待进程完成，避免阻塞事件循环
        loop = asyncio.get_running_loop()
        is_alive = await loop.run_in_executor(None, _wait_process)
        
        if is_alive:
            # 进程仍在运行，说明超时了，强制终止
            process.terminate()
            await loop.run_in_executor(None, lambda: process.join(timeout=1))  # 给一点时间让进程清理
            if process.is_alive():
                process.kill()  # 如果还在运行，强制杀死
                await loop.run_in_executor(None, lambda: process.join(timeout=1))
            return None, f"执行超时（>{timeout}秒）", False
        
        # 进程已完成，获取结果
        if not result_queue.empty():
            output, error, success = result_queue.get_nowait()
            return output, error, success
        else:
            # 进程异常退出但没有结果
            return None, "进程异常退出", False
            
    except Exception as e:
        # 确保进程被终止
        if process.is_alive():
            process.terminate()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: process.join(timeout=1))
            if process.is_alive():
                process.kill()
                await loop.run_in_executor(None, lambda: process.join(timeout=1))
        import traceback
        return None, f"执行出错: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}", False


async def execute_and_get_result(code: str, timeout: int = 30) -> Tuple[Optional[str], bool]:
    """Execute Python code and return the result

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds

    Returns:
        (result_text, success_flag)
        - result_text: The result (output or return value)
        - success_flag: True if execution succeeded
    """
    output, error, success = await execute_python_code(code, timeout)

    if not success:
        logger.warning(f"Code execution failed: {error}")
        return error, False

    return output, True


async def extract_and_execute_code(text: str, timeout: int = 30, max_concurrent_execute_code: int = 128) -> Tuple[Optional[str], bool]:
    """Extract Python code from text and execute it

    Args:
        text: Text that may contain Python code
        timeout: Execution timeout in seconds
        max_concurrent_execute_code: Maximum number of concurrent code executions

    Returns:
        (result_text, is_code_flag)
        - result_text: The execution result if code was found and executed, None if execution failed, original text if not code
        - is_code_flag: True if code was detected (regardless of execution success), False if not code
    """
    code = extract_python_code(text)

    if code is None:
        # No code found, return original text and flag as not code
        return text, False

    # Code was found, try to execute
    # 使用共享的 Semaphore 来控制全局并发数量
    semaphore = _get_execution_semaphore(max_concurrent_execute_code)
    async with semaphore:
        result, success = await execute_and_get_result(code, timeout)

    if success:
        # Execution succeeded, return the result
        return result, True
    else:
        # Execution failed, return None to indicate failure (but flag as code)
        logger.warning(
            f"Code execution failed: {result[:200] if result else 'Unknown error'}")
        return None, True
