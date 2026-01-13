"""
Code execution module
Execute Python code returned by agents and capture results
"""

import ast
import io
import re
import sys
import traceback
from typing import Optional, Tuple
from metagpt.logs import logger


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


def execute_python_code(code: str, timeout: int = 30) -> Tuple[Optional[str], Optional[str], bool]:
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
    # Create a new namespace for code execution (isolated environment)
    namespace = {}

    # Redirect stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        # Execute the code
        exec(code, namespace)

        # Get output
        output = stdout_capture.getvalue()
        error = stderr_capture.getvalue()

        # If there's an error output, execution failed
        if error:
            return None, error, False

        return output, None, True

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg, False

    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def execute_and_get_result(code: str, timeout: int = 30) -> Tuple[Optional[str], bool]:
    """Execute Python code and return the result

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds

    Returns:
        (result_text, success_flag)
        - result_text: The result (output or return value)
        - success_flag: True if execution succeeded
    """
    output, error, success = execute_python_code(code, timeout)

    if not success:
        logger.warning(f"Code execution failed: {error}")
        return error, False

    return output, True


def extract_and_execute_code(text: str, timeout: int = 30) -> Tuple[Optional[str], bool]:
    """Extract Python code from text and execute it

    Args:
        text: Text that may contain Python code
        timeout: Execution timeout in seconds

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
    result, success = execute_and_get_result(code, timeout)

    if success:
        # Execution succeeded, return the result
        return result, True
    else:
        # Execution failed, return None to indicate failure (but flag as code)
        logger.warning(
            f"Code execution failed: {result[:200] if result else 'Unknown error'}")
        return None, True
