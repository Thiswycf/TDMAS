import concurrent
import sys
import traceback
import inspect
from typing import List, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

# 直接导入当前目录下的模块
from ScoreFlow.scripts.GSM8K.operator_an import *
from ScoreFlow.scripts.GSM8K.op_prompt import *
from metagpt.actions.action_node import ActionNode
from metagpt.logs import logger
import asyncio
# 使用绝对导入路径
from utils.llm_manager import get_llm_instance
from utils.message import Message
from utils.common import TIME_TO_EXEC_CODE


class Operator:
    """Base operator class, all specific operators should inherit from this class"""

    def __init__(self,
                 llm_name: Optional[str] = None,
                 problem: Optional[str] = None,
                 prompt_format: str = "",
                 fields: Optional[List[str]] = None,
                 name: str = "Operator"):
        """Initialize operator

        Args:
            llm_name: Name of the large language model
            problem: Initial problem
            prompt_format: Prompt template
            fields: List of parameter fields needed to fill the prompt
            name: Operator name
        """
        # 查找调用栈中的 Workflow 实例（以 Workflow 为单位检查）
        workflow_instance = self._find_workflow_instance()
        if workflow_instance is not None:
            # 检查同一 Workflow 中是否已经实例化了同一类型的算子
            operator_class = self.__class__
            if not hasattr(workflow_instance, '_operator_instances'):
                workflow_instance._operator_instances = {}

            if operator_class in workflow_instance._operator_instances:
                raise RuntimeError(
                    f"Operator of type {operator_class.__name__} can only be instantiated once per Workflow. "
                    f"An instance already exists in this Workflow."
                )

            # 在 Workflow 实例中注册当前算子
            workflow_instance._operator_instances[operator_class] = self

        # Get LLM instance using llm_manager
        llm_name = 'Qwen2.5-14B-Instruct'  # NOTE: modified for inference test
        self.llm_instance = get_llm_instance(llm_name)
        self.problem = problem
        self.prompt_format = prompt_format
        self.fields = fields or []
        self.name = name

        # Runtime parameters - 改为列表，保存每次调用的输入输出关系
        self.call_history: List[dict] = []  # 每次调用的历史记录
        # 保持向后兼容的接口
        self.input_messages: List[Message] = []
        self.output_content: str = ""

    def _find_workflow_instance(self):
        """通过调用栈查找 Workflow 实例"""
        try:
            # 遍历调用栈，查找 Workflow 实例
            for frame_info in inspect.stack():
                frame = frame_info.frame
                # 检查局部变量中是否有 'self'，且其类型名包含 'Workflow'
                if 'self' in frame.f_locals:
                    obj = frame.f_locals['self']
                    # 检查对象是否是 Workflow 实例（通过类名判断）
                    if hasattr(obj, '__class__') and 'Workflow' in obj.__class__.__name__:
                        return obj
        except Exception:
            # 如果查找失败，返回 None（允许在没有 Workflow 的情况下实例化，用于测试等场景）
            pass
        return None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def check_used(self):
        """记录算子调用（不再阻止重复调用）"""
        # 不再阻止重复调用，只记录调用次数
        pass

    def set_track(self, input_messages: List[Message], output_content: str) -> int:
        """设置算子的输入消息和输出内容，并保存到调用历史

        Returns:
            本次调用的索引
        """
        # 保存本次调用的输入输出关系
        call_index = len(self.call_history)
        call_record = {
            "input_messages": input_messages.copy() if input_messages else [],
            "output_content": output_content,
            "call_index": call_index
        }
        self.call_history.append(call_record)

        # 保持向后兼容：更新最新的输入输出（用于 ExecutionGraph 的旧逻辑）
        self.input_messages = input_messages
        self.output_content = output_content

        return call_index

    def get_call_history(self) -> List[dict]:
        """获取算子的调用历史"""
        return self.call_history

    def get_latest_call(self) -> Optional[dict]:
        """获取最近一次调用的记录"""
        return self.call_history[-1] if self.call_history else None

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        """填充ActionNode

        Args:
            op_class: Pydantic模型类
            prompt: 提示词
            mode: 填充模式
            **extra_kwargs: 额外参数

        Returns:
            填充后的内容
        """
        fill_kwargs = {"context": prompt, "llm": self.llm_instance}
        if mode:
            fill_kwargs["mode"] = mode
        fill_kwargs.update(extra_kwargs)
        node = await ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
        return node.instruct_content.model_dump(), node._raw_output

    def __str__(self) -> str:
        """返回算子的字符串表示"""
        return f"{self.name}(problem={self.problem[:50]}..., fields={self.fields})"


class Custom(Operator):
    """Custom operator for generating solutions"""

    def __init__(self,
                 llm_name: Optional[str] = None,
                 problem: Optional[str] = None,
                 prompt_format: str = "{instruction}{problem}"):
        """Initialize custom operator

        Args:
            llm_name: Name of the large language model
            problem: Initial problem
            prompt_format: Prompt template
        """
        super().__init__(
            llm_name=llm_name,
            problem=problem,
            prompt_format=prompt_format,
            fields=["instruction", "problem"],
            name="Custom"
        )

    async def __call__(self, instruction: Message | str, ignore_used: bool = False):
        """执行自定义算子

        Args:
            instruction: 包含指令内容的消息
            ignore_used: 是否忽略已使用状态

        Returns:
            生成的解决方案消息
        """
        # 检查算子是否已被使用
        if not ignore_used:
            self.check_used()
        # print('Custom Operator Called.')

        if isinstance(instruction, str):
            instruction_message = Message(content=instruction)
        else:
            instruction_message = instruction
        instruction = instruction_message.content if instruction_message else ""

        # 构建提示词
        prompt = self.prompt_format.format(
            instruction=instruction, problem=self.problem)

        # 生成响应
        response, raw_output = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        self.output_content = response.get("response", "")

        # 更新执行记录并获取调用索引
        call_index = self.set_track([instruction_message], self.output_content)

        return Message(source_operator=self, content=self.output_content, raw_output=raw_output, call_index=call_index)


class Review(Operator):
    """Review operator for reviewing and revising solutions"""

    def __init__(self,
                 llm_name: Optional[str] = None,
                 problem: Optional[str] = None,
                 prompt_format: str = REVIEW_PROMPT):
        """Initialize review operator

        Args:
            llm_name: Name of the large language model
            problem: Initial problem
        """
        super().__init__(
            llm_name=llm_name,
            problem=problem,
            prompt_format=prompt_format,
            fields=["problem", "solution"],
            name="Review"
        )

    async def __call__(self, pre_solution: Message | str):
        """执行评审算子

        Args:
            pre_solution: 包含待评审解决方案的消息

        Returns:
            修订后的解决方案消息
        """
        self.check_used()
        # print('Review Operator Called.')

        if isinstance(pre_solution, str):
            pre_solution_message = Message(content=pre_solution)
        else:
            pre_solution_message = pre_solution
        pre_solution = pre_solution_message.content

        # 构建提示词
        prompt = self.prompt_format.format(
            problem=self.problem, solution=pre_solution)

        # 生成响应
        response, raw_output = await self._fill_node(ReviewOp, prompt, mode="xml_fill")
        self.output_content = response.get("revised_solution", "")

        # 更新执行记录并获取调用索引
        call_index = self.set_track(
            [pre_solution_message], self.output_content)

        return Message(source_operator=self, content=self.output_content, raw_output=raw_output, call_index=call_index)


def run_code(code: str):
    """执行Python代码

    Args:
        code: 要执行的Python代码

    Returns:
        代码执行结果
    """
    try:
        # Create a new global namespace
        global_namespace = {}

        disallowed_imports = [
            "os", "sys", "subprocess", "multiprocessing",
            "matplotlib", "seaborn", "plotly", "bokeh", "ggplot",
            "pylab", "tkinter", "PyQt5", "wx", "pyglet"
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace)
        # Assume the code defines a function named 'solve'
        if 'solve' in global_namespace and callable(global_namespace['solve']):
            result = global_namespace['solve']()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"


class Programmer(Operator):
    """Programmer operator for generating and executing code solutions"""

    _executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)

    def __init__(self,
                 llm_name: Optional[str] = None,
                 problem: Optional[str] = None,
                 prompt_format: str = PYTHON_CODE_VERIFIER_PROMPT):
        """Initialize programmer operator

        Args:
            llm_name: Name of the large language model
            problem: Initial problem
        """
        super().__init__(
            llm_name=llm_name,
            problem=problem,
            prompt_format=prompt_format,
            fields=["problem", "analysis", "feedback"],
            name="Programmer"
        )

    async def exec_code(self, code: str, timeout: int = TIME_TO_EXEC_CODE) -> tuple[str, str]:
        """异步执行代码

        Args:
            code: 要执行的代码
            timeout: 超时时间（秒）

        Returns:
            代码执行结果
        """
        loop = asyncio.get_running_loop()
        try:
            # Submit run_code task to the static process pool
            future = loop.run_in_executor(self._executor, run_code, code)
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return "Error", "Code execution timed out"
        except Exception as e:
            return "Error", f"Unknown error: {str(e)}"

    async def _exec_code(self, code: str, timeout: int = TIME_TO_EXEC_CODE) -> tuple[str, str]:
        """异步执行代码（内部实现）

        Args:
            code: 要执行的代码
            timeout: 超时时间（秒）

        Returns:
            代码执行结果
        """
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            try:
                # Submit run_code task to the process pool
                future = loop.run_in_executor(executor, run_code, code)
                # Wait for the task to complete or timeout
                result = await asyncio.wait_for(future, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                # Timeout, attempt to shut down the process pool
                executor.shutdown(wait=False, cancel_futures=True)
                return "Error", "Code execution timed out"
            except Exception as e:
                return "Error", f"Unknown error: {str(e)}"

    async def code_generate(self, problem: str, analysis: str, feedback: str, mode: str) -> dict:
        """生成代码

        Args:
            problem: 问题描述
            analysis: 分析内容
            feedback: 反馈信息
            mode: 生成模式

        Returns:
            代码生成结果
        """
        # 构建提示词
        prompt = self.prompt_format.format(
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )

        # 生成代码
        response, raw_output = await self._fill_node(CodeGenerateOp, prompt, mode, function_name="solve")
        return response, raw_output

    async def __call__(self, analysis: Message | str) -> Message:
        """执行程序员算子

        Args:
            analysis_message: 包含分析内容的消息

        Returns:
            代码执行结果或错误信息的消息
        """
        self.check_used()
        # print('Programmer Operator Called.')

        if isinstance(analysis, str):
            analysis_message = Message(content=analysis)
        else:
            analysis_message = analysis
        analysis_content = analysis_message.content if analysis_message else "None"

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        async def _retryable_call(analysis_content):
            code = None
            execution_result = None
            feedback = ""
            raw_output = None

            for i in range(3):
                # 生成代码
                code_response, raw_output = await self.code_generate(self.problem, analysis_content, feedback, mode="code_fill")
                code = code_response.get("code")

                if not code:
                    self.output_content = "No code generated"
                    # 更新执行记录并获取调用索引
                    call_index = self.set_track(
                        [analysis_message], self.output_content)
                    return Message(source_operator=self, content=self.output_content, raw_output=raw_output, call_index=call_index)

                # 执行代码
                execution_status, execution_result = await self.exec_code(code)

                if execution_status == "Success":
                    self.output_content = f"After executing the following code written by llm agent.\n{code}\nWe have the following output: {execution_result}"
                    # 更新执行记录并获取调用索引
                    call_index = self.set_track(
                        [analysis_message], self.output_content)
                    return Message(source_operator=self, content=self.output_content, raw_output=raw_output, call_index=call_index)
                else:
                    logger.info(
                        f"Execution error on attempt {i + 1}, error message: {execution_result}")
                    feedback = (
                        f"\nThe result of the error from the code you wrote in the previous round:\n"
                        f"Code: {code}\n\nStatus: {execution_status}, {execution_result}"
                    )

            self.output_content = f"After executing the following code written by llm agent.\n{code}\nWe have the following output: {execution_result if execution_result else 'Unknown error'}"
            # 更新执行记录并获取调用索引
            call_index = self.set_track(
                [analysis_message], self.output_content)
            return Message(source_operator=self, content=self.output_content, raw_output=raw_output, call_index=call_index)

        return await _retryable_call(analysis_content)


class ScEnsemble(Operator):
    """Ensemble operator for integrating multiple solutions"""

    def __init__(self,
                 llm_name: Optional[str] = None,
                 problem: Optional[str] = None,
                 prompt_format: Optional[str] = SC_ENSEMBLE_PROMPT):
        """Initialize ensemble operator

        Args:
            llm_name: Name of the large language model
            problem: Initial problem
        """
        super().__init__(
            llm_name=llm_name,
            problem=problem,
            prompt_format=prompt_format,
            fields=["problem", "solutions"],
            name="ScEnsemble"
        )

    async def __call__(self, solutions: List[Message | str]) -> Message:
        """执行评分集成算子

        Args:
            solutions: 包含解决方案的消息列表

        Returns:
            最佳解决方案的消息
        """
        self.check_used()
        # print('ScEnsemble Operator Called.')

        # 转换字符串消息为消息对象
        solution_messages = [msg if isinstance(
            msg, Message) else Message(content=msg) for msg in solutions]

        if not solution_messages:
            self.output_content = "No solutions provided"
            # 更新执行记录并获取调用索引
            call_index = self.set_track(solution_messages, self.output_content)
            return Message(source_operator=self, content=self.output_content, raw_output=None, call_index=call_index)

        # 从消息中提取解决方案内容
        solutions = [msg.content for msg in solution_messages]

        # 构建解决方案文本
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        # 构建提示词
        prompt = self.prompt_format.format(
            problem=self.problem, solutions=solution_text)

        # 生成响应
        response, raw_output = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")
        answer_letter = response.get("solution_letter", "").strip().upper()

        # 获取最佳解决方案
        best_index = answer_mapping.get(answer_letter, 0)
        self.output_content = solutions[best_index]

        # 更新执行记录并获取调用索引
        call_index = self.set_track(solution_messages, self.output_content)

        return Message(source_operator=self, content=self.output_content, raw_output=raw_output, call_index=call_index)
