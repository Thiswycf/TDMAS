import re
from typing import Callable, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ScoreFlow.benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger
from utils.message import Message


class AIME24Benchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def _normalize_answer(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        raw = str(text).strip().replace(",", "")
        boxed = re.findall(r"\\boxed\{([^}]*)\}", raw)
        candidate = boxed[-1] if boxed else raw
        candidate = candidate.strip()

        numbers = re.findall(r"-?\d+(?:\.\d+)?", candidate)
        if numbers:
            num = numbers[-1]
            if re.fullmatch(r"-?\d+", num):
                # 去掉多余前导零，保持符号
                num = str(int(num))
            return num
        return candidate

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        expected = self._normalize_answer(expected_output)
        predicted = self._normalize_answer(prediction)
        if expected is None or predicted is None:
            return 0.0, predicted
        return (1.0 if expected == predicted else 0.0), predicted

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=False,
    )
    async def _generate_outputs(self, graph, problem, content_only: bool = False):
        result = await graph(problem=problem)()
        if content_only:
            return getattr(result, "content", None) if isinstance(result, Message) else result
        else:
            return result

    async def _filter(self, extraction, question, answer):
        return await extraction(question, answer)

    def get_input_text(self, problem):
        return problem["question"]

    def get_graph_input_text(self, problem):
        return problem["question"]

    def get_problem_id(self, problem):
        return problem.get("id", "")

    async def evaluate_problem(
        self,
        problem: dict,
        extraction: Optional[Callable],
        judger: Optional[Callable],
        graph: Callable,
        zcp: str = None,
    ) -> Tuple[str, str, str, float]:
        judger = None
        input_text = self.get_input_text(problem)
        expected_output = problem["answer"]

        try:
            raw_output = await self._generate_outputs(graph, input_text, content_only=False)
            output = getattr(raw_output, "content", None) if isinstance(
                raw_output, Message) else raw_output

            if extraction is not None:
                output = await self._filter(extraction, input_text, output)
                output = getattr(output, "content", None) if isinstance(
                    output, Message) else output

            score, extracted_output = self.calculate_score(
                expected_output, output)
            return input_text, output, expected_output, score

        except Exception as e:
            logger.info(
                f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score"]
