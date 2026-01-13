import re
from typing import Callable, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ScoreFlow.benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger
from utils.message import Message


class GPQABenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def _extract_choice(self, text: Optional[str], options: List[dict]) -> Optional[str]:
        if text is None:
            return None
        content = str(text)
        # 1) 直接找 A-D
        match = re.search(r"([A-D])", content, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 2) 用选项文本匹配
        lowered = content.lower()
        found = []
        for opt in options:
            if opt["text"] and opt["text"].lower() in lowered:
                found.append(opt["label"])
        if len(found) == 1:
            return found[0]
        return None

    def calculate_score(self, expected_output: str, prediction: str, options: List[dict]) -> Tuple[float, str]:
        predicted = self._extract_choice(prediction, options)
        if predicted is None:
            return 0.0, ""
        return (1.0 if predicted == expected_output else 0.0), predicted

    @retry(
        stop=stop_after_attempt(3),
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
        expected_output = problem["final_answer"]

        try:
            raw_output = await self._generate_outputs(graph, input_text, content_only=False)
            output = getattr(raw_output, "content", None) if isinstance(
                raw_output, Message) else raw_output

            if extraction is not None:
                output = await self._filter(extraction, input_text, output)
                output = getattr(output, "content", None) if isinstance(
                    output, Message) else output

            score, extracted_output = self.calculate_score(
                expected_output, output, problem.get("options", []))
            return input_text, output, expected_output, score

        except Exception as e:
            logger.info(
                f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score"]
