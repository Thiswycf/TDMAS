import re
import torch
from typing import Callable, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.logs import logger


class AIME24Benchmark(BaseBenchmark):
    def __init__(
        self,
        name: str,
        file_path: str,
        log_path: str,
        batch_size: int,
        controller: torch.nn.Module,
        operator_embeddings,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__(name, file_path, log_path, batch_size, controller, operator_embeddings, optimizer)

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
                num = str(int(num))
            return num
        return candidate

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        expected = self._normalize_answer(expected_output)
        predicted = self._normalize_answer(prediction)
        if expected is None or predicted is None:
            return 0.0, predicted
        return (1.0 if expected == predicted else 0.0), predicted

    @retry(stop=stop_after_attempt(20), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable):
        input_text = problem["question"]
        expected_output = problem["answer"]

        try:
            output, cost, logprob = await self._generate_output(graph, input_text)
            if not output:
                raise ValueError("output is empty")

            score, extracted_output = self.calculate_score(expected_output, output)

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            return input_text, output, expected_output, score, cost, logprob

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return (
                input_text,
                str(e),
                expected_output,
                0.0,
                0.0,
                torch.tensor(0.0, dtype=torch.float32, device=self.device),
            )

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost", "logprob"]

