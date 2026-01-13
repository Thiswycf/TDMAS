import re
import torch
from typing import Callable, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.logs import logger


class GPQABenchmark(BaseBenchmark):
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

    def _extract_choice(self, text: Optional[str], options: List[dict]) -> Optional[str]:
        if text is None:
            return None
        content = str(text)
        match = re.search(r"([A-D])", content, re.IGNORECASE)
        if match:
            return match.group(1).upper()

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

    @retry(stop=stop_after_attempt(20), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable):
        input_text = problem["question"]
        expected_output = problem["final_answer"]

        try:
            output, cost, logprob = await self._generate_output(graph, input_text)
            if not output:
                raise ValueError("output is empty")

            score, extracted_output = self.calculate_score(expected_output, output, problem.get("options", []))

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

