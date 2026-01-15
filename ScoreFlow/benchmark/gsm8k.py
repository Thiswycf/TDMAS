import re
from typing import Callable, List, Optional, Tuple, Optional
import traceback
from termcolor import cprint

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ScoreFlow.benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger
from utils.message import Message


class GSM8KBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_number(self, text: str) -> Optional[float]:
        matches = re.findall(
            r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            last_number = matches[-1].replace(",", "")
            try:
                return float(last_number)
            except ValueError:
                return None
        else:
            return None

    def calculate_score(self, expected_output: float, prediction: float) -> Tuple[float, float]:
        if prediction is None:
            return 0.0, prediction
        return 1.0 if abs(expected_output - prediction) <= 1e-6 else 0.0, prediction

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=False)
    async def _generate_outputs(self, graph, problem, content_only: bool = False):
        result = await graph(problem=problem)()
        if content_only:
            return getattr(result, "content", None) if isinstance(result, Message) else result
        else:
            return result

    async def _filter(self, extraction, question, answer):
        return await extraction(question, answer)

    def get_input_text(self, problem):
        input_text = problem["question"]
        return input_text

    async def judge_answer(self, judger, question, model_answer, right_answer):
        return await judger(question, model_answer, right_answer)

    def get_graph_input_text(self, problem):
        input_text = problem["question"]
        return input_text

    def get_problem_id(self, problem):
        return problem["id"]

    def convert_to_binary(self, s):
        if s.strip() == "0":
            return 0
        elif s.strip() == "1":
            return 1
        else:
            return 0

    def direct_judge(self, predicted, problem):
        predicted_number = self.extract_number(predicted)
        expected_number = self.extract_number(problem["answer"])
        return self.calculate_score(expected_number, predicted_number)[0]

    async def evaluate_problem(self, problem: dict, extraction: Optional[Callable], judger: Optional[Callable], graph: Callable, zcp: str = None) -> Tuple[str, str, float, float, float]:
        judger = None
        input_text = self.get_input_text(problem)
        expected_output = self.extract_number(problem["answer"])

        try:
            raw_output = await self._generate_outputs(graph, input_text, content_only=False)
            output = getattr(raw_output, "content", None) if isinstance(
                raw_output, Message) else raw_output

            if extraction != None:
                output = await self._filter(extraction, input_text, output)
                output = getattr(output, "content", None) if isinstance(
                    output, Message) else output
            predicted_number = self.extract_number(output)

            if judger == None:
                if zcp == 'accuracy':
                    score, extracted_output = self.calculate_score(
                        expected_output, predicted_number)
                else:
                    from utils.execution_graph import ExecutionGraph
                    evaluate = getattr(__import__(
                        f"ZCMetrics.ZCP.{zcp}", fromlist=["evaluate"]), "evaluate")
                    execution_graph = ExecutionGraph(raw_output)
                    score = evaluate(execution_graph)
            else:
                score = await self.judge_answer(judger, input_text, str(predicted_number), str(expected_output))
                score = self.convert_to_binary(score)

            return input_text, predicted_number, expected_output, score

        except Exception as e:
            # raise e
            traceback.print_exc()
            if 'retry' in str(e):
                print(repr(e.last_attempt.exception()))
            logger.info(
                f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score"]
