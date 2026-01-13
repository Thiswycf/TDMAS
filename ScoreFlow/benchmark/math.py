import inspect
import re
from math import isclose
from typing import Any, Callable, List, Tuple, Optional

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ScoreFlow.benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger


class MATHBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_model_answer(self, text: str) -> str:
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        else:
            return 0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        if str(prediction) == str(reference):
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=1e-3)
        except:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except:
            pass

        return False

    def is_digit(self, num):
        return self.parse_digits(num) is not None

    def parse_digits(self, num):
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

    def symbolic_equal(self, a, b):
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except:
            pass
        return False

    def get_function_code(self, func):
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_outputs(self, graph):
        return await graph()

    def get_input_text(self, problem):
        input_text = problem["problem"]
        return input_text

    def get_graph_input_text(self, problem):
        input_text = problem["problem"]
        return input_text

    async def _filter(self, extraction, question, answer):
        return await extraction(question, answer)

    async def judge_answer(self, judger, question, model_answer, right_answer):
        return await judger(question, model_answer, right_answer)

    def get_problem_id(self, problem):
        return problem["problem"]

    def convert_to_binary(self, s):
        if s.strip() == "0":
            return 0
        elif s.strip() == "1":
            return 1
        else:
            return 0

    async def evaluate_problem(self, problem: dict, extraction: Optional[Callable], judger: Optional[Callable], graph: Callable) -> Tuple[str, str, str, int, float]:
        input_text = self.get_input_text(problem)
        expected_output = problem["solution"]

        try:
            output = await self._generate_outputs(graph)
            if extraction != None:
                output = await self._filter(extraction, input_text, output)

            if judger == None:
                uni_score, extracted_output = self.calculate_score(expected_output, output)
            else:
                extracted_output = self.extract_model_answer(output)
                expected_answer = self.extract_model_answer(expected_output)
                uni_score = await self.judge_answer(judger, input_text, extracted_output, expected_answer)
                uni_score = self.convert_to_binary(uni_score)

            return input_text, output, expected_output, uni_score

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score"]
