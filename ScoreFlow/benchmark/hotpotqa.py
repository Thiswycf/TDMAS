import re
import string
from collections import Counter
from typing import Callable, List, Tuple, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ScoreFlow.benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger


class HotpotQABenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def normalize_answer(self, s: str) -> str:
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, prediction
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, prediction

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_outputs(self, graph):
        return await graph()

    async def _filter(self, extraction, question, answer):
        return await extraction(question, answer)

    async def judge_answer(self, judger, question, context, model_answer, right_answer):
        return await judger(question, context, model_answer, right_answer)

    def get_context_str(self, problem):
        context_str = ""
        for item in problem["context"]:
            title = item[0]
            paragraphs = ""
            for para in item[1]:
                paragraphs = paragraphs + para
            context_str = context_str + title + ": " + paragraphs + "\n"
        return context_str
        
    def get_input_text(self, problem):
        context_str = self.get_context_str(problem)
        input_text = "\nStep by Step solve the following Question Answering problem:" + problem["question"] + "\nNote that you are given context:\n" + context_str + "\n\n"
        return input_text

    def get_graph_input_text(self, problem):
        return problem["question"]

    def get_problem_id(self, problem):
        return problem["_id"]

    def convert_to_binary(self, s):
        if s.strip() == "0":
            return 0
        elif s.strip() == "1":
            return 1
        else:
            return 0
    
    async def evaluate_problem(self, problem: dict, extraction: Optional[Callable], judger: Optional[Callable], graph: Callable) -> Tuple[str, str, str, str, float, float]:
        question = problem["question"]
        expected_output = problem["answer"]
        input_text = self.get_input_text(problem)
        
        try:
            output = await self._generate_outputs(graph)
            if extraction != None:
                output = await self._filter(extraction, question, output)
            if judger == None:
                score, extracted_output = self.calculate_score(expected_output, output)
            else:
                ratio = len(output)/len(expected_output)
                if ratio >= 10 or ratio <= 1/10:
                    score = 0
                else:
                    score = await self.judge_answer(judger, question, self.get_context_str(problem), output, expected_output)
                    score = self.convert_to_binary(score)
            
            return question, output, expected_output, score

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return question, str(e), expected_output, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score"]
