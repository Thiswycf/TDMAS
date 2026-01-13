import re
import string
from collections import Counter
from typing import Callable, List, Tuple, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from ScoreFlow.benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger


class DROPBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def normalize_answer(self, s: str) -> List[str]:
        """
        Normalize answers for evaluation.
        """

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
        """
        Compute the F1 score between prediction and ground truth answers.
        """
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

    def get_passage(self, problem):
        return re.search(r"Passage:(.*?)Question:", problem["context"], re.DOTALL).group(1).strip()

    def get_input_text(self, problem):
        question = self.get_graph_input_text(problem)
        passage = self.get_passage(problem)
        input_text = "\nStep by Step solve the following Question Answering problem:" + question + "\nNote that you are given passage:\n" + passage + "\n\n"
        return input_text

    def get_graph_input_text(self, problem):
        return re.search(r"Question:(.*?)Answer:", problem["context"], re.DOTALL).group(1).strip()

    def get_problem_id(self, problem):
        return problem["id"]

    def convert_to_binary(self, s):
        if s.strip() == "0":
            return 0
        elif s.strip() == "1":
            return 1
        else:
            return 0

    async def evaluate_problem(self, problem: dict, extraction: Optional[Callable], judger: Optional[Callable], graph: Callable):
        input_text = problem["context"]
        expected_output = problem["ref_text"]
        answers = expected_output.split("|")
        question = self.get_graph_input_text(problem)

        try:
            output = await self._generate_outputs(graph)
            if extraction != None:
                output = await self._filter(extraction, question, output)

            if judger == None:
                f1_scores = []
                for answer in answers:
                    if answer.strip() != "":
                        output_parts = output.split("|")
                        for output_part in output_parts:
                            f1_score, _ = self.calculate_score(answer, output_part)
                            f1_scores.append(f1_score)
                uni_score = max(f1_scores)
            else:
                answer_list = ""
                for answer in answers:
                    if answer.strip() != "":
                        if answers.index(answer) != len(answers) - 1:
                            answer_list = answer_list + answer + ", or "
                        else:
                            answer_list = answer_list + answer
                score = await self.judge_answer(judger, question, self.get_passage(problem), output, answer_list)
                uni_score = self.convert_to_binary(score)

            return input_text, output, expected_output, uni_score

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score"]
