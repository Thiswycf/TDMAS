from typing import Literal
import ScoreFlow.scripts.GSM8K.operator as operator
from utils.llm_manager import get_llm_instance


class Workflow:
    def __init__(
        self,
        llm_name,
    ) -> None:
        self.custom = operator.Custom(llm_name=llm_name, problem="")

    async def __call__(self, question: str, model_answer: str, right_answer: str):
        """
        Implementation of the judger.
        """
        prompt = "Given the question: " + question + "\nWe have the ground truth answer: " + right_answer + \
            "\nNow please judge if the following answer is correct: " + model_answer + \
            "\nOnly output 1 as correct, or 0 as wrong, no any other character."

        response = await self.custom(instruction=prompt, ignore_used=True)

        return response
