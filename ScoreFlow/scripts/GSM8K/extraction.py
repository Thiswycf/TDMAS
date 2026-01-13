from typing import Literal
import ScoreFlow.scripts.GSM8K.operator as operator
from utils.llm_manager import get_llm_instance


class Workflow:
    def __init__(
        self,
        llm_name,
    ) -> None:
        self.custom = operator.Custom(llm_name=llm_name, problem="")

    async def __call__(self, question: str, context: str):
        """
        Implementation of the problem extraction.
        """
        prompt = "Given the question: " + question + \
            "\nYou need to directly extract the numerical final answer (without any modification!) for this question from the following context, no any other character! Context: " + context
        response = await self.custom(instruction=prompt, ignore_used=True)

        return response
