from typing import Literal
import ScoreFlow.scripts.MATH.operator as operator
from metagpt.provider.llm_provider_registry import create_llm_instance

class Workflow:
    def __init__(
        self,
        llm_config,
    ) -> None:
        self.llm = create_llm_instance(llm_config)
        

    async def __call__(self, question: str, context: str):
        """
        Implementation of the problem extraction.
        """
        self.custom = operator.Custom(self.llm, None, "")
        prompt = "Given the question: " + question + "\nYou need to directly extract the final answer 'x' and output in the format '\\boxed{x}' (without any modification!) for this question from the following context, no any other character! Context: " + context
        response = await self.custom(instruction=prompt)
        
        return response