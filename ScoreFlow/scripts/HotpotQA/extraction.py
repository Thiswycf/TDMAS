from typing import Literal
import ScoreFlow.scripts.HotpotQA.operator as operator
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
        self.custom = operator.Custom(self.llm, "")
        prompt = "Given the question: " + question + "\nYou need to directly extract the final answer (without any modification!) for this question from the following context, no any other character! The final answer should be concise, accurate, and directly addressing the question. For example, if the answer is a person's name, just provide the name. If the answer is Yes/No, just response Yes/No, no any other character. \nContext: " + context
        response = await self.custom(instruction=prompt)
        
        return response