from typing import Literal
import ScoreFlow.scripts.HotpotQA.operator as operator
from metagpt.provider.llm_provider_registry import create_llm_instance

class Workflow:
    def __init__(
        self,
        llm_config,
    ) -> None:
        self.llm = create_llm_instance(llm_config)
        

    async def __call__(self, question: str, context: str, model_answer: str, right_answer: str):
        """
        Implementation of the judger.
        """
        self.custom = operator.Custom(self.llm, "")
        prompt = "Given the question: " + question + "\nWe have the ground truth answer: " + right_answer + "\nNow please judge if the following answer is correct (do not try to answer the problem by yourself): " + model_answer + "\nNote that you have the context information: " + context + "\nOnly output 1 as correct, or 0 as wrong, no any other character."
        
        response = await self.custom(instruction=prompt)
        
        return response