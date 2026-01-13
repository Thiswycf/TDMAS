from typing import Literal
import ScoreFlow.scripts.DROP.operator as operator
from metagpt.provider.llm_provider_registry import create_llm_instance

class Workflow:
    def __init__(
        self,
        llm_config,
    ) -> None:
        self.llm = create_llm_instance(llm_config)
        

    async def __call__(self, question: str, passage: str, model_answer: str, right_answers: str):
        """
        Implementation of the judger.
        """
        self.custom = operator.Custom(self.llm, "")
        prompt = "Given the question: " + question + "\nWe have the ground truth answer (may be more than more one): " + right_answers + "\nNow please judge if the following answer is correct (do not try to answer the problem by yourself): " + model_answer + "\nNote that you have the context information: " + passage + "\nOnly output 1 as correct, or 0 as wrong, no any other character."
        
        response = await self.custom(instruction=prompt)
        
        return response