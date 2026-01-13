import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple

from tenacity import retry, stop_after_attempt, wait_fixed

from ScoreFlow.scripts.HotpotQA.operator_an import *
from ScoreFlow.scripts.HotpotQA.op_prompt import *
from metagpt.actions.action_node import ActionNode
from metagpt.llm import LLM
from metagpt.logs import logger
import re


class Operator:
    def __init__(self, llm: LLM):
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        fill_kwargs = {"context": prompt, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        fill_kwargs.update(extra_kwargs)
        node = await ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
        return node.instruct_content.model_dump()


class Custom(Operator):
    def __init__(self, llm: LLM, problem: str = None):
        super().__init__(llm)
        self.problem = problem

    async def __call__(self, instruction):
        
        prompt = instruction + self.problem
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        
        return response["response"]
    
class AnswerGenerate(Operator):
    def __init__(self, llm: LLM, problem: str = None):
        super().__init__(llm)
        self.problem = problem

    async def __call__(self) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=self.problem)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
        answer = response.get("answer", "")
        thought = response.get("thought", "")
        final_response = thought + "\n So we have the final results: " +  answer  
        
        return final_response

class Review(Operator):
    def __init__(self, llm: LLM, problem: str = None):
        super().__init__(llm)
        self.problem = problem

    async def __call__(self, pre_solution):
        
        prompt = REVIEW_PROMPT.format(problem=self.problem, solution=pre_solution)
        response = await self._fill_node(ReviewOp, prompt, mode="xml_fill")
        answer = response.get("revised_solution", "")

        return answer


class ScEnsemble(Operator):

    def __init__(self, llm: LLM, problem: str = None):
        super().__init__(llm)
        self.problem = problem

    async def __call__(self, solutions: List[str]):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(problem=self.problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")
        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()
        
        return solutions[answer_mapping[answer]]