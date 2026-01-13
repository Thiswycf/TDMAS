from typing import Literal

import ScoreFlow.scripts.GPQA.operator as operator
from utils.llm_manager import get_llm_instance


class Workflow:
    """
    为选择题提取单个选项字母（A/B/C/D）。
    """

    def __init__(
        self,
        llm_name,
    ) -> None:
        self.custom = operator.Custom(llm_name=llm_name, problem="")

    async def __call__(self, question: str, context: str):
        """
        仅返回一个大写选项字母 A/B/C/D。
        """
        prompt = (
            "You are given a multiple-choice question and a model's answer.\n"
            "Your task is to extract the single option letter (A, B, C, or D) "
            "that the model selected.\n"
            "You MUST output only one uppercase letter A/B/C/D, with no extra text.\n\n"
            f"Question and options:\n{question}\n\n"
            f"Model answer:\n{context}"
        )
        response = await self.custom(instruction=prompt, ignore_used=True)
        return response
