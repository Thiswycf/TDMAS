from pydantic import BaseModel, Field
from typing import Optional, List, Any


class GenerateOp(BaseModel):
    """生成操作符输出"""
    response: str = Field(default="", description="问题解决方案")


class CodeGenerateOp(BaseModel):
    """代码生成操作符输出"""
    code: str = Field(default="", description="Your complete code solution for this problem")


class ScEnsembleOp(BaseModel):
    """集成操作符输出"""
    thought: str = Field(default="", description="The thought in the process of ensemble.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")
    

class ReviewOp(BaseModel):
    """评审操作符输出"""
    thought: str = Field(default="", description="The thought in the process of review.")
    revised_solution: str = Field(default="", description="The revised solution.")
    