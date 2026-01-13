from pydantic import BaseModel, Field


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")


class CodeGenerateOp(BaseModel):
    code: str = Field(default="", description="Your complete code solution for this problem")


class ScEnsembleOp(BaseModel):
    thought: str = Field(default="", description="The thought in the process of ensemble.") 
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")
    

class ReviewOp(BaseModel):
    thought: str = Field(default="", description="The thought in the process of review.")
    revised_solution: str = Field(default="", description="The revised solution.")
    