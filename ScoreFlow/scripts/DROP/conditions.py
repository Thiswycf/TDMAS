PYTHON_START = '''import asyncio
from typing import Literal
import ScoreFlow.scripts.DROP.operator as operator
from metagpt.provider.llm_provider_registry import create_llm_instance as create

'''

PYTHON_END = '''

    async def __call__(self):
        TIMEOUT = {time}
        return await asyncio.wait_for(self.run_workflow(), timeout=TIMEOUT)'''

START_PROMPT = '''You objective is to output a workflow graph, based on the following template (but you must modify it):

<graph>
class Workflow:
    def __init__(
        self,
        config,
        problem
    ) -> None:
        self.problem = problem
        self.agent = create(config)
        self.custom = operator.Custom(self.agent, self.problem)
        self.sc_ensemble = operator.ScEnsemble(self.agent, self.problem)
        self.answer_generate = operator.AnswerGenerate(self.agent, self.problem)
        self.review = operator.Review(self.agent, self.problem)

    async def run_workflow(self):
        """
        This is a workflow graph.
        """
        solution = await self.answer_generate()
        
        return solution
</graph>


Here's an introduction to operators you can use: (these are all you can use, do not create new operators)
1. Custom:
Usage: Generates anything based on fixed input problem and modifiable instruction.
Format MUST follow: custom(instruction: str) -> str
You can modify the instruction prompt, such like "Can you break down the problem into smaller steps?", "Can you solve this problem by breaking it down into detailed steps and explaining the reasoning behind each step?", "Explain how to solve the problem with clear reasoning for each step", etc. For example:
solution = await self.custom(instruction="Can you solve this problem by breaking it down into detailed steps and explaining the reasoning behind each step?")
The output can serve as the input of next operators or the final output.
2. AnswerGenerate:
Usage: Directly generate answer (including thought) to the given problem.
Format MUST follow: answer_generate() -> str
For example:
solution = await self.answer_generate()
The output can serve as the input of next operators or the final output.
3. ScEnsemble:
Usage: Evaluate every solutions, then select the best solution in the solution list.
Format MUST follow: sc_ensemble(solutions: List[str]) -> str
You can ensemble few solutions, for example:
ensembled_solution = await self.sc_ensemble(solutions=solution_list)
The output can serve as the input of next operators or the final output.
4. Review:
Usage: Given previous solution, Review operator reviews the previous solution to regenerate the solution.
Format MUST follow: review(pre_solution: str) -> str
pre_solution should be solution from previous operator, for example
rev_solution = await self.review(pre_solution=pre_solution)
The output can serve as the input of next operators or the final output.


We have the problem input as follow. But your output graph can not contain any specific information of the this problem.
Question: '''

END_PROMPT = '''

You need to notice:

**Ensure your graph is based on the given template and is correct to avoid runtime failures.** Do NOT import the modules operator and create, which have already been automatically imported. Do not load the operators not provided.

**Introducing multiple operators at appropriate points can enhance performance.** Consider Python's loops (for, list comprehensions) to generate multiple solutions to ensemble.

**Every operator(agent)'s output should contribute to the final return output, otherwise, do not use them.**

**The graph complexity may corelate with the problem complexity.** The graph complexity must between 3 and 8. Considering information loss, complex graphs may yield better results, but insufficient information transmission can omit the solution.

**As for the instruction prompt for custom operator. Your instruction prompt should focus on encouraging agent to think step by step. Do not ask agent to generate multiple (a few, some, etc) answers in one operator's instruction. Also note that different agents are independent, so do not use prompts like "generate another/alternative/different answer", "generate the first/second answer", etc.**

**Your output graph must be optimized and different from the given template graph. Do not output graph without modification!**

**Your output graph can not contain any information of the given problem due to project requirement. All the information of this problem will be given as input "problem" (self.problem) and other agents will execute this workflow.**

Only output the optimized graph (remember to add <graph> and </graph>, and the output can not contain any information of the given problem).

Here is the optimized graph without any problem information: '''


TEMP_AVOID = '''class Workflow:
    def __init__(
        self,
        config,
        problem
    ) -> None:
        self.problem = problem
        self.agent = create(config)
        self.custom = operator.Custom(self.agent, self.problem)
        self.sc_ensemble = operator.ScEnsemble(self.agent, self.problem)
        self.answer_generate = operator.AnswerGenerate(self.agent, self.problem)
        self.review = operator.Review(self.agent, self.problem)

    async def run_workflow(self):
        """
        This is a workflow graph.
        """
        solution = await self.answer_generate()
        
        return solution'''


TEST_PROMPT = "How many children are there? Note that you are given context: there are 3 children playing."

NO_EXCEPTION_LIST = ['''.split(' ')''', '''int(''']

TIME_LIMIT_TEST = 60
TIME_LIMIT = 120
sim_threshold = 0.75

