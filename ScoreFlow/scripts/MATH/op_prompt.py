SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. Carefully go through every solutions, then determine the most accurate answer. They are as follows:
{solutions}

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most accurate solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

SC_ENSEMBLE_PROCESS_PROMPT = """
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Now we carefully evaluate these solutions and identify the most accurate answer.
"""

REVIEW_PROMPT = """
Given the question described as follows: {problem}
We already have one solution as follow: 
{solution}

Now you need to evaluate this solution very carefully, then give the revised solution. When you evaluate, you need to critique this solution based on four dimensions: "Logical correctness", "Accuracy of calculation", "Potential misunderstanding of the problem", and "Level of Detail". Unless the given solution is absolutely perfect, you should rewrite the solution based on your revision and given solution.

Provide a critique for each dimension in the "thought" field, and provide the revised answer in your "revised_solution" field.
"""

PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

Problem description: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
"""