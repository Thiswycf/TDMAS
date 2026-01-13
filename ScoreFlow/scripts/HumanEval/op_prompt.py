SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. Carefully go through every solutions, then determine the most accurate answer. They are as follows:
{solutions}

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most accurate solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

CustomCodeGenerate_PROMPT = """Note that you should think carefully based on four dimensions: "Logical correctness", "Consideration of all situations", "Potential misunderstanding of the problem", and "If the function name is the given entry_point"."""

REFLECTION_ON_PUBLIC_TEST_PROMPT = """
Given a code problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution.: 
### problem
{problem}

### Code Solution
{solution}

### Execution Result
{exec_pass}

#### Failed Test Case
{test_fail}

Please provide a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases. Remember to keep the entry_point function name: {entry_point}. You MUST NOT give a code with dead loop!
"""


