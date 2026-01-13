"""
Prompt template definitions
"""

# First question prompt template (leader agent l's first round input)
FIRST_QUESTION_PROMPT_TEMPLATE = """I have a question: {question}. You need to solve it and provide feedback. You have two ways to solve it: directly answer (i.e., directly return the answer or runnable Python code that can directly return the answer) and decompose the question into several sub-questions.

Your feedback needs to include a score (out of 100) and an evaluation text (i.e., whether the question is well-formulated, whether there are unreasonable or ambiguous aspects).

If you choose to provide Python code:
- The code should be self-contained and executable
- Use print() to output the final answer
- Do not use input() or any interactive functions
- Avoid using external libraries that may not be available
- The code should produce a clear, final output that answers the question

Format output requirements as follows:
1. If you choose to answer directly, please use the following format:
   <answer>
   [Your answer or Python code (wrap code in ```python ... ``` if using code blocks)]
   </answer>
   <score>Score (0-100)</score>
   <evaluation>Evaluation text</evaluation>

2. If you choose to decompose into sub-questions, please use the following format:
   <subquestions>
   <subquestion id="1">
   [Sub-question 1]
   </subquestion>
   <subquestion id="2">
   [Sub-question 2]
   </subquestion>
   ...
   </subquestions>
   <score>Score (0-100)</score>
   <evaluation>Evaluation text</evaluation>
"""

# Reply prompt template (leader agent l's non-first round input)
REPLY_PROMPT_TEMPLATE = """For the {num_subquestions} sub-questions you proposed, the current replies are as follows (including answers and feedback):
{subquestion_replies}

Please score (out of 100) and evaluate each of their answers based on the sub-question replies (i.e., evaluate their answer performance).

Additionally, if you think the question {original_question} has been ultimately answered, then answer directly (i.e., directly return the answer or runnable Python code that can directly return the answer); otherwise, please ask follow-up questions about the previously proposed questions or propose new sub-questions, and do not ask about sub-questions that have already been well answered.

If you choose to provide Python code:
- The code should be self-contained and executable
- Use print() to output the final answer
- Do not use input() or any interactive functions
- Avoid using external libraries that may not be available
- The code should produce a clear, final output that answers the question

Format output requirements as follows:
1. If the question has been answered, please use the following format:
   <answer>
   [Your final answer or Python code (wrap code in ```python ... ``` if using code blocks)]
   </answer>
   <subquestion_scores>
   <subquestion id="1">Score (0-100)</subquestion>
   <subquestion id="2">Score (0-100)</subquestion>
   ...
   </subquestion_scores>
   <subquestion_evaluations>
   <subquestion id="1">Evaluation text</subquestion>
   <subquestion id="2">Evaluation text</subquestion>
   ...
   </subquestion_evaluations>

2. If you need to continue asking questions, please use the following format:
   <subquestions>
   <subquestion id="{next_id}">
   [New sub-question or follow-up question]
   </subquestion>
   ...
   </subquestions>
   <subquestion_scores>
   <subquestion id="1">Score (0-100)</subquestion>
   <subquestion id="2">Score (0-100)</subquestion>
   ...
   </subquestion_scores>
   <subquestion_evaluations>
   <subquestion id="1">Evaluation text</subquestion>
   <subquestion id="2">Evaluation text</subquestion>
   ...
   </subquestion_evaluations>
"""

# Non-first question prompt template (leader agent l receives a question again after submitting the first question)
NON_FIRST_QUESTION_PROMPT_TEMPLATE = """For your replied answer, the current reply is as follows (including answer and feedback):
{previous_reply}

I have a question: {question}. You need to solve it and provide feedback. You have two ways to solve it: directly answer (i.e., directly return the answer or runnable Python code that can directly return the answer) and decompose the question into several sub-questions.

Your feedback needs to include a score (out of 100) and an evaluation text (i.e., whether the question is well-formulated, whether there are unreasonable or ambiguous aspects).

If you choose to provide Python code:
- The code should be self-contained and executable
- Use print() to output the final answer
- Do not use input() or any interactive functions
- Avoid using external libraries that may not be available
- The code should produce a clear, final output that answers the question

Format output requirements as follows:
1. If you choose to answer directly, please use the following format:
   <answer>
   [Your answer or Python code (wrap code in ```python ... ``` if using code blocks)]
   </answer>
   <score>Score (0-100)</score>
   <evaluation>Evaluation text</evaluation>

2. If you choose to decompose into sub-questions, please use the following format:
   <subquestions>
   <subquestion id="1">
   [Sub-question 1]
   </subquestion>
   <subquestion id="2">
   [Sub-question 2]
   </subquestion>
   ...
   </subquestions>
   <score>Score (0-100)</score>
   <evaluation>Evaluation text</evaluation>
"""


def format_first_question_prompt(question: str) -> str:
    """Format the first question prompt"""
    return FIRST_QUESTION_PROMPT_TEMPLATE.format(question=question)


def format_reply_prompt(original_question: str, subquestion_replies: list, num_subquestions: int) -> str:
    """Format the reply prompt

    Args:
        original_question: Original question
        subquestion_replies: List of sub-question replies, each element contains sub-question ID, answer, score, and evaluation
        num_subquestions: Number of sub-questions
    """
    reply_texts = []
    for i, reply in enumerate(subquestion_replies, 1):
        reply_text = f"{i}. Sub-question {reply.get('id', i)}: {reply.get('question', '')}\n"
        reply_text += f"   Answer: {reply.get('answer', '')}\n"
        reply_text += f"   Score: {reply.get('score', 'N/A')}\n"
        reply_text += f"   Evaluation: {reply.get('evaluation', 'N/A')}"
        reply_texts.append(reply_text)

    subquestion_replies_text = "\n\n".join(reply_texts)

    # Calculate next sub-question ID
    next_id = num_subquestions + 1

    return REPLY_PROMPT_TEMPLATE.format(
        original_question=original_question,
        subquestion_replies=subquestion_replies_text,
        num_subquestions=num_subquestions,
        next_id=next_id
    )


def format_non_first_question_prompt(question: str, previous_reply: dict) -> str:
    """Format the non-first question prompt

    Args:
        question: Current question
        previous_reply: Previous reply, containing answer, score, and evaluation
    """
    reply_text = f"Answer: {previous_reply.get('answer', '')}\n"
    reply_text += f"Score: {previous_reply.get('score', 'N/A')}\n"
    reply_text += f"Evaluation: {previous_reply.get('evaluation', 'N/A')}"

    return NON_FIRST_QUESTION_PROMPT_TEMPLATE.format(
        question=question,
        previous_reply=reply_text
    )
