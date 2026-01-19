"""
Prompt template definitions
"""
from typing import Optional, Tuple, Union, List, Dict

# First question prompt template (leader agent l's first round input)
FIRST_QUESTION_PROMPT_TEMPLATE = """I have a question as following:
{question}
You need to solve it and provide feedback. You have two ways to solve it: directly answer (i.e., directly return the answer or runnable Python code that directly prints the answer) and decompose the question into several sub-questions.

If you choose to decompose into sub-questions, each sub-question must contain all necessary context and relevant information required to answer it. Do not ask fragmented questions that lack essential context.

Your feedback needs to include a score (out of 100) and an evaluation text (i.e., whether the question is well-formulated, whether there are unreasonable or ambiguous aspects).

If you choose to provide Python code:
- The code should be self-contained and executable
- Use print() to output the final answer
- Do not use input() or any interactive functions
- Avoid using external libraries that are unnecessary or may not be available
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
   [Sub-question 1 - must include all necessary context and relevant information from the original question]
   </subquestion>
   <subquestion id="2">
   [Sub-question 2 - must include all necessary context and relevant information from the original question]
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

Additionally, if you think the question {original_question} has been ultimately answered, then answer directly (i.e., directly return the answer or runnable Python code that directly prints the answer); otherwise, please ask follow-up questions about the previously proposed questions or propose new sub-questions, and do not ask about sub-questions that have already been well answered.

When you ask follow-up questions, if the question is about a previously asked sub-question, you MUST reuse the same id as that sub-question; if it is a completely new sub-question, an incrementing ID that has not been used before must be used.

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
   [New sub-question or follow-up question - must include all necessary context and relevant information needed to answer it]
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
NON_FIRST_QUESTION_PROMPT_TEMPLATE = """Your response has been evaluated. Here is your supervisor's feedback:
<feedback>
Score: {feedback_score}
Evaluation: {feedback_text}
</feedback>

Now you need to solve the new question as following:
{question}
You need to solve it and provide feedback. You have two ways to solve it: 
directly answer (i.e., directly return the answer or runnable Python code that directly prints the answer) 
and decompose the question into several sub-questions.

If you choose to decompose into sub-questions, each sub-question must contain all necessary context and relevant 
information required to answer it. Do not ask fragmented questions that lack essential context.

Your feedback needs to include a score (out of 100) and an evaluation text (i.e., whether the question is 
well-formulated, whether there are unreasonable or ambiguous aspects).

If you choose to provide Python code:
- The code should be self-contained and executable
- Use print() to output the final answer
- Do not use input() or any interactive functions
- Avoid using external libraries that are unnecessary or may not be available
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
   [Sub-question 1 - must include all necessary context and relevant information from the original question]
   </subquestion>
   <subquestion id="2">
   [Sub-question 2 - must include all necessary context and relevant information from the original question]
   </subquestion>
   ...
   </subquestions>
   <score>Score (0-100)</score>
   <evaluation>Evaluation text</evaluation>
"""


def format_first_question_prompt(question: str,
                                 use_chat_template: bool = True,
                                 ) -> Union[List[str], List[Dict[str, str]]]:
    """Format the first question prompt"""
    prompt = FIRST_QUESTION_PROMPT_TEMPLATE.format(question=question)
    if use_chat_template:
        return [{"role": "user", "content": prompt}]
    else:
        return [prompt]


def format_reply_prompt(original_question: str, subquestion_replies: list, num_subquestions: int, use_chat_template: bool = True) -> Union[List[str], List[Dict[str, str]]]:
    """Format the reply prompt

    Args:
        original_question: Original question
        subquestion_replies: List of sub-question replies, each element contains sub-question ID, answer, score, and evaluation
        num_subquestions: Number of sub-questions
    """
    reply_texts = []
    for i, reply in enumerate(subquestion_replies, 1):
        reply_text = f"{i}. Sub-question {reply.get('subq_id', i)}: {reply.get('question', '')}\n"
        reply_text += f"   Answer: {reply.get('answer', '')}\n"
        reply_text += f"   Score: {reply.get('score', 'N/A')}\n"
        reply_text += f"   Evaluation: {reply.get('evaluation', 'N/A')}"
        reply_texts.append(reply_text)

    subquestion_replies_text = "\n\n".join(reply_texts)

    # Calculate next sub-question ID
    next_id = num_subquestions + 1

    prompt = REPLY_PROMPT_TEMPLATE.format(
        original_question=original_question,
        subquestion_replies=subquestion_replies_text,
        num_subquestions=num_subquestions,
        next_id=next_id
    )
    if use_chat_template:
        return [{"role": "user", "content": prompt}]
    else:
        return [prompt]


def format_non_first_question_prompt(
    question: str,
    previous_conversation: Union[List[str], List[Dict[str, str]]],
    feedback: Optional[Tuple[float, str]] = None,
    use_chat_template: bool = True,
) -> Union[List[str], List[Dict[str, str]]]:
    """Format the non-first question prompt

    Args:
        question: Current question
        previous_conversation: Previous conversation, each element can be a string (no use chat template) or a dictionary (use chat template)
        feedback: Supervisor feedback for the previous round (score + evaluation)
        use_chat_template: Kept for compatibility; when True returns multi-round chat messages.
    """
    feedback_score, feedback_text = feedback

    user_content = NON_FIRST_QUESTION_PROMPT_TEMPLATE.format(
        question=question,
        feedback_score=feedback_score,
        feedback_text=feedback_text,
    )

    # Return multi-round conversation format compatible with VLLMAdapter:
    # - odd number of messages
    # - roles alternate user/assistant/user/...
    # - last message is user
    if use_chat_template:
        conversation = previous_conversation + \
            [{"role": "user", "content": user_content}]
    else:
        conversation = previous_conversation + [user_content]

    # Default: single-round user message (still a valid multi-round structure of length 1)
    return conversation


# Debug prompt template (when code execution fails)
DEBUG_PROMPT_TEMPLATE = """The code you provided previously encountered an error during execution. The error message is as follows:
<error>
{error_message}
</error>

Please fix the issues in your code and provide the corrected executable code. Your code should follow the following requirements:
- The code should be self-contained and executable
- Use print() to output the final answer
- Do not use input() or any interactive functions
- Avoid using external libraries that may not be available
- The code should produce a clear, final output that answers the question

Please output the corrected code in the following format and <score> and <evaluation> are NOT needed:

<answer>
```python
[Corrected executable code]
```
</answer>
"""


def format_debug_prompt(
    error_message: str,
    previous_conversation: Union[List[str], List[Dict[str, str]]],
    use_chat_template: bool = True,
) -> Union[List[str], List[Dict[str, str]]]:
    """Format the debug prompt when code execution fails

    Args:
        question: Original question
        code: The code that failed to execute
        error_message: Error message from code execution
        previous_conversation: Previous conversation history
        use_chat_template: Whether to use chat template format
    """
    user_content = DEBUG_PROMPT_TEMPLATE.format(
        error_message=error_message,
    )

    # Return multi-round conversation format compatible with VLLMAdapter
    if use_chat_template:
        conversation = previous_conversation + [{"role": "user", "content": user_content}]
    else:
        conversation = previous_conversation + [user_content]

    return conversation
