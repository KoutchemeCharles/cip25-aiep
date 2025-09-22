import dspy
from typing import Dict

class BaseJudgingSignature(dspy.Signature):
    """
    # Context

    You are a computer science professor for an introductory Python programming course.
    Your task is to evaluate the quality of a feedback response provided to a student.

    You will be provided with:
    - A problem description.
    - A student solution.
    - A grading rubric used to evaluate the quality of student solutions.
    - A feedback response written by a teaching assistant (TA) to evaluate.
        - The TA was instructed to highlight all positive aspects of the student solution, mentioning every satisfied rubric item.
        - If there were mistakes, the TA was instructed to explain only the first two mistakes, corresponding to the first two unmet rubric items in order.

    # Evaluation Criteria

    When evaluating the feedback response, consider these four binary dimensions:

    1. **Correctness:**  
        - Does the feedback accurately identify all positive aspects of the student solution?
        - If there are mistakes in the student solution, does the feedback explain only the first two mistakes?
    2. **Helpfulness:**  
        - Is the feedback communicated in a way that a typical student would find useful for improving their solution?
        - Focus on whether the information is actionable, specific, and enables learning.
    3. **Clarity:**  
        - Is the feedback clear and understandable?
    4. **Positivity:**  
        - Is the feedback positive and encouraging in tone?

    **Only feedback that is fully correct can be rated as helpful. If any part of the feedback is incorrect or misleading, helpfulness must be rated as `false`.**

    # Task

    ## Reasoning

    Follow these steps, reasoning carefully at each stage:
    - Grade the student solution according to the provided rubric.
    - Write your own feedback for the student solution, using your filled-in rubric and following the same instructions given to the teaching assistant.
    - Assess the correctness of the evaluated (TA) feedback using your filled-in rubric and written feedback.
    - Consider helpfulness, clarity, and positivity by comparing the evaluated feedback to your own feedback as a standard of high quality.

    ## Evaluation

    Provide your final assessment as a JSON dictionary with `true` or `false` for each criterion:

    {
        "correctness": true/false,
        "helpfulness": true/false,
        "clarity": true/false,
        "positivity": true/false
    }
    """

    problem_description = dspy.InputField()
    student_code = dspy.InputField()
    items_description = dspy.InputField()
    evaluated_feedback = dspy.InputField()

    reasoning = dspy.OutputField()
    evaluation: Dict[str, bool] = dspy.OutputField()

    @classmethod
    def build_dspy_dataset(cls, dataframe):
        dspy_dataset = []
        for row in dataframe.itertuples(index=False):
            dspy_dataset.append(dspy.Example(
                problem_description=row.description,
                student_code=row.code,
                items_description=row.items_description,
                evaluated_feedback=row.feedback
            ).with_inputs("problem_description", 
                            "student_code", 
                            "items_description",
                            "evaluated_feedback"))

        return dspy_dataset

