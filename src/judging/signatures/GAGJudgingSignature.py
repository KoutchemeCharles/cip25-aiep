import dspy
from typing import Dict

class GAGJudgingSignature(dspy.Signature):
    """
    # Context and Input 

    You are a computer science professor for an introductory Python programming course.

    You will be provided with:
    - A problem description.
    - A student solution.
    - A grading rubric used to evaluate the quality of student solutions.
    - A feedback response written by a teaching assistant (TA) to evaluate.
        - The TA was instructed to highlight all positive aspects of the student solution, mentioning every satisfied rubric item.
        - If there were mistakes, the TA was instructed to explain only the first two mistakes, corresponding to the first two unmet rubric items in order.

    Your task is to evaluate the quality of the provided feedback response.
    To help you in that task, you are also given:
    - A reference feedback (an example of high-quality feedback for the same student solution).

    # Evaluation Criteria

    When evaluating the feedback response, consider these two binary dimensions:

    1. **Correctness:**  
        - Does the feedback accurately identify all positive aspects of the student solution?
        - If there are mistakes in the student solution, does the feedback explain only the first two mistakes?
    2. **Helpfulness:**  
        - Is the feedback communicated in a way that a typical student would find useful for improving their solution?
        - Focus on whether the information is actionable, specific, and enables learning.

    **Only feedback that is fully correct can be rated as helpful. If any part of the feedback is incorrect or misleading, helpfulness must be rated as `false`.**

    # Task

    ## Reasoning

    Follow these steps, reasoning carefully at each stage:
    - Assess the correctness of the evaluated feedback using the provided reference feedback.
    - Then, consider helpfulness using once again the reference feedback as a guide for what constitutes high-quality feedback.  
    The evaluated feedback does not need to match the reference exactly; there may be multiple equally valid or even better ways to be helpful. 

    ## Evaluation (output)

    Provide your final assessment as a JSON dictionary with `true` or `false` for each criterion:

    {
        "correctness": true/false,
        "helpfulness": true/false,
    }
    """

    problem_description = dspy.InputField()
    student_code = dspy.InputField()
    items_description = dspy.InputField()
    reference_feedback = dspy.InputField()
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
                reference_feedback=row.teacher_feedback,
                evaluated_feedback=row.feedback,
            ).with_inputs("problem_description", 
                            "student_code", 
                            "items_description",
                            "reference_feedback",
                            "evaluated_feedback"))

        return dspy_dataset

    @classmethod
    def build_single_example(cls, row):
        return dspy.Example(
                problem_description=row.description,
                student_code=row.code,
                items_description=row.items_description,
                reference_feedback=row.teacher_feedback,
                evaluated_feedback=row.feedback,
            ).with_inputs("problem_description", 
                            "student_code", 
                            "items_description",
                            "reference_feedback",
                            "evaluated_feedback")


class JudgingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(GAGJudgingSignature)

    def forward(
        self,
        problem_description,
        student_code,
        items_description,
        reference_feedback,
        evaluated_feedback,
    ):
        return self.predictor(
            problem_description=problem_description,
            student_code=student_code,
            items_description=items_description,
            reference_feedback=reference_feedback,
            evaluated_feedback=evaluated_feedback,
        )

    @classmethod
    def build_dspy_dataset(cls, dataframe):
        return GAGJudgingSignature.build_dspy_dataset(dataframe)

    @classmethod
    def build_single_example(cls, row):
        return GAGJudgingSignature.build_single_example(row)
