import dspy 
from typing import Dict

class GenerateFeedback(dspy.Signature):
    """
    # Context and Input

    You are an inspiring computer science tutor, helping students learn the basic intuition and implementation of code in Python.
    Your job is to provide your students with high-quality, constructive feedback.

    You are provided with:
    - A problem description for a programming task.
    - A solution submitted by a student.
    - A grading rubric that defines key evaluation items.

    # Task and Outputs

    ## Reasoning

    First, follow these steps, reasoning carefully at each stage:
    - Begin by reflecting on the quality of the student's code, considering each rubric item to determine how well it is satisfied.
    - Identify any errors, misconceptions, or omissions in the logic, structure, or output.
    - Explain your reasoning clearly and precisely.

    ## Grading

    Next, provide a grading output: for each rubric item, select the most appropriate option ID that reflects the student's performance.
    Represent this grading as a JSON dictionary where each key is a rubric item ID and the value is the chosen option ID.

    ## Feedback

    Finally, write constructive feedback directly to the student. Your feedback should:
    - Highlight all positive aspects of the student's solution, mentioning every rubric item they satisfied.
    - Identify and explain **only the first two mistakes**, selecting the first two unmet rubric items in the order they appear in the rubric.
    - When a mistake has a clear and simple fix, suggest a specific way the student could address it.
    - Ensure the feedback is clear and encouraging.

    Do not restate the rubric or the problem unless necessary.
    Do not suggest performance improvements.
    Focus on helping the student understand what went wrong and how to improve.
    """

    problem_description = dspy.InputField()
    student_code = dspy.InputField()
    items_description = dspy.InputField()

    reasoning: str = dspy.OutputField()
    grading: Dict[str, int] = dspy.OutputField()
    feedback: str = dspy.OutputField()

class FeedbackModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(GenerateFeedback)

    def forward(self, problem_description, student_code, items_description):
        return self.predictor(
            problem_description=problem_description,
            student_code=student_code,
            items_description=items_description,
        )

    @classmethod
    def build_dspy_dataset(cls, dataframe):
        dspy_dataset = []
        for row in dataframe.itertuples(index=False):
            dspy_dataset.append(
                dspy.Example(
                    problem_description=row.description,
                    student_code=row.code,
                    items_description=row.items_description,
                ).with_inputs(
                    "problem_description", 
                    "student_code", 
                    "items_description"
                ))

        return dspy_dataset