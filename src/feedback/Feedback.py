from src.Generate import Generate
from src.feedback.signatures.GenerateFeedback import FeedbackModule

class Feedback(Generate):

    def __init__(self, config, test_run):
        super().__init__(config, test_run, FeedbackModule)
    