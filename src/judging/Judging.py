from src.judging.signatures.GAGJudgingSignature import JudgingModule
from src.Generate import Generate

class Judging(Generate):

    def __init__(self, config, test_run):
        super().__init__(config, test_run, JudgingModule, can_batch=True)
        