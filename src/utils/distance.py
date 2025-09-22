import ast
from tokenize_rt import src_to_tokens
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
    
def rouge1_dist(buggy, correction):
    scores = scorer.score(buggy, correction)['rouge1']
    return 1 - scores[-1]

def rouge2_dist(buggy, correction):
    scores = scorer.score(buggy, correction)['rouge2']
    return 1 - scores[-1]

def rougel_dist(buggy, correction):
    scores = scorer.score(buggy, correction)['rougeL']
    return 1 - scores[-1]

def rougelcsum_dist(buggy, correction, get_score=False):
    scores = scorer.score(buggy, correction)['rougeLsum']
    if get_score: return scores[-1]
    return 1 - scores[-1]