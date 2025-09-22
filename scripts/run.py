"""
Run an experiment from a config file.
"""

from argparse import ArgumentParser
from src.judging.Judging import Judging
from src.feedback.Feedback import Feedback
from src.utils.files import read_config
from src.utils.core import set_seed
from src.trl.SFT import SFT
from src.trl.DPO import DPO

def parse_args():
    parser = ArgumentParser(description="Running experiments")
    parser.add_argument("--config", required=True,
                        help="Path towards the configuration file")
    parser.add_argument('--test_run',
                        help="Whether to do a test run to ensure the pipeline works without issues",
                        action="store_true")

    return parser.parse_args()

def load_experiment(name):
    if "grade" in name or "feedback" in name:
        experiment = Feedback 
    elif "sft" in name:
        experiment = SFT  
    elif "dpo" in name:
        experiment = DPO 
    elif "judge" in name:
        experiment = Judging
    else:
        raise ValueError(f"Unknown experiment for config {name}")
        
    return experiment


def main():
    args = parse_args()
    config = read_config(args.config)
    set_seed(config.seed)

    EXP_CLASS = load_experiment(config.name)
    experiment = EXP_CLASS(config, test_run=args.test_run)
    experiment.run()


if __name__ == "__main__":
    main()

