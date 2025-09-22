"""

https://huggingface.co/docs/trl/main/en/sft_trainer

"""
import torch 
import pandas as pd
import numpy as np
from src.trl.TRL import TRL
from trl import SFTConfig, SFTTrainer
from datasets import Dataset, DatasetDict

class SFT(TRL):

    def __init__(self, config, test_run) -> None:
        super().__init__(config, test_run)
        

    def prepare_dataset(self):
        df = self.load_dataframe()
        
        df["messages"] = list(map(eval, df["messages"]))
        df["prompt"] = df["messages"] # messages 
        f = lambda r: [{"role": "assistant", "content": eval(r)[0]}]
        df["completion"] = list(map(f, df["outputs"]))
        
        grading = list(map(eval, df["teacher_grading"]))
        df["is_correct"] = [sum([v == 0 for v in d.values()]) == len(d) for d in grading]
        print(df.groupby(["diag_exercise", "is_correct"]).student_id.count())

        sampling = self.config.task.sampling.toDict()
        train_df, val_df = zipf_sample_balanced(df, **sampling)

        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(val_df)
        })

        return dataset

    def train(self, dataset, train_args, peft_config, **other_trainer_args):

        model = self.agent.model

        args = SFTConfig(**train_args)

        print("Dataset", dataset)
        print("Arguments", args)
        print("Peft config", peft_config)
        print("Model", model)

        trainer = SFTTrainer(
            model=model,
            peft_config=peft_config,
            processing_class=self.agent.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=args,
            **other_trainer_args
        )

        trainer.train()
        trainer.save_model()

        del self.agent.model
        del trainer
        torch.cuda.empty_cache()


def sample_group(df, train_size, random_state):
    """
    Sample a DataFrame using normalized_cluster_frequency as probabilities.
    """
    np.random.seed(random_state)

    # Normalize the cluster frequencies to sum to 1 within the group
    df = df.copy()
    df['normalized_prob'] = df['normalized_cluster_frequency'] / df['normalized_cluster_frequency'].sum()

    # Sample indices based on the normalized probabilities
    train_indices = np.random.choice(df.index, size=train_size, replace=False, p=df['normalized_prob'])

    train_sample = df.loc[train_indices]

    # Validation sample is the remaining data
    val_sample = df.drop(train_indices)

    return train_sample, val_sample

def zipf_sample_balanced(df, train_frac=0.8, head_frac=None, random_state=42):
    """
    Balanced sampling using normalized_cluster_frequency:
    - For each exercise, and within each correctness group (correct/incorrect), apply sampling based on normalized_cluster_frequency.
    - Returns: training_df, validation_df
    """
    train_rows = []
    val_rows = []

    for ex_id, group in df.groupby("diag_exercise"):
        # Separate correct and incorrect submissions
        correct_df = group[group["is_correct"] == True]
        incorrect_df = group[group["is_correct"] == False]

        # Sample from correct submissions
        train_size_correct = int(train_frac * len(correct_df))
        correct_train, correct_val = sample_group(correct_df, train_size_correct, random_state)

        # Sample from incorrect submissions
        train_size_incorrect = int(train_frac * len(incorrect_df))
        incorrect_train, incorrect_val = sample_group(incorrect_df, train_size_incorrect, random_state)

        # Combine samples
        train_rows.append(correct_train)
        train_rows.append(incorrect_train)
        val_rows.append(correct_val)
        val_rows.append(incorrect_val)

    train_full = pd.concat(train_rows).reset_index(drop=True)
    val_full = pd.concat(val_rows).reset_index(drop=True)

    print(f"Training dataset:", train_full)
    print(f"Validation dataset", val_full)

    print(train_full.groupby(["diag_exercise", "is_correct"]).student_id.count())
    print(val_full.groupby(["diag_exercise", "is_correct"]).student_id.count())

    return train_full[["prompt", "completion"]], val_full[["prompt", "completion"]]
