import torch
from peft import PeftModel
from trl import DPOConfig, DPOTrainer
from datasets import Dataset, DatasetDict 
from warnings import warn 
from src.trl.TRL import TRL
from src.trl.KTO import (
    add_metadata, 
    format_prompt_completion, 
    stratified_train_val_split_zipf
)

class DPO(TRL):

    def __init__(self, config, test_run) -> None:
        super().__init__(config, test_run)
        self.other_args = {
            "max_length": 4096,
            "max_prompt_length": 2048,
            "max_completion_length": 2048
        }

    def prepare_dataset(self):
        df = self.load_dataframe()
        df = add_metadata(df)
        df = format_prompt_completion(df)
        df = create_preference_pairs(df)
        
        train_df, val_df = stratified_train_val_split_zipf(df, **self.config.task.sampling)
        columns = ["prompt", "chosen", "rejected"]
        train_df, val_df = train_df[columns], val_df[columns]

        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "test": Dataset.from_pandas(val_df)
        })

        return dataset 

        
    def train(self, dataset, train_args, peft_config):

        model = self.agent.model
        # https://huggingface.co/docs/trl/main/en/dpo_trainer#using-option-3---load-the-adapter-twice
        # if the model we loaded already has an adapter
        # we want to use a reference model which has this adapter
        # not the raw base model. We could have merged the adapter
        # but this makes it impossible to retrieve the "fully" trained
        # adapter later. Currently, this means that a model that was trained
        # with adapters will continue being trained with adapters as well
        
        ref_adapter_name, merge_full_model = None, False
        if isinstance(model, PeftModel) and peft_config:
            m = """
            Model loaded already has an adapter, 
            but we passed in a PeftConfig, so gonna merge and unload.
            Be careful of the consequences. We now need to save the full
            model to recover the entire performance 
            """
            warn(m)
            model = model.merge_and_unload()
            merge_full_model = True 

        elif isinstance(model, PeftModel):
            m = """
            Model loaded already has an adapter, 
            we are continuing training with a frozen reference model
            """
            warn(m)
            ref_adapter_name = "reference"
            model.load_adapter(self.agent.config.name, 
                               adapter_name=ref_adapter_name, 
                               is_trainable=False)
            train_args["ref_adapter_name"] = ref_adapter_name
            model.set_adapter("default")
        

        args = DPOConfig(**train_args)

        trainer = DPOTrainer(
            model=model,
            peft_config=peft_config,
            processing_class=self.agent.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=args,
        )

        print("Dataset", dataset)
        print("Arguments", args)
        print("peft config", peft_config)
        
        trainer.train()

        if not merge_full_model:
            trainer.save_model()

        elif self.agent.accelerator.is_main_process:
            # https://github.com/huggingface/peft/issues/1381
            # https://github.com/huggingface/peft/issues/636#issuecomment-1607852012
            # https://github.com/huggingface/peft/issues/636#issuecomment-2082473781
            model = trainer.model
            model = model.merge_and_unload()
            model.base_model.save_pretrained(args.output_dir)
            self.agent.tokenizer.save_pretrained(args.output_dir)
            print("Merged model", model)

        del self.agent.model
        del trainer
        torch.cuda.empty_cache()


def create_preference_pairs(df):

    student_mask = (~df.feedback.isna())
    student_df = df[student_mask].reset_index(drop=True)
    student_df = student_df.set_index(["student_id", "diag_exercise"])
    student_df["rejected"] = student_df["completion"]

    teacher_df = df[~student_mask].reset_index(drop=True)
    teacher_df["feedback"] = teacher_df.teacher_feedback
    teacher_df = teacher_df.set_index(["student_id", "diag_exercise"])
    teacher_df["chosen"] = teacher_df["completion"]

    # Merge the two, using teacher as good, student as bad 
    df = teacher_df.join(student_df[["rejected"]])
    df = df.sample(frac=1).reset_index(drop=False) # shuffling
    df = df.dropna(subset=["chosen", "rejected"], axis=0) 

    return df
