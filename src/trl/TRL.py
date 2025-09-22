import os
from peft import LoraConfig
from src.Experiment import Experiment
from src.utils.files import create_dir
from src.model.HuggingFaceLocalModel import HuggingFaceLocalModel


class TRL(Experiment):

    def __init__(self, config, test_run) -> None:

        super().__init__(config, test_run)
        self.dataset_save_path = os.path.join(self.save_dir, "dataset")
        create_dir(self.dataset_save_path)
        self.other_args = {}

    def run(self):

        self.agent = load_model_agent(self.config)

        dataset = self.prepare_dataset()
        if self.dataset_save_path: 
            dataset.save_to_disk(self.dataset_save_path)

        train_args = self.prepare_training()
        peft_config = self.prepare_peft_config(train_args)

        self.train(dataset, train_args, peft_config)


    def prepare_training(self):
        """
        Prepare a dictionary of training arguments for the Trainer.

        This is a simplified version of the arguments, which will be
        updated with the experiment specific arguments.

        Args:
            None

        Returns:
            base_training_args: A dictionary of training arguments
        """

        base_training_args = {}
        base_training_args["output_dir"] = self.save_dir
        base_training_args["overwrite_output_dir"] = True
        ## Efficient training
        base_training_args["fp16"] = True
        base_training_args["bf16"] = False
        base_training_args["gradient_accumulation_steps"] = 8
        base_training_args["per_device_train_batch_size"] = 1
        base_training_args["per_device_eval_batch_size"] = 1
        base_training_args["gradient_checkpointing"] = True
        ## Base training arguments
        base_training_args["num_train_epochs"] = 3
        base_training_args["lr_scheduler_type"] = "cosine"
        base_training_args["max_grad_norm"] = 1.0
        base_training_args["warmup_ratio"] = 0.1
        base_training_args["eval_strategy"] = "steps"
        base_training_args["eval_steps"] = 0.1
        base_training_args["save_strategy"] = "steps"
        base_training_args["save_steps"] = 0.1
        base_training_args["logging_strategy"] = "steps"
        base_training_args["logging_steps"] = 5 
        base_training_args["save_total_limit"] = 2
        base_training_args["load_best_model_at_end"] = True
        base_training_args["max_length"] = 16384
        base_training_args["label_names"] = ["labels"]
        ## Bonus 
        base_training_args["report_to"] = "wandb" #"none" 
 
        if self.test_run:
            base_training_args["max_steps"] = 10
            base_training_args["eval_strategy"] = "no" 
            base_training_args["load_best_model_at_end"] = False  

        if self.other_args:
            base_training_args.update(**self.other_args)
       
        if self.config.task.args:
            base_training_args.update(**self.config.task.args.toDict())

        return base_training_args


    def prepare_peft_config(self, base_training_args):
        # We are training LORA adapter so we can
        # load the model in any precision we want
        # by default, we loaded the model in fp16
        peft_config = None 
        if self.config.task.lora:
            lora_config = {
                "r": 8, 
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                # "target_modules": "all-linear",
                "task_type": "CAUSAL_LM",
            }
            lora_config.update(self.config.task.lora)
            peft_config = LoraConfig(**lora_config) 
            
        elif self.agent.supports_flash_attention:
            base_training_args["fp16"] = False 
            base_training_args["bf16"] = True 
            base_training_args["tf32"] = True
            base_training_args["bf16_full_eval"] = True 

        return peft_config 
    


def load_model_agent(config):

    # We loaded a prior experiment, meaning that we must have trained a model
    # this model might be saved either fully or with adapters
    # The location/path to the model to load is the TRL save directory 
    agent_config = config.model
    if "model" in config.model:
        experiment = TRL(config.model, test_run=False)
        agent_config = experiment.config.model
        agent_config.name = experiment.save_dir
        if config.task.model_args:
            agent_config.update(config.task.model_args)
        
    return HuggingFaceLocalModel(agent_config, is_training=True)
