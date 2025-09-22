import os 
import re
from warnings import warn
import dspy
import pandas as pd 
from src.Experiment import Experiment
from src.model.HuggingFaceLocalModel import HuggingFaceLocalModel
from src.model.HugLM import HugLM

from tqdm import tqdm
from src.trl.TRL import TRL
from rapidfuzz import fuzz


class Generate(Experiment):
    """
    A generic experiment runner for generating outputs using any DSPy module.

    Handles both batch and sequential generation, and aligns outputs with
    internal LM history using fuzzy matching to extract reasoning, feedback, etc.
    """

    def __init__(self, config, test_run, module_cls, can_batch=False):
        """
        Initialize the generation experiment.

        Args:
            config (Config): Experiment configuration.
            test_run (bool): Whether this is a test run.
            module_cls (class): A subclass of DSPy Module to use for generation.
        """
        super().__init__(config, test_run)
        self.module_cls = module_cls
        self.can_batch = can_batch


    def run(self):
        """
        Main execution method.

        - Loads the input dataframe.
        - Loads the LM and DSPy module.
        - Runs generation (batch or single depending on model type).
        - Aligns and merges results into a final dataframe.
        - Saves to disk and prints summary stats.
        """
        dataframe = self.load_dataframe()
        module = self.module_cls()
        self.lm = self.load_model()

        generate = self._generate
        if self.config.model.source == "openai" and self.can_batch:
            # Currently, this can cause issue and fallback is hard, so better use it only for judging 
            generate = self._batch_generate
        
        dspy_dataset = self.module_cls.build_dspy_dataset(dataframe)
        df = generate(dspy_dataset, module)
        columns = [c for c in dataframe.columns if c not in df.columns]
        dataframe = dataframe[columns]
        dataframe = dataframe.join(df, how="left").dropna(axis=1, how="all")
        dataframe.to_csv(self.results_save_path)
    
        print("Created dataframe", dataframe, dataframe.columns)

        if "cost" in self.lm.history[-1]:
            print("Total cost of generations", sum(h["cost"] for h in self.lm.history if h["cost"]))


    def _batch_generate(self, dataset, module):
        """
        Generate predictions in batch and align them with language model history.

        This function:
        - Runs batch generation using the given DSPy module.
        - Matches each prediction to its corresponding entry in `self.lm.history` 
          using fuzzy matching on prediction fields (e.g., reasoning, feedback).
        - Collects both model outputs and LM metadata.
        - Returns a DataFrame aligned with the original input dataset.

        Args:
            dataset (List[dspy.Example]): Input dataset of examples.
            module (dspy.Module): DSPy module used to generate predictions.

        Returns:
            pd.DataFrame: A DataFrame indexed by example index, containing:
                - Prediction outputs (e.g., reasoning, feedback, grading)
                - LM metadata (e.g., messages, model type, cost)
        """
        predictions = module.batch(dataset, num_threads=4, max_errors=1)

        if len(predictions) != len(dataset):
            warn("Processing dataset failed for one or more examples")

        lm_history_df = pd.DataFrame(self.lm.history)
        mapping = self.config.task.outputs.toDict()
        matched_indices = match_predictions_to_history(predictions, lm_history_df, threshold=50)

        matched_data = []
        for i, (pred, idx) in enumerate(zip(predictions, matched_indices)):
            outputs = {v: getattr(pred, k) for k, v in mapping.items()}
            matched_data.append({
                "index": i,
                "hist_index": idx,
                **outputs
            })

        df = pd.DataFrame(matched_data)
        return df.set_index("index")


    def _generate(self, dataset, module):
        """
        Generate predictions sequentially and align them with language model history.

        This function:
        - Runs generation one example at a time using the given DSPy module.
        - Relies on the fact that `self.lm.history[-1]` contains the metadata 
          for the most recent generation.
        - Collects prediction outputs and LM metadata in order.
        - Returns a DataFrame aligned with the input dataset.

        Args:
            dataset (List[dspy.Example]): Input dataset of examples.
            module (dspy.Module): DSPy module used to generate predictions.

        Returns:
            pd.DataFrame: A DataFrame indexed by example index, containing:
                - Prediction outputs (e.g., reasoning, feedback, grading)
                - LM metadata (e.g., messages, model type, cost)
        """
        output_dataframe = []
        mapping = self.config.task.outputs.toDict()

        for i, x in enumerate(tqdm(dataset)):
            try:
                pred = module(**x.inputs())
                outputs = {v: getattr(pred, k) for k, v in mapping.items()}
                outputs.update({"index": i, **self.lm.history[-1]})
                output_dataframe.append(outputs)
            except Exception:
                warn(f"Generation failed for example {x}")
                continue

        return pd.DataFrame(output_dataframe).set_index("index")


    def load_model(self):
        """
        Load the LM backend for DSPy, either OpenAI or HuggingFace local.

        Returns:
            dspy.LM: a DSPy-compatible language model interface
        """
        if self.config.model.source == "openai":
            lm = dspy.LM(f'{self.config.model.source}/{self.config.model.name}', 
                        api_key=os.environ["OPENAI_API_KEY"], 
                        temperature=0.0, top_p=1.0, max_tokens=4096, stop=None, cache=False)
        else:
            local_instance = load_model_agent(self.config)
            lm = HugLM(local_instance,
                       temperature=0.0, top_p=1.0, max_tokens=4096, stop=None, cache=False)

        dspy.configure(lm=lm)
        return lm 



def load_model_agent(config):
    """
    Load a locally trained model from disk (base + optional adapters).

    Args:
        config: full experiment config containing model path.

    Returns:
        HuggingFaceLocalModel: loaded model instance
    """
    agent_config = config.model
    if "model" in config.model:
        experiment = TRL(config.model, test_run=False)
        agent_config = experiment.config.model
        agent_config.name = experiment.save_dir
        if config.task.model_args:
            agent_config.update(config.task.model_args)
        
    return HuggingFaceLocalModel(agent_config)



def extract_fields(output: str) -> dict:
    """
    Extracts delimited sections from an output string into a dict:
    {field_name: field_value}
    Delimiters must be in the format: [[ ## FIELD_NAME ## ]]

    Args:
        output (str): LM-generated output string

    Returns:
        dict: mapping from field name (lowercase) to field content
    """
    pattern = re.compile(r"\[\[\s*##\s*(.*?)\s*##\s*\]\](.*?)(?=\[\[\s*##|$)", re.DOTALL)
    return {
        field.strip().lower(): content.strip()
        for field, content in pattern.findall(output)
    }


def match_predictions_to_history(predictions, df, threshold=70, fields_to_match=None):
    """
    Matches a list of prediction objects to rows in a history dataframe based on fuzzy matching.

    Enforces 1:1 matching between predictions and rows (no reuse of history rows).

    Args:
        predictions (List[Prediction]): List of prediction objects with attributes.
        df (pd.DataFrame): DataFrame with 'outputs' column containing lists of strings.
        threshold (int): Fuzzy match threshold for average score across fields.
        fields_to_match (List[str] or None): Specific fields to compare (default: all common fields).

    Returns:
        List[int]: List of matched DataFrame indices in the same order as `predictions`.
    """
    # Preprocess history: extract fields once
    history_data = []
    for i, row in df.iterrows():
        try:
            output = row["outputs"][0]
            if isinstance(output, str):
                extracted = extract_fields(output)
                history_data.append((i, extracted))
        except Exception:
            continue

    used_indices = set()
    matched_indices = []

    for pred in predictions:
        best_score = -1
        best_index = None

        # Determine which fields to match on
        fields = fields_to_match or [
            field for field in history_data[0][1].keys() if hasattr(pred, field)
        ]

        for i, extracted in history_data:
            if i in used_indices:
                continue

            scores = []
            for field in fields:
                pred_val = getattr(pred, field, None)
                extracted_val = extracted.get(field)
                if pred_val and extracted_val:
                    score = fuzz.token_set_ratio(str(pred_val), str(extracted_val))
                    scores.append(score)

            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score >= threshold and avg_score > best_score:
                    best_score = avg_score
                    best_index = i

                # early exit if perfect match
                if avg_score == 100:
                    best_index = i
                    break

        if best_index is None:
            print("Fields", fields)
            raise ValueError("No matching row found for prediction", pred)

        used_indices.add(best_index)
        matched_indices.append(best_index)

    return matched_indices
