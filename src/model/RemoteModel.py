""" 
Interface to OpenAI and HuggingFace models.
https://platform.openai.com/docs/api-reference/chat

"""


import os
from time import sleep 
from warnings import warn
from anthropic import Anthropic
from openai import OpenAI
from huggingface_hub import InferenceClient

class RemoteModel():

    def __init__(self, config, seed=42) -> None:
        self.config = config 
        self.seed = seed
        self.name = self.config.name 
        self.max_retries, self.expo_factor = 15, 2
        self.batch_size = 1

        if self.config.source == "huggingface":
            self.client = InferenceClient().chat.completions
        elif self.config.source == "anthropic":
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            self.client = Anthropic().messages
        elif self.config.source == "google":
            self.client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=os.environ.get("GOOGLE_API_KEY")
                # api_key defaults to os.environ.get("OPENAI_API_KEY")
                #max_retries=self.max_retries,
                #timeout=300.0, # 5 minutes timeout
            ).chat.completions
        else:
            self.client = OpenAI(
                # api_key defaults to os.environ.get("OPENAI_API_KEY")
                max_retries=self.max_retries,
                timeout=300.0, # 5 minutes timeout
            ).chat.completions

    
    def batch_query(self, batch, gen_kwargs):
        return [self.query(m, gen_kwargs) for m in batch]
    
    def query(self, messages, gen_kwargs):            
        if type(messages[0]) == list:
            msg = """
            You passed in argument as multiple list of messages 
            but not suported yet, only generating for one"
            """
            raise ValueError(msg)
                    
        if ("gemma" in self.config.name.lower() or self.config.source == "anthropic") and (messages[0]["role"] == "system"):
            messages = messages[1:]
            
        tries = 1
        while tries <= self.max_retries:
            rejected = ["num_beams"]
            if self.config.source == "anthropic":
                rejected.extend(["response_format", "seed", "n"])
                if gen_kwargs["top_p"] is None:
                    gen_kwargs["top_p"] = 1.0
                    
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if k not in rejected}

            try:
                completions = self.client.create(
                    model=self.config.name,
                    messages=messages,
                    **gen_kwargs,
                )

                if self.config.source == "anthropic":
                    r = completions.content[0].text
                else:
                    r = completions.choices[0].message.content

                return r 
            
            except Exception as e:

                if "PRO" in str(e):
                    sleep_time = 60 * 10
                    m = f"""An error occured while generating: {e}.
                    Retrying generation in half an hour to wait for pro subscription
                    to come back 
                    """
                elif "500 Server Error:" in str(e):
                    sleep_time = 0
                    m = f"""An error occured while generating: {e}.
                    Retrying generation in 2 seconds to see if it works then
                    """
                    tries -= 1
                else:
                    sleep_time = (20 * self.expo_factor) #* tries
                    m = f"""An error occured while generating: {e}.
                    Retrying generation in {sleep_time / 60} minute
                    {self.max_retries - tries} lefts
                    """
                warn(m)
                sleep(sleep_time)
                tries += 1
        
        warn(f"Could not generate, giving up")
        return ""
