"""
Custom class for inference on a local huggingface model
using DSPY.

https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/base_lm.py
"""

import uuid
import time 
from dspy import BaseLM
from src.model.HuggingFaceLocalModel import adapt_gen_kwargs
from dotmap import DotMap 

class HugLM(BaseLM):

    def __init__(self, local_instance, model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs):
        self.local_instance = local_instance
        self.config = self.local_instance.config
        super().__init__(self.config.name, model_type, temperature, max_tokens, cache, **kwargs)

    def forward(self, prompt, messages=None, **kwargs):
        """
        Runs inference using the HuggingFaceLocalModel and returns a fully OpenAI-style response.
        """
        # Build messages if not present
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # Build generation kwargs, as before
        gen_kwargs = {**self.kwargs}
        gen_kwargs.update(kwargs)
        gen_kwargs = adapt_gen_kwargs(gen_kwargs)

        # Generate response(s)
        generations = self.local_instance.query(messages, gen_kwargs)
        if not isinstance(generations, list):
            generations = [generations]

        # Simulate token usage (optional: use tokenizer to compute exact if you wish)
        prompt_tokens = sum(len(self.local_instance.tokenizer.encode(msg['content'])) for msg in messages)
        completion_tokens_list = [len(self.local_instance.tokenizer.encode(gen)) for gen in generations]
        total_tokens_list = [prompt_tokens + ct for ct in completion_tokens_list]

        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.name,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": gen
                    },
                    "finish_reason": "stop"  # Could be 'length' if max_tokens reached, etc.
                }
                for i, gen in enumerate(generations)
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens_list[0] if completion_tokens_list else 0,
                "total_tokens": total_tokens_list[0] if total_tokens_list else 0,
            }
        }
        
        return DotMap(response)
