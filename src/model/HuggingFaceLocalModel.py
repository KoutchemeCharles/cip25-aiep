""" Wrapper class around common HuggingFace model loading and inference functionalities """

import os 
import torch 
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, pipeline
)
from copy import deepcopy
from peft import AutoPeftModelForCausalLM
from trl import get_kbit_device_map

class HuggingFaceLocalModel():
    
    def __init__(self, config, is_training=False) -> None:
        """
        Initialize the HuggingFaceLocalModel model.

        Args:
            config (dict): The configuration of the model.
            is_training (bool, optional): Whether to use the model for training or not. Defaults to False.
        """
        print("Huggingface model", config)
        self.config = config 
        self.is_training = is_training
        self.accelerator = Accelerator()  # ← Add this

        self.device = get_current_device(self.accelerator)
        self.supports_flash_attention = supports_flash_attention()
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()           
        self.pipe = self.load_pipeline() if not self.is_training else None 

    def batch_query(self, batch, gen_kwargs):
        """
        Batch query the model for a list of messages.

        Args:
            batch (list of dict): A list of messages in the format of 
                [{"role": "user", "content": "message"}, ...].
            gen_kwargs (dict): Generation arguments.

        Returns:
            list of str: The generated responses.
        """

        tokenizer = self.pipe.tokenizer
        new_kwargs = adapt_gen_kwargs(deepcopy(gen_kwargs))
        agp = batch[-1][-1]["role"] == "user"

        # For trained models, this pipeline is more efficeint
        inputs = tokenizer.apply_chat_template(batch, tokenize=False, 
                                               add_generation_prompt=agp,
                                               continue_final_message=not agp,
                                               pad_to_multiple_of=8)
        responses = self.pipe(inputs, return_full_text=False, **new_kwargs)
        
        if True: #and not self.needs_custom_pipeline: # Because using personal cache 
            responses = [resp[j]['generated_text'] 
                        for resp in responses 
                        for j in range(len(resp))]
        
        return responses
        

    def query(self, messages, gen_kwargs):
        return self.batch_query([messages], gen_kwargs)
    

    def load_model(self):
        """
        Load a model from the config parameters.

        The model is loaded using the transformers library AutoModelForCausalLM
        or AutoPeftModelForCausalLM. The model is loaded
        with the specified dtype and device map.

        If the model is a quantized model, it is loaded with the specified
        quantization configuration.

        If the model is an adapter model, it is loaded with the specified
        adapter configuration.

        The model is then converted to the specified dtype and device map.

        Returns:
            The loaded model.
        """

        if self.config.dtype == "fp32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16

        if self.supports_flash_attention:
            attn_implementation = "flash_attention_2"
            if self.is_training:
                torch_dtype = torch.bfloat16
        else:
            attn_implementation = "eager"

        other_args = {}
        if self.config.quant == 4:
            print("Loading model in 4bit")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.quant == 8:
            print("Loading model in 8bit precision")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None 
        
        if bnb_config:  other_args["quantization_config"] = bnb_config 

        
        device_map = "auto"
        has_adapters = has_saved_adapters(self.config.name)
        if self.config.device_map: 
            device_map = self.config.device_map
        elif self.is_training:
            device_map = get_kbit_device_map() #if bnb_config is not None else "auto"

        ## Note: AutoModelForCausalLM can technically directly load
        ## the adapters. However, it does not have the merge_and_unload()
        ## functionality which is useful for speeding up inference

        if has_adapters: 
            print("Loading saved adapters at", self.config.name)
            model = AutoPeftModelForCausalLM.from_pretrained(
                self.config.name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
                is_trainable=self.is_training,
                low_cpu_mem_usage=not self.is_training,
                **other_args
            )

            if self.is_training:
                model.print_trainable_parameters()
            else:
                if bnb_config is None: # Avoiding at the cost of lower training for new quantization tupes
                    print("Merging model with adapters for faster inference")
                    model = model.merge_and_unload()
        else:
            print("Loading model from normal source")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.name,
                torch_dtype=torch_dtype,
                device_map = device_map, 
                attn_implementation=attn_implementation,
                low_cpu_mem_usage=True,
                **other_args
            )

        # Bellow "padding_right" issue might arise, especially with Phi models 
        # https://github.com/huggingface/trl/issues/1217#issuecomment-1889282654
        if self.is_training:
            model.config.use_cache = False
        else:
            model.eval()

        print("-----")
        print("MODEL", model)
        print("MODEL dtype", torch_dtype)
        print("MODEL USING ATTENTION", attn_implementation)
        print("MODEL device map", model.hf_device_map)
        print("MODEL memory footprint", model.get_memory_footprint())
        print("-----")


        return model 


    def load_tokenizer(self):
        """
        Load the tokenizer from the pretrained model specified in the configuration.

        The tokenizer is configured to ensure compatibility with the model's 
        requirements. If the pad token is not set, it defaults to the end-of-sequence 
        token. The padding and truncation are set to 'left' to handle specific 
        generation conditions.

        Returns:
            The configured tokenizer instance.
        """

        tokenizer = AutoTokenizer.from_pretrained(self.config.name)
        if tokenizer.pad_token == None: 
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left' # to prevent errors with FA
        tokenizer.truncation_side = 'left' # to prevent cutting off last generation

        return tokenizer 
    

    def load_pipeline(self):
        return pipeline("text-generation", 
                        model=self.model, 
                        tokenizer=self.tokenizer)
    
def get_current_device(accelerator):
    return accelerator.local_process_index #.device 

def adapt_gen_kwargs(gen_kwargs):

    gen_kwargs.pop("seed", None)
    gen_kwargs.pop("response_format", None)

    if "n" in gen_kwargs and "num_return_sequences" not in gen_kwargs:
        gen_kwargs["num_return_sequences"] = gen_kwargs.pop("n")

    if "max_tokens" in gen_kwargs:
        gen_kwargs["max_new_tokens"] = gen_kwargs.pop("max_tokens")

    gen_kwargs["do_sample"] = True 
    if (gen_kwargs["top_p"] == None or gen_kwargs["top_p"] == 1.0) and (gen_kwargs["temperature"] == 0.0):
        gen_kwargs["top_p"] = None
        gen_kwargs["temperature"] = None
        gen_kwargs["top_k"] = None
        gen_kwargs["do_sample"] = False

    return gen_kwargs

def has_saved_adapters(path):
    return os.path.isdir(path) and "adapter_config.json" in os.listdir(path)

def supports_flash_attention():
    """Check if a GPU supports FlashAttention."""

    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    major, minor = torch.cuda.get_device_capability(DEVICE)
    
    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0
    
    return is_sm8x or is_sm90



def compile_for_inference(model):

    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
    except Exception:
        pass 

    return model 

