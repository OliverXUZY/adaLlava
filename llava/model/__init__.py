import os

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",    
    # Add other models as needed
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        print(f"Failed to import {model_name} from llava.language_model.{model_name}. Error: {e}")
from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
from .language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenConfig
from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
from .language_model.llava_mixtral import LlavaMixtralForCausalLM, LlavaMixtralConfig
from .language_model.llava_gemma import LlavaGemmaForCausalLM, LlavaGemmaConfig
from .language_model.llava_qwen_moe import LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig
from .adaptive_model.adaptive_llava_qwen import AdaptiveLlavaQwenForCausalLM