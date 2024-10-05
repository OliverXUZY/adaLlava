import sys
sys.path.append('./')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
from torch import nn

import sys
import warnings
from pdb import set_trace as pds
warnings.filterwarnings("ignore")
pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-si"
model_name = "llava_qwen_adaptive"
# model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

def format_number(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num}"
    
def count_parameters(model):
    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    llm_params = count_params(model.model.layers) + count_params(model.model.embed_tokens) + count_params(model.model.norm) + count_params(model.lm_head)
    vision_params = count_params(model.model.vision_tower)
    mm_projector_params = count_params(model.model.mm_projector)

    total_params = llm_params + vision_params + mm_projector_params

    print(f"Language Model: {format_number(llm_params)} parameters")
    print(f"Vision Encoder: {format_number(vision_params)} parameters")
    print(f"Multimodal Projector: {format_number(mm_projector_params)} parameters")
    print(f"\nTotal trainable parameters: {format_number(total_params)}")



def main():
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation = "eager")  # Add any other thing you want to pass in llava_model_args
    # pds()
    # model.eval()
    model.train()
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)

    print(count_parameters(model))

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor] # [torch.Size([10, 3, 384, 384])]

    conv_template = "qwen_2"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]



    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(text_outputs)

if __name__ == "__main__":
    main()