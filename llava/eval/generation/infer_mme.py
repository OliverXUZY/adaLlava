import os
from tqdm import tqdm
import shortuuid
import argparse
import sys
import json


# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import warnings

warnings.filterwarnings("ignore")

from pdb import set_trace as pds


# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

def eval_model(args):
    model_name_or_path = args.model_path

    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(model_name_or_path, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

    model.eval()

    questions = load_json(os.path.expanduser(args.question_file))
    answers_file = os.path.expanduser(args.answers_file)

    conv_template = args.conv_mode  # Make sure you use correct chat template for different models
    if args.conv_mode is not None:
        answers_file = f"{answers_file[:-6]}_{args.conv_mode}.jsonl"

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["id"]
        qs = line["conversations"][0]["value"]

        images = []
        if 'image' in line:
            image_files = line["image"]
            if type(image_files) is not list:
                assert type(image_files) is str, "image path should should be a str"
                image_files = [image_files]
            for image_file in image_files:
                image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
                images.append(image)
            image_tensor = process_images(images, image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        else:
            raise NotImplementedError


        ### to reproduce lmm setting
        qs = qs.replace(" Please answer yes or no.","\nAnswer the question using a single word or phrase.")


        question = qs
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]
        # pds()


        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "model_input": prompt_question,
                                   "text": text_outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")



def parge_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-onevision-qwen2-0.5b-si")
    parser.add_argument("--image-folder", type=str, default="./data/MME/images")
    parser.add_argument("--question-file", type=str, default="data/MME/json_qa/qa_MME.json")
    parser.add_argument("--answers-file", type=str, default="answers/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parge_args()
    eval_model(args)