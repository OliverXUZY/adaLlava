import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import transformers


from llava.train.train import make_supervised_data_module, LazySupervisedDataset, DataCollatorForSupervisedDataset
from llava import conversation as conversation_lib


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
from pdb import set_trace as pds
from pprint import pprint
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import time
import sys

import warnings

warnings.filterwarnings("ignore")


class Timer(object):

    def __init__(self):

        self.start()

    def start(self):
        self.v = time.time()

    def end(self):
        return time.time() - self.v


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t > 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


device = 'cuda'

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path, indent = 4):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent = indent)

def ensure_path(path, early_exit = False):
    if os.path.exists(path):
        if early_exit:
            if input('{:s} exists, continue? ([y]/n): '.format(path)) == 'n':
                sys.exit(0)
    else:
        os.makedirs(path)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

def main(args):
    timer = Timer()
    # Create the DataArguments object
    data_args = DataArguments(
        data_path=args.question_file,
        lazy_preprocess=True,
        is_multimodal=True,
        image_folder=args.image_folder,
        image_aspect_ratio='pad'
    )

    print(f"read from file: {args.question_file}")
    print("image_folder: ", args.image_folder)
    


    # Model
    # disable_torch_init()
    model_name_or_path = os.path.expanduser(args.model_path)
    # model_name = "llava_qwen"
    model_name = "llava_qwen_adaptive"
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(model_name_or_path, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode]
    # conversation_lib.default_conversation = conversation_lib.conv_templates["mistral_instruct"]
    

    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                data_path=data_args.data_path,
                data_args=data_args)
    
    # eg = eval_dataset[0]
    
    # eg1 = eval_dataset[1]
    # eg2 = eval_dataset[2]
    # eg24 = eval_dataset[24]
    # eg25 = eval_dataset[26]
    # eg26 = eval_dataset[26]
    # pds()


    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    dataloader_params = {
            "batch_size": 1,
            "collate_fn": data_collator,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": False,
        }
    
    eval_loader = DataLoader(eval_dataset, **dataloader_params)

    # batch = next(iter(eval_loader))
    # # input_ids = batch['input_ids'][0]
    # # labels = batch['labels'][0]
    # # image = batch['images'][0]
    # # attention_mask = batch['attention_mask'][0]
    # # pds()

    # batch = {k: (v.to(device).half() if v.dtype in [torch.float32, torch.float64] else v.to(device))
    #         if torch.is_tensor(v) else v 
    #         for k, v in batch.items()}
    # image_tensor = batch['images']
    # batch['images'] = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
   
    # out = model(**batch)
    # loss = out['loss']
    # pds()

    losses = []

    # drop_mask = torch.randint(0, 2, (24, 1)).long().to(device)
    # drop_mask[:] = 1
    # pds()

    branch_idx = args.branch_idx
    all_masks = np.load(args.mask_array)
    drop_mask = torch.from_numpy(all_masks[branch_idx]).long().to(device)
    

    for idx, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Evaluating"):
        # if idx > 10:
        #     break
        batch = {k: (v.to(device).half() if v.dtype in [torch.float32, torch.float64] else v.to(device))
            if torch.is_tensor(v) else v 
            for k, v in batch.items()}
        image_tensor = batch['images']
        batch['images'] = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        batch['drop_mask'] = drop_mask
        out = model(**batch)
        loss = out.loss  # Assuming the loss is stored in out.loss

        # Append the loss value to the list
        losses.append(loss.item())  # .item() extracts the scalar value from the tensor

    print(f"forward done, using time {time_str(timer.end())}")

    # After the loop, convert the list to a numpy array
    losses_array = np.array(losses, dtype=np.float32)

    # Now losses_array is a numpy array of float32 containing all your loss values
    print(f"Shape of losses array: {losses_array.shape}")
    print(f"Data type of losses array: {losses_array.dtype}")

    save_path = args.save_path
    ensure_path(save_path)

    # Assuming losses_array is your numpy array of shape (336825,) and dtype float32
    np.save(f'{save_path}/branch{branch_idx}_losses.npy', losses_array)
    print(f"save to {save_path}/subset_branch{branch_idx}_losses.npy, using time {time_str(timer.end())}")




    

def parge_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-onevision-qwen2-0.5b-si")
    parser.add_argument("--image-folder", type=str, default="./data/MME/images")
    parser.add_argument("--question-file", type=str, default="data/MME/json_qa/subset_qa_MME_choice.json")
    parser.add_argument("--answers-file", type=str, default="answers/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")

    parser.add_argument("--mask-array", type=str, default="./mask_variations_5.npy")
    parser.add_argument("--save-path", type=str, default="data/MME/ada_losses/subset/mask_5")
    parser.add_argument("--branch-idx", type=int, default=0)
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parge_args()
    # eval_model(args)
    main(args)