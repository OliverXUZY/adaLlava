import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from PIL import Image
import math
from pdb import set_trace as pds
from pprint import pprint
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import time
import sys
import gc

from llava.train.ada_train import make_supervised_data_module, LazySupervisedDataset, DataCollatorForSupervisedDataset
from llava import conversation as conversation_lib


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.adaptive_model.ada_scheduler import ada_Scheduler, ada_SchedulerCfg

def save_grad_status(model, output_dir = "save"):
    """
    Save parameter names based on their requires_grad status to separate files.
    
    Args:
    model: The PyTorch model
    output_dir: Directory to save the output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    grad_true_file = os.path.join(output_dir, "grad_true_params.txt")
    grad_false_file = os.path.join(output_dir, "grad_false_params.txt")
    all_file = os.path.join(output_dir, "all_params.txt")
    
    with open(grad_true_file, 'w') as f_true, open(grad_false_file, 'w') as f_false, open(all_file, 'w') as f_all:
        for name, param in model.named_parameters():
            if param.requires_grad:
                f_true.write(f"{name}\n")
            else:
                f_false.write(f"{name}\n")
            f_all.write(f"{name}\n")
    
    print(f"Parameters with requires_grad=True saved to: {grad_true_file}")
    print(f"Parameters with requires_grad=False saved to: {grad_false_file}")

import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(0)

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

# @dataclass
# class DataArguments:
#     data_path: str = field(default=None,
#                            metadata={"help": "Path to the training data."})
#     lazy_preprocess: bool = False
#     is_multimodal: bool = False
#     image_folder: Optional[str] = field(default=None)
#     image_aspect_ratio: str = 'square'

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)

def main(args):
    timer = Timer()
    # Create the DataArguments object
    data_args = DataArguments(
        data_path=args.question_file,
        lazy_preprocess=True,
        is_multimodal=True,
        image_folder=args.image_folder,
        early_mix_text=False,
        image_aspect_ratio='anyres_max_9',
        image_grid_pinpoints=[[384, 384],
                                [384, 768],
                                [384, 1152],
                                [384, 1536],
                                [384, 1920],
                                [384, 2304],
                                [768, 384],
                                [768, 768],
                                [768, 1152],
                                [768, 1536],
                                [768, 1920],
                                [768, 2304],
                                [1152, 384],
                                [1152, 768],
                                [1152, 1152],
                                [1152, 1536],
                                [1152, 1920],
                                [1152, 2304],
                                [1536, 384],
                                [1536, 768],
                                [1536, 1152],
                                [1536, 1536],
                                [1536, 1920],
                                [1536, 2304],
                                [1920, 384],
                                [1920, 768],
                                [1920, 1152],
                                [1920, 1536],
                                [1920, 1920],
                                [1920, 2304],
                                [2304, 384],
                                [2304, 768],
                                [2304, 1152],
                                [2304, 1536],
                                [2304, 1920],
                                [2304, 2304]],
            image_crop_resolution=None,
            image_split_resolution=None,
            video_folder=None,
            video_fps=1,
            frames_upbound=32,
            add_time_instruction=False,
            force_sample=False,
    )

    print(f"read from file: {args.question_file}")
    print("image_folder: ", args.image_folder)
    


    # Model
    # disable_torch_init()
    model_name_or_path = os.path.expanduser(args.model_path)
    # model_name = "llava_qwen"
    model_name = "llava_qwen_adaptive"
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    device_map = "cuda"

    ### set up scheduler
    ada_schdeuler_cfg = {
            "latency_dim": 64,
            "content_inp_dim": 896,
            "content_dim": 256,
            "n_knobs": 21,
            "combine_type": "concatenate",
        }

    overwrite_config = {"ada_schdeuler_cfg": ada_schdeuler_cfg}

    # ada_schdeuler_cfg = ada_SchedulerCfg()
    # ada_scheduler = ada_Scheduler(ada_schdeuler_cfg)
    # ada_scheduler.to(device)


    tokenizer, model, image_processor, max_length = load_pretrained_model(
                    model_name_or_path, 
                    None, 
                    model_name, 
                    device_map=device_map,
                    overwrite_config = overwrite_config,
                )  # Add any other thing you want to pass in llava_model_args
    
    model.train()

    # Get the lm_head and input embeddings
    # lm_head_weight = model.lm_head.weight
    # input_embeddings_weight = model.model.embed_tokens.weight

    # # Check if the weights are the same object in memory
    # are_tied = (lm_head_weight.data_ptr() == input_embeddings_weight.data_ptr())
    # print(f"Weights are tied: {are_tied}")

    # # Double-check by comparing the actual values
    # # if not are_tied:
    # are_equal = torch.allclose(lm_head_weight, input_embeddings_weight, atol=1e-5)
    # print(f"Weights are equal: {are_equal}")

    # pds()
    # p model.model.layers[0].mlp.up_proj.weight
    # p model.model.ada_scheduler.combined_fc.up_proj.weight
    # p model.lm_head.weight

    #### important: tie lm_head weights otherwise it will be random init !!!!
    model.lm_head.weight = model.model.embed_tokens.weight



    

    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode]
    # conversation_lib.default_conversation = conversation_lib.conv_templates["mistral_instruct"]
    

    eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                data_path=data_args.data_path,
                data_args=data_args)
    
    # eg = eval_dataset[0]
    # pprint(eg)
    # img = eg['image'][0][0]
    # for idx, eg in enumerate(eval_dataset):
    #     if idx > 3:
    #         break
    #     input_ids = eg['input_ids']
    #     print(f"input_ids: {input_ids.shape}")
    #     print(input_ids[-10:])
    #     print("==============================================")


    
    # eg1 = eval_dataset[1]
    # # eg2 = eval_dataset[2]
    # # eg24 = eval_dataset[24]
    # # eg25 = eval_dataset[26]
    # # eg26 = eval_dataset[26]


    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    batch_size = 1
    dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": False,
        }
    
    eval_loader = DataLoader(eval_dataset, **dataloader_params)

    # batch = next(iter(eval_loader))
    # batch = next(iter(eval_loader))
    # # for idx, batch in enumerate(eval_loader):
    # #     if idx > 1:
    # #         break
    # input_ids = batch['input_ids']
    # print(f"input_ids: {input_ids.shape}")
    # labels = batch['labels']
    # print(f"labels: {labels.shape}")
    # attention_mask = batch['attention_mask']
    # print(f"attention_mask: {attention_mask.shape}")
    # image_sizes = batch['image_sizes']
    # print(f"image_sizes: {type(image_sizes)} |{len(image_sizes)}, {image_sizes}")
    # modalities = batch['modalities']
    # print(f"modalities: {type(modalities)} |{len(modalities)}, {modalities}")
    # image = batch['images']
    # print(f"image: {type(image)} |{len(image)}, {image[0].shape}")
    # print([i.shape for i in image])
    # print(input_ids[:,-10:])
    # print("==============================================")
    # img = image[0]
    # print(img.mean())
    # print(img[0].mean())
    # print(img[1].mean())
    # print(img[2].mean())

    # pds()

    # batch = {k: (v.to(device).half() if v.dtype in [torch.float32, torch.float64] else v.to(device))
    #         if torch.is_tensor(v) else v 
    #         for k, v in batch.items()}
    # image_tensor = batch['images']
    # batch['images'] = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    # # # batch['drop_mask'] = torch.ones(24*batch_size).view(24,batch_size).long().to(device)
    # # # batch['drop_mask'] = torch.zeros(24*batch_size).view(24,batch_size).long().to(device)
    # # # batch['drop_mask'] = torch.randint(0, 2, (24, batch_size)).long().to(device)

    # # # batch['latency'] = torch.tensor(0.8).half().to(device)
    # latency = torch.tensor([args.latency]).half().to(device)
    # # batch['latency'] = torch.rand(batch_size).half().to(device)
    # # # print(batch['latency'])
    # # pds()
    # # batch['use_cache'] = False
    # batch['latency'] = latency
   
    # out = model(**batch)
    # loss = out['loss']
    # print(loss)
    # pds()


    # return

    # drop_mask = torch.randint(0, 2, (24, 1)).long().to(device)
    # drop_mask[:] = 1

    # branch_idx = args.branch_idx
    # all_masks = np.load(args.mask_array)
    # drop_mask = torch.from_numpy(all_masks[branch_idx]).long().to(device)
    # pds()
    
    token_losses = []
    macs_losses = []
    flops_all = []
    skipped_samples = []
    if args.latency is not None:
        latency = torch.tensor([args.latency]).half().to(device)  ## only support bs = 1
    else:
        all_latencys = np.load(args.mask_latency)
        latency = torch.tensor(all_latencys[args.latency_idx]).half().to(device).view(batch_size)
    
    for idx, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Evaluating"):
        batch = {k: (v.to(device).half() if v.dtype in [torch.float32, torch.float64] else v.to(device))
            if torch.is_tensor(v) else v 
            for k, v in batch.items()}
        image_tensor = batch['images']
        batch['images'] = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        batch['use_cache'] = False

        # print(batch['images'][0].shape)
        # pds()

        ### drop_mask
        # drop_mask = drop_mask.repeat(1, batch['input_ids'].shape[0])
        # pds()

        ### latency
        # batch['latency'] = torch.rand(batch_size).half().to(device)
        batch['latency'] = latency
        with torch.no_grad():
            try:
                out = model(**batch)
                loss = out.loss  # Assuming the loss is stored in out.loss
                token_loss = out.token_loss
                # print("combine_loss: ", loss)
                # print("token_loss: ", out.token_loss)
                # print("macs_loss: ", out.macs_loss)
                macs_loss = out.macs_loss
                flops = out.flops

                # Append the loss value to the list
                token_losses.append(token_loss.item())  # .item() extracts the scalar value from the tensor
                macs_losses.append(macs_loss.item())  # .item() extracts the scalar value from the tensor
                flops_all.append(flops.item())  # .item() extracts the scalar value from the tensor
            except torch.cuda.OutOfMemoryError:
                print(f"OOM error occurred at index {idx}. Skipping this sample.")
                skipped_samples.append(str(idx))
                token_losses.append(-1)  # .item() extracts the scalar value from the tensor
                macs_losses.append(-1)  # .item() extracts the scalar value from the tensor
                flops_all.append(-1)  # .item() extracts the scalar value from the tensor
                continue

        # Clear unnecessary variables
        del out, loss, token_loss, macs_loss, flops
        torch.cuda.empty_cache()
        gc.collect()


    print(f"forward done, using time {time_str(timer.end())}")

    # After the loop, convert the list to a numpy array
    token_losses_array = np.array(token_losses, dtype=np.float32)
    macs_losses_array = np.array(macs_losses, dtype=np.float32)
    flops_all_array = np.array(flops_all, dtype=np.float32)

    # Now losses_array is a numpy array of float32 containing all your loss values
    print(f"Shape of token_losses array: {token_losses_array.shape}")
    print(f"Data type of token_losses array: {token_losses_array.dtype}")

    save_path = args.save_path
    ensure_path(save_path)

    # Assuming losses_array is your numpy array of shape (336825,) and dtype float32
    np.save(f'{save_path}/token_losses_latency_{args.latency_idx}.npy', token_losses_array)
    print(f"save to {save_path}/token_losses_latency_{args.latency_idx}.npy, using time {time_str(timer.end())}")
    np.save(f'{save_path}/macs_losses_latency_{args.latency_idx}.npy', macs_losses_array)
    print(f"save to {save_path}/macs_losses_latency_{args.latency_idx}.npy, using time {time_str(timer.end())}")
    np.save(f'{save_path}/flops_all_latency_{args.latency_idx}.npy', flops_all_array)
    print(f"save to {save_path}/flops_all_latency_{args.latency_idx}.npy, using time {time_str(timer.end())}")

    save_json(skipped_samples, f'{save_path}/skipped_id_{args.latency_idx}.json')

    

def parge_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-onevision-qwen2-0.5b-si")
    parser.add_argument("--image-folder", type=str, default="./data/MME/images",
                        choices=[
                             "/home/ubuntu/projects/vqaData/data/llava_onevision",
                             "./data/MME/images"
                        ])
    parser.add_argument("--question-file", type=str, default="data/MME/json_qa/qa_MME_choice.json", 
                        choices=[
                            "data/MME/json_qa/subset_qa_MME_choice.json", 
                            "data/MME/json_qa/qa_MME_choice.json", 
                            "/home/ubuntu/projects/vqaData/data/llava_onevision/llava-onevision-si/jsons/ai2d_gpt4v.json",
                        ])
    parser.add_argument("--answers-file", type=str, default="answers/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")

    parser.add_argument("--mask-array", type=str, default="./mask_variations_5.npy")
    parser.add_argument("--save-path", type=str, default="data/MME/ada_losses/fullset/latency_56",)
    parser.add_argument("--latency", type=float, default=None)

    parser.add_argument("--mask-latency", type=str, default="./latency_variations_56.npy")
    parser.add_argument("--latency-idx", type=int, default=0)
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parge_args()
    # eval_model(args)
    main(args)