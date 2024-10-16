# adaLlava

#### Installation
```
conda create -n adallava python=3.10 -y
conda activate adallava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```

#### Naive inference
Inference using autoregressive generation.

```
python -m llava.eval.generation.infer_mme
```
replace post prompt with lmm setting, but has higher score. Leave it for now.

#### Inference baseline for different branches
##### generate different branches
```
python debug/branches.py
```
get `mask_variations_5.npy`, shape of [102, 21], keep first 3 blocks, random select rest of 21 blocks. Total sampled 102 branches.

##### Run inference for different branches for `mask_variations_5.npy`.
```
python -m llava.eval.forwards.infer_mme --branch-idx $branch
```

where `branch-idx` range from 0-101.

batched operation
```
bash scripts/eval/forwards/eval_branch.sh
```
save loss to `/home/ubuntu/projects/vqaData/data/MME/ada_losses`.


#### Inference baseline for adascheduler
```
python -m llava.eval.forwards.infer_adascheduler_mme
```



-----
###### model architect
```
lmms-lab/llava-onevision-qwen2-0.5b-si: 

AdaptiveLlavaQwenForCausalLM(
  (model): AdaptiveLlavaQwenModel(
    (embed_tokens): Embedding(151647, 896)
    (layers): ModuleList(
      (0-23): 24 x AdaptiveQwen2DecoderLayer(
        (self_attn): Qwen2FlashAttention2(
          (q_proj): Linear(in_features=896, out_features=896, bias=True)
          (k_proj): Linear(in_features=896, out_features=128, bias=True)
          (v_proj): Linear(in_features=896, out_features=128, bias=True)
          (o_proj): Linear(in_features=896, out_features=896, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
          (up_proj): Linear(in_features=896, out_features=4864, bias=False)
          (down_proj): Linear(in_features=4864, out_features=896, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm()
        (post_attention_layernorm): Qwen2RMSNorm()
      )
    )
    (norm): Qwen2RMSNorm()
    (vision_tower): SigLipVisionTower(
      (vision_tower): SigLipVisionModel(
        (vision_model): SigLipVisionTransformer(
          (embeddings): SigLipVisionEmbeddings(
            (patch_embedding): Conv2d(3, 1152, kernel_size=(14, 14), stride=(14, 14), padding=valid)
            (position_embedding): Embedding(729, 1152)
          )
          (encoder): SigLipEncoder(
            (layers): ModuleList(
              (0-25): 26 x SigLipEncoderLayer(
                (self_attn): SigLipAttention(
                  (k_proj): Linear(in_features=1152, out_features=1152, bias=True)
                  (v_proj): Linear(in_features=1152, out_features=1152, bias=True)
                  (q_proj): Linear(in_features=1152, out_features=1152, bias=True)
                  (out_proj): Linear(in_features=1152, out_features=1152, bias=True)
                )
                (layer_norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
                (mlp): SigLipMLP(
                  (activation_fn): PytorchGELUTanh()
                  (fc1): Linear(in_features=1152, out_features=4304, bias=True)
                  (fc2): Linear(in_features=4304, out_features=1152, bias=True)
                )
                (layer_norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
              )
            )
          )
          (post_layernorm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True)
          (head): Identity()
        )
      )
    )
    (vision_resampler): IdentityMap()
    (mm_projector): Sequential(
      (0): Linear(in_features=1152, out_features=896, bias=True)
      (1): GELU(approximate='none')
      (2): Linear(in_features=896, out_features=896, bias=True)
    )
  )
  (lm_head): Linear(in_features=896, out_features=151647, bias=False)
)

```
