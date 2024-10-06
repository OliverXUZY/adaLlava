# adaLlava

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

