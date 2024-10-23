Clone the LLM-Viewer repository from GitHub: 
```git clone https://github.com/hahnyuan/LLM-Viewer.git   ```

```
analyzer = FlexibleAnalyzer(
    args.model_id, 
    args.hardware, 
    args.config_file, 
    source=args.source
)

results = analyzer.analyze_generate_task(
    prompt_len=128,
    gen_len=2048,
    num_heads=[32] * 28,
    batchsize=args.batchsize,
    w_bit=args.w_bit,
    a_bit=args.a_bit,
    kv_bit=args.kv_bit,
    use_flashattention=args.use_flashattention,
    tp_size=args.tp_size
)
```

Arguments: 
  - ```prompt_len```: number of prompt tokens
  - ```gen_len```: number of generated tokens
  - ```num_heads```: a list of number of attention heads for each layers. The length of the list is the number of executed layers. E.g. [32] * 28 means executing 28 layers, each has 32 heads. [8, 12, 24] means executing 3 layers with number of heads equal to 8, 12, 24 respectively. 
  - ```batchsize```: batch size
