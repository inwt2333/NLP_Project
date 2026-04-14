import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress import KVCacheCompressor
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# 1. 加载模型和分词器 (使用你已经下载好的路径)
model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)
tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

# 2. 初始化 kvpress
# 这里我们选一个简单的压缩方法，比如 H2O
compressor = KVCacheCompressor(model, method="h2o", kv_cache_budget=128)

# 3. 准备输入
prompt = "The quick brown fox jumps over"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 4. 运行推理 (在 kvpress 的上下文中)
with compressor:
    output = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(output[0], skip_special_tokens=True))