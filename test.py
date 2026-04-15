import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from streaming_llm.enable_streaming_llm import enable_streaming_llm
import math


# 如果是 Apple Silicon (M1/M2/M3)，优先使用 mps 加速
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "EleutherAI/pythia-70m-deduped"
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
).to(device)
model.eval()

model_streaming = AutoModelForCausalLM.from_pretrained(
  model_id,
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
).to(device)
model_streaming.eval()

enable_streaming_llm(model_streaming, start_size=4, recent_size=1020)

tokenizer = AutoTokenizer.from_pretrained(
  model_id,
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)
tokenizer.pad_token = tokenizer.eos_token

def measure_latency(model, tokenizer, prompt, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # ===== 测试 TTFT (第一个Token的时间) =====
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    
    first_token_time = time.perf_counter()
    ttft = first_token_time - start_time
    
    # ===== 测试 TPOT (后续生成每个Token的平均时间) =====
    past_key_values = outputs.past_key_values
    input_ids = next_token
    
    start_tpot = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_token_logits = out.logits[:, -1, :]
            input_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
    end_time = time.perf_counter()
    
    tpot_total_time = end_time - start_tpot
    tpot = tpot_total_time / (max_new_tokens - 1)
    
    total_time = end_time - start_time
    throughput = max_new_tokens / total_time 
    
    return {
        "TTFT (ms)": round(ttft * 1000, 2),
        "TPOT (ms/token)": round(tpot * 1000, 2),
        "Throughput (tokens/s)": round(throughput, 2)
    }



def calculate_streaming_ppl(model, tokenizer, dataset, text_column, sample_size=1, max_tokens=4000, chunk_size=256, kv_cache_evictor=None):
    """
    真正的“流式” PPL 评测：把过去的记忆 (past_key_values) 连续往后保留！
    如果传入了 kv_cache_evictor，会在每次生成后截断缓存，这才是真正触发 StreamingLLM 内存优化的逻辑。
    """
    total_nll = 0.0
    total_tokens = 0
    
    texts = [dataset[i][text_column] for i in range(min(sample_size, len(dataset))) if dataset[i][text_column].strip()]
    text = "\n\n".join(texts)
    
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[:, :max_tokens].to(model.device)
    seq_len = input_ids.size(1)
    
    past_key_values = None
    
    # 按照 256 为字块，像看书一样连续读下去，带着过去全部的记忆
    for i in tqdm(range(0, seq_len - 1, chunk_size), desc="Streaming Eval"):
        chunk_ids = input_ids[:, i : i + chunk_size]
        
        with torch.no_grad():
            outputs = model(chunk_ids, past_key_values=past_key_values, use_cache=True, labels=chunk_ids)
            
            valid_len = chunk_ids.size(1) - 1
            if valid_len > 0:
                total_nll += outputs.loss.item() * valid_len
                total_tokens += valid_len
            
            # StreamingLLM 的核心：必须手动截断超出的 KV 缓存！
            if kv_cache_evictor is not None:
                past_key_values = kv_cache_evictor(outputs.past_key_values)
            else:
                past_key_values = outputs.past_key_values

    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float('nan')

if __name__ == "__main__":
    prompt = "The future of artificial intelligence in natural language processing is"
    latency_metrics = measure_latency(model, tokenizer, prompt, max_new_tokens=64)
    print("=== 原始性能指标 Baseline ===")
    for k, v in latency_metrics.items():
        print(f"{k}: {v}")
    latency_metrics = measure_latency(model_streaming, tokenizer, prompt, max_new_tokens=64)
    print("=== 流式性能指标 Streaming_LLM ===")
    for k, v in latency_metrics.items():
        print(f"{k}: {v}")
    print("Loading test datasets...")
    pg19_test = load_dataset("emozilla/pg19-test", split="test", streaming=True)
    pg19_sample = list(pg19_test.take(1))

    wiki_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    print("=== 初始化 StreamingLLM 的 KV Cache 截断器 ===")
    # 之前在其他 cell 调用了 enable_streaming_llm 但没有保存返回值，
    # 我们这里重新调用并保存，因为它的内部有 evict 逻辑处理。
    kv_cache_evictor = enable_streaming_llm(model_streaming, start_size=4, recent_size=1020)

    print("\n开始长文档连续流式测试 (4000 tokens)")

    print("\n--- PG-19 测试 ---")
    pg19_ppl_baseline = calculate_streaming_ppl(model, tokenizer, pg19_sample, text_column="text", max_tokens=4000, chunk_size=256)
    print(f"PG-19 PPL Baseline: {pg19_ppl_baseline:.2f}")

    pg19_ppl_streaming = calculate_streaming_ppl(model_streaming, tokenizer, pg19_sample, text_column="text", max_tokens=4000, chunk_size=256, kv_cache_evictor=kv_cache_evictor)
    print(f"PG-19 PPL StreamingLLM: {pg19_ppl_streaming:.2f}")

    print("\n--- Wikitext-2 测试 ---")
    # Wikitext的单条数据较短，我们增加 sample_size 获取足够长的数据用于评测
    wiki_ppl_baseline = calculate_streaming_ppl(model, tokenizer, wiki_test, text_column="text", sample_size=1000, max_tokens=4000, chunk_size=256)
    print(f"Wikitext-2 PPL Baseline: {wiki_ppl_baseline:.2f}")

    wiki_ppl_streaming = calculate_streaming_ppl(model_streaming, tokenizer, wiki_test, text_column="text", sample_size=1000, max_tokens=4000, chunk_size=256, kv_cache_evictor=kv_cache_evictor)
    print(f"Wikitext-2 PPL StreamingLLM: {wiki_ppl_streaming:.2f}")