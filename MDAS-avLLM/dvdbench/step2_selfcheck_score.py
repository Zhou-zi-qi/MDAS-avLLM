import os
import torch
import pandas as pd
import json
import spacy
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ====================== 🔥 核心配置（只改这里） ======================
# 切换模式："visual" = 视觉描述打分；"audio" = 音频描述打分
MODE = "visual"

# 可选："NLI"  或  "BERTScore"
USE_DETECTOR = "NLI"
# =========================================================================

# ====================== 强制离线环境 ======================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ====================== 路径配置（自动按模式匹配） ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_MAP = {
    "visual": "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_selfcheck_visual_output.json",
    "audio":  "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_selfcheck_audio_output.json"
}
OUTPUT_CSV_MAP = {
    "visual": "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_selfcheck_visual_result.csv",
    "audio":  "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_selfcheck_audio_result.csv"
}
OUTPUT_JSON_MAP = {
    "visual": "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_selfcheck_visual_with_scores.json",
    "audio":  "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_selfcheck_audio_with_scores.json"
}

INPUT_JSON  = INPUT_MAP[MODE]
OUTPUT_CSV  = OUTPUT_CSV_MAP[MODE]
OUTPUT_JSON = OUTPUT_JSON_MAP[MODE]

LOCAL_MODEL = "/home/srt32/activation-steering/deberta-v3-large-mnli"

# ====================== 全局加载模型（只加载一次） ======================
print("加载模型与分词器...")
nlp = spacy.load("en_core_web_sm")

if USE_DETECTOR == "NLI":
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, local_files_only=True)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL, local_files_only=True
    ).to(device).eval()
    # 自动获取模型类别数，兼容二分类/三分类
    num_labels = nli_model.config.num_labels
    # 自动匹配蕴含类别的索引：三分类取2，二分类取1
    entail_idx = 2 if num_labels == 3 else 1
    print(f"模型类别数: {num_labels} | 蕴含类别索引: {entail_idx}")

elif USE_DETECTOR == "BERTScore":
    from bert_score import score

# ====================== 读取 DVD 数据 ======================
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ 加载数据：{len(data)} 个视频，当前模式：{MODE}")

# ====================== 统一打分函数（完全复用原逻辑） ======================
@torch.no_grad()
def get_consistency_score(sentence, references):
    if USE_DETECTOR == "NLI":
        scores = []
        for ref in references:
            inputs = tokenizer(
                sentence, ref, 
                max_length=256, padding=True, truncation=True, 
                return_tensors="pt"
            ).to(device)
            logits = nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
            # 取蕴含概率作为一致性分数
            entail_prob = probs[entail_idx].item()
            scores.append(entail_prob)
        return np.mean(scores)

    elif USE_DETECTOR == "BERTScore":
        P, R, F1 = score(
            [sentence] * len(references), references,
            model_type=LOCAL_MODEL,
            device=device,
            verbose=False,
            num_layers=24
        )
        return F1.mean().item()

# ====================== 打分逻辑 ======================
def score_hallucination(target_text, reference_texts):
    sentences = [s.text.strip() for s in nlp(target_text).sents if s.text.strip()]
    if not sentences:
        return [], [], 0.0
    sentence_scores = [get_consistency_score(s, reference_texts) for s in sentences]
    doc_score = np.mean(sentence_scores)
    return sentences, sentence_scores, doc_score

# ====================== 主逻辑（适配 DVD 字段） ======================
results_csv = []
results_json = []

for sample in tqdm(data, desc="打分进度"):
    video_id = sample["video_id"]
    mode = sample["mode"]
    prompt = sample.get("prompt", "")
    reference_caption = sample.get("reference_caption", "")
    sampled_texts = sample["sampled_descriptions"]
    
    scored_samples = []
    all_doc_scores = []

    for i in range(len(sampled_texts)):
        main_text = sampled_texts[i]
        # 留一法：其他9条作为参考
        refs = [sampled_texts[j] for j in range(len(sampled_texts)) if j != i]
        sents, sent_scores, doc_score = score_hallucination(main_text, refs)
        
        scored_samples.append({
            "text": main_text,
            "sentences": sents,
            "sentence_scores": sent_scores,
            "document_score": round(float(doc_score), 4)
        })
        all_doc_scores.append(doc_score)

    final_score = round(float(np.mean(all_doc_scores)), 4)
    results_csv.append({
        "video_id": video_id,
        "mode": mode,
        "final_hallucination_score": final_score
    })
    results_json.append({
        "video_id": video_id,
        "mode": mode,
        "prompt": prompt,
        "reference_caption": reference_caption,
        "final_hallucination_score": final_score,
        "sampled_descriptions_scored": scored_samples
    })

# ====================== 保存文件 ======================
pd.DataFrame(results_csv).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)

print(f"\n🏆 运行完成！当前检测器：{USE_DETECTOR} | 模式：{MODE}")
print(f"📄 汇总CSV：{OUTPUT_CSV}")
print(f"📄 句子级详细分数JSON：{OUTPUT_JSON}")