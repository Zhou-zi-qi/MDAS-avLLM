import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# ====================== 🔥 核心配置（只改这里） ======================
# 切换模式："visual" = 视觉幻觉训练对；"audio" = 音频幻觉训练对
MODE = "audio"

# 分位数配置：正例取前TOP_PCT高分，负例取后BOTTOM_PCT低分
TOP_PCT = 0.2    # 前20%高分作为真实正例
BOTTOM_PCT = 0.2 # 后20%低分作为幻觉负例

# 过滤配置
MIN_SENTENCE_LEN = 8    # 最小句子长度，过滤碎句
MAX_PAIRS_PER_VIDEO = 60  # 每个视频最多生成多少对，避免样本爆炸
# ===================================================================

# ====================== 路径自动匹配 ======================
INPUT_MAP = {
    "visual": "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_selfcheck_visual_with_scores.json",
    "audio":  "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_selfcheck_audio_with_scores.json"
}
OUTPUT_CSV_MAP = {
    "visual": "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_visual_hallucination_pairs.csv",
    "audio":  "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_audio_hallucination_pairs.csv"
}
OUTPUT_JSON_MAP = {
    "visual": "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_visual_hallucination_pairs.json",
    "audio":  "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_audio_hallucination_pairs.json"
}

INPUT_JSON = INPUT_MAP[MODE]
OUTPUT_CSV = OUTPUT_CSV_MAP[MODE]
OUTPUT_JSON = OUTPUT_JSON_MAP[MODE]

# ====================== 加载数据 ======================
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ 加载数据：{len(data)} 个视频，当前模式：{MODE}")
print(f"⚙️  配置：正例前{TOP_PCT*100:.0f}% | 负例后{BOTTOM_PCT*100:.0f}% | 单视频最多{MAX_PAIRS_PER_VIDEO}对")

# ====================== 构造配对 ======================
all_pairs = []
total_pos = 0
total_neg = 0

for sample in tqdm(data, desc="构造训练对"):
    video_id = sample["video_id"]
    
    # 收集该视频所有句子及其分数
    all_sentences = []
    all_scores = []
    for desc in sample["sampled_descriptions_scored"]:
        for sent, score in zip(desc["sentences"], desc["sentence_scores"]):
            sent = sent.strip()
            # 过滤过短句子
            if len(sent.split()) < MIN_SENTENCE_LEN:
                continue
            all_sentences.append(sent)
            all_scores.append(score)
    
    if len(all_sentences) < 4:
        continue  # 句子太少的跳过
    
    all_scores = np.array(all_scores)
    all_sentences = np.array(all_sentences)
    
    # 计算分位数阈值
    top_thresh = np.quantile(all_scores, 1 - TOP_PCT)
    bottom_thresh = np.quantile(all_scores, BOTTOM_PCT)
    
    # 筛选正负例
    pos_mask = all_scores >= top_thresh
    neg_mask = all_scores <= bottom_thresh
    
    pos_sents = all_sentences[pos_mask].tolist()
    neg_sents = all_sentences[neg_mask].tolist()
    
    total_pos += len(pos_sents)
    total_neg += len(neg_sents)
    
    # 交叉配对，限制最大数量
    pairs = []
    for pos_sent in pos_sents:
        for neg_sent in neg_sents:
            if pos_sent == neg_sent:
                continue
            pairs.append({
                "video_id": video_id,
                "positive": pos_sent,  # 真实描述（高分）
                "negative": neg_sent,  # 幻觉描述（低分）
                "pos_score": float(all_scores[pos_mask][pos_sents.index(pos_sent)]),
                "neg_score": float(all_scores[neg_mask][neg_sents.index(neg_sent)]),
                "score_gap": float(all_scores[pos_mask][pos_sents.index(pos_sent)] - all_scores[neg_mask][neg_sents.index(neg_sent)])
            })
            if len(pairs) >= MAX_PAIRS_PER_VIDEO:
                break
        if len(pairs) >= MAX_PAIRS_PER_VIDEO:
            break
    
    all_pairs.extend(pairs)

# ====================== 保存结果 ======================
# 保存CSV（兼容PCA训练代码，直接用）
df = pd.DataFrame(all_pairs)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# 保存JSON（方便排查）
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(all_pairs, f, ensure_ascii=False, indent=2)

# ====================== 统计信息 ======================
avg_gap = np.mean([p["score_gap"] for p in all_pairs])
print(f"\n🏆 训练对构造完成！")
print(f"📊 总配对数：{len(all_pairs)}")
print(f"📊 正例总句子数：{total_pos} | 负例总句子数：{total_neg}")
print(f"📊 平均正负分差：{avg_gap:.4f}（分差越大对比度越强，向量效果越好）")
print(f"📄 CSV训练文件：{OUTPUT_CSV}")
print(f"📄 JSON详情文件：{OUTPUT_JSON}")