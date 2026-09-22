import pandas as pd
import json
import sys
import io

# ========== 编码不乱码 ==========
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ====================== 路径配置 ======================
INPUT_JSON = "/home/srt32/activation-steering/avhbench_selfcheck_output_nomain_with_scores_video.json"
OUTPUT_PAIRS_PATH = "/home/srt32/activation-steering/avhbench_sentence_contrast_pairs_video.csv"

# ====================== 配置 ======================
MIN_SENTENCE_LEN = 5
MAX_PAIRS_PER_VIDEO = 2  # 每个视频固定2组最优对比对

# ====================== 读取数据 ======================
print("reading scored data...")
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"loaded data: {len(data)} samples")

# ====================== 核心：动态排序取最值（无阈值，数量拉满） ======================
all_sentence_pairs = []

for idx, sample in enumerate(data):
    video_id = sample["video_id"]
    task = sample["task"]
    prompt = sample.get("prompt", "")
    scored_descriptions = sample["sampled_descriptions_scored"]

    # 收集这个视频下【所有】句子+分数（不设阈值，全部参与内部排序）
    all_sentences_with_score = []

    for desc in scored_descriptions:
        sentences = desc["sentences"]
        sentence_scores = desc["sentence_scores"]

        for sent, score in zip(sentences, sentence_scores):
            sent = sent.strip()
            if len(sent) < MIN_SENTENCE_LEN:
                continue
            all_sentences_with_score.append((score, sent))

    # 关键：当前视频内部分数排序，生成正负样本
    if len(all_sentences_with_score) >= 2:
        # 1. 全部句子按分数从高到低排序
        sorted_all = sorted(all_sentences_with_score, key=lambda x: -x[0])
        
        # 2. 正例：分数最高的2句
        pos_candidates = sorted_all[:MAX_PAIRS_PER_VIDEO]
        # 3. 负例：分数最低的2句（取末尾）
        neg_candidates = sorted_all[-MAX_PAIRS_PER_VIDEO:]
        
        # 提取句子
        pos_final = [item[1] for item in pos_candidates]
        neg_final = [item[1] for item in neg_candidates]
        
        # 一对一配对
        for pos, neg in zip(pos_final, neg_final):
            all_sentence_pairs.append({
                "video_id": video_id,
                "task": task,
                "prompt": prompt,
                "positive_sentence": pos,
                "negative_sentence": neg,
                "source_sample_idx": idx + 1
            })

# ====================== 去重 + 保存 ======================
df_pairs = pd.DataFrame(all_sentence_pairs)
df_pairs = df_pairs.drop_duplicates(subset=["positive_sentence", "negative_sentence"])

print(f"\n[OK] 生成完成！共 {len(df_pairs)} 组高质量对比对！")
df_pairs.to_csv(OUTPUT_PAIRS_PATH, index=False, encoding="utf-8-sig")
print(f"[OK] 结果保存到：{OUTPUT_PAIRS_PATH}")

# ====================== 展示示例 ======================
print("\n===== Top 5 Contrast Pairs =====")
for i in range(min(5, len(df_pairs))):
    row = df_pairs.iloc[i]
    print(f"\n[Video ID] {row['video_id']} | [Task] {row['task']}")
    print(f"[Prompt] {str(row['prompt'])[:50]}...")
    print(f"✅ [Real] {row['positive_sentence']}")
    print(f"❌ [Hallucination] {row['negative_sentence']}")
    print("-" * 80)