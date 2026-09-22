import json
import os
from typing import List, Dict

# ===================== 配置项（可直接修改） =====================
# 原始数据集路径
AVHBENCH_QA_JSON = "/home/srt32/activation-steering/avhbench_data/QA.json"
# 输出的条件向量训练集路径
OUTPUT_CONDITION_JSON = "/home/srt32/activation-steering/avh_condition_data.json"
# 跳过前2000条测试集，仅使用后续数据作为训练集
SKIP_SAMPLES = 2000

# ===================== 核心功能函数 =====================
def load_and_filter_questions(file_path: str) -> tuple[List[str], List[str]]:
    """
    加载数据集，筛选出两类幻觉问题（自动去重）
    :return: (音频驱动幻觉列表, 视频驱动幻觉列表)
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在：{file_path}")

    # 加载数据
    with open(file_path, 'r', encoding='utf-8') as f:
        qa_data: List[Dict] = json.load(f)

    audio_driven = []  # Audio-driven Video Hallucination
    video_driven = []  # Video-driven Audio Hallucination

    # 跳过测试集，遍历训练集
    for item in qa_data[SKIP_SAMPLES:]:
        # 安全校验：防止字段缺失报错
        if "task" not in item or "text" not in item:
            continue
        
        task = item["task"].strip()
        question = item["text"].strip()
        
        # 筛选目标任务 + 去重
        if task == "Audio-driven Video Hallucination" and question not in audio_driven:
            audio_driven.append(question)
        elif task == "Video-driven Audio Hallucination" and question not in video_driven:
            video_driven.append(question)

    return audio_driven, video_driven

def generate_training_pairs(audio_list: List[str], video_list: List[str]) -> List[Dict]:
    """
    一对一配对训练数据，取最短长度保证均衡
    """
    min_length = min(len(audio_list), len(video_list))
    pairs = []
    
    for a_ques, v_ques in zip(audio_list[:min_length], video_list[:min_length]):
        pairs.append({
            "audio_driven_video_hallucination": a_ques,
            "video_driven_audio_hallucination": v_ques
        })
    return pairs

def save_training_data(pairs: List[Dict], save_path: str):
    """
    保存训练数据为标准格式
    """
    result = {"train": pairs}
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

# ===================== 主执行逻辑 =====================
if __name__ == "__main__":
    print("🔄 开始生成条件向量训练数据集...")
    
    # 1. 加载并筛选数据
    audio_questions, video_questions = load_and_filter_questions(AVHBENCH_QA_JSON)
    
    # 2. 生成配对
    train_pairs = generate_training_pairs(audio_questions, video_questions)
    
    # 3. 保存文件
    save_training_data(train_pairs, OUTPUT_CONDITION_JSON)
    
    # 4. 打印详细统计
    print("=" * 50)
    print("✅ 训练数据集生成完成！")
    print(f"📊 音频驱动幻觉样本数：{len(audio_questions)}")
    print(f"📊 视频驱动幻觉样本数：{len(video_questions)}")
    print(f"📊 最终均衡训练配对数：{len(train_pairs)}")
    print(f"📂 文件保存路径：{OUTPUT_CONDITION_JSON}")
    print("=" * 50)