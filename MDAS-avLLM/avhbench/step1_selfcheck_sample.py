import os
import gc
import torch
import json
import warnings
from tqdm import tqdm
from qwen_omni_utils import process_mm_info
warnings.filterwarnings("ignore")

# ====================== 1. 基础环境配置 ======================
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.cuda.empty_cache()
gc.collect()

MODEL_PATH = "/home/srt32/activation-steering/Qwen2.5-Omni-3B"
USE_AUDIO_IN_VIDEO = True

# ====================== 2. 数据集 & SelfCheck 配置 ======================
AVHBENCH_VIDEO_DIR = "/home/srt32/activation-steering/avhbench_data/videos"
AVHBENCH_QA_JSON = "/home/srt32/activation-steering/avhbench_data/QA.json"
OUTPUT_JSON = "/home/srt32/activation-steering/avhbench_selfcheck_output_nomain_audio.json"

START_IDX = 2000
END_IDX = 2200
SAMPLE_COUNT = 10  # 20次采样
PRINT_MAX = 10

FIXED_DESCRIBE_PROMPT = (
     "Describe the audio content of this video in detail, "
    "including all sounds, audio events, sound types, voices and ambient environment sounds. "
    "Just describe objectively, do not ask questions, do not interact with me."
)

# ====================== 3. 生成参数 ======================
GEN_CONFIG = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "num_beams": 1,
    "pad_token_id": 151643,
    "eos_token_id": 151643,
}

# ====================== 4. 加载数据 ======================
def load_avhbench_questions():
    with open(AVHBENCH_QA_JSON, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    selected_data = qa_data[START_IDX:END_IDX]
    multimodal_questions = []
    for sample in selected_data:
        video_id = sample["video_id"]
        video_path = os.path.join(AVHBENCH_VIDEO_DIR, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            continue
        question = {
            "text": FIXED_DESCRIBE_PROMPT,
            "multimodal": [{"type": "video", "video": video_path}],
            "video_id": video_id,
            "task": sample["task"],
            "label": sample["label"]
        }
        multimodal_questions.append(question)
    print(f"✅ 加载 AVHBench 描述任务样本：{len(multimodal_questions)} 条")
    return multimodal_questions

multimodal_questions = load_avhbench_questions()

# ====================== 5. 加载模型 ======================
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

print("🔄 加载 Qwen2.5-Omni ...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True
)
model.disable_talker()
model.eval()

processor = Qwen2_5OmniProcessor.from_pretrained(
    MODEL_PATH, use_fast=False, trust_remote_code=True, local_files_only=True
)
tokenizer = processor.tokenizer

# ====================== 6. 构建对话（官方完整System Prompt，必选！） ======================
def build_conversation(question_dict):
    user_content = [{"type": "text", "text": question_dict["text"]}]
    for modal in question_dict["multimodal"]:
        m_type = modal["type"]
        user_content.append({"type": m_type, m_type: modal[m_type]})

    # 🔥 恢复官方完整默认System Prompt（解决卡死+警告）
    conversation = [
        {"role": "system", "content": [{"type": "text", 
            "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": user_content}
    ]
    return conversation

# ====================== 7. 核心优化：只预处理1次输入，生成20次 ======================
def get_model_inputs(conversation):
    # 视频/音频只处理1次，复用20次（彻底解决卡死）
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    kw = {
        "text": text, "return_tensors": "pt", "padding": True,
        "use_audio_in_video": USE_AUDIO_IN_VIDEO
    }
    if images: kw["images"] = images
    if videos: kw["videos"] = videos
    if audios is not None:
        kw["audio"] = audios.cpu().numpy() if torch.is_tensor(audios) else audios
    return processor(**kw).to(model.device, model.dtype)

def generate_from_inputs(inputs):
    with torch.no_grad():
        ids = model.generate(**inputs, **GEN_CONFIG)
    ans = processor.batch_decode(ids, skip_special_tokens=True)[0]
    ans = ans.split("assistant")[-1].strip() if "assistant" in ans else ans.strip()
    return ans

# ====================== 8. 主流程 ======================
if __name__ == "__main__":
    print("===== AVHBench 10次采样（纯音频描述）=====")
    all_results = []

    for idx, q_dict in enumerate(tqdm(multimodal_questions, desc="生成进度"), 1):
        video_id = q_dict["video_id"]
        task = q_dict["task"]
        conv = build_conversation(q_dict)
        
        # 🔥 关键：1次预处理，复用20次
        model_inputs = get_model_inputs(conv)
        sample_descs = []
        
        for _ in range(SAMPLE_COUNT):
            sample_descs.append(generate_from_inputs(model_inputs))
        
        # 预览
        if idx <= PRINT_MAX:
            print(f"\n【{idx}】video_id: {video_id}")
            print(f"采样预览: {sample_descs[0][:200]}...")

        # 保存结果
        all_results.append({
            "video_id": video_id,
            "task": task,
            "prompt": FIXED_DESCRIBE_PROMPT,
            "sampled_descriptions": sample_descs
        })
        
        # 显存清理
        del model_inputs
        torch.cuda.empty_cache()
        gc.collect()

    # 保存文件
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 完成！文件已保存：{OUTPUT_JSON}")