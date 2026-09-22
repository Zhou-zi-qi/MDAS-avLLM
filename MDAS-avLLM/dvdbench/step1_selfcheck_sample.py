import os
import sys
import gc
import torch
import json
import warnings
from tqdm import tqdm

# 添加上级目录，导入工具包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_omni_utils import process_mm_info

warnings.filterwarnings("ignore")

# ====================== 🔥 核心配置（只改这里） ======================
# 切换模式："visual" = 纯视觉描述；"audio" = 纯音频描述
MODE = "visual"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.cuda.empty_cache()
gc.collect()

MODEL_PATH = "/home/srt32/activation-steering/Qwen2.5-Omni-3B"
# 🔥 根据模式自动开关音频：视觉模式关闭，音频模式开启
USE_AUDIO_IN_VIDEO = False if MODE == "visual" else True

# 数据集路径（已更新为转码后目录）
INPUT_DATA = "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_train_ytb_valid.json"
# 输出文件按模式自动区分
OUTPUT_MAP = {
    "visual": "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_selfcheck_visual_output.json",
    "audio":  "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_selfcheck_audio_output.json"
}
OUTPUT_JSON = OUTPUT_MAP[MODE]

# 处理样本范围
START_IDX = 0
END_IDX = 988
SAMPLE_COUNT = 10  # 每个视频采样次数
PRINT_MAX = 5

# 两套描述 Prompt，完全对应你的版本
PROMPT_MAP = {
    "visual": (
        "Describe the visual content of this video in detail, "
        "including all visible objects, scenes, colors, layout, people, actions and environment. "
        "Just describe objectively, do not ask questions, do not interact with me."
    ),
    "audio": (
        "Describe the audio content of this video in detail, "
        "including all sounds, voices, timbre, rhythm, volume and acoustic environment. "
        "Just describe objectively, do not ask questions, do not interact with me."
    )
}
FIXED_DESCRIBE_PROMPT = PROMPT_MAP[MODE]

# ====================== 生成参数（放开长度限制） ======================
GEN_CONFIG = {
    "max_new_tokens": 2048,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "num_beams": 1,
    "pad_token_id": 151643,
    "eos_token_id": 151643,
}

# ====================== 加载 DVD 数据集 ======================
def load_dvd_samples():
    with open(INPUT_DATA, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    selected_data = all_data[START_IDX:END_IDX]
    multimodal_questions = []
    
    for sample in selected_data:
        video_path = sample["video_path"]
        if not os.path.exists(video_path):
            continue
        
        question = {
            "text": FIXED_DESCRIBE_PROMPT,
            "multimodal": [{"type": "video", "video": video_path}],
            "video_id": sample["video_id"],
            "reference_caption": sample["reference_caption"]
        }
        multimodal_questions.append(question)
    
    print(f"✅ 加载 DVD 视频样本：{len(multimodal_questions)} 条")
    print(f"📝 当前模式：{MODE} | 音频解码：{'开启' if USE_AUDIO_IN_VIDEO else '关闭'}")
    print(f"📝 描述Prompt：{FIXED_DESCRIBE_PROMPT[:80]}...")
    return multimodal_questions

multimodal_questions = load_dvd_samples()

# ====================== 加载模型 ======================
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

# ====================== 构建对话 ======================
def build_conversation(question_dict):
    user_content = [{"type": "text", "text": question_dict["text"]}]
    for modal in question_dict["multimodal"]:
        m_type = modal["type"]
        user_content.append({"type": m_type, m_type: modal[m_type]})

    conversation = [
        {"role": "system", "content": [{"type": "text", 
            "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": user_content}
    ]
    return conversation

# ====================== 核心优化：1次预处理，N次采样复用 ======================
def get_model_inputs(conversation):
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

# ====================== 主流程（容错版） ======================
if __name__ == "__main__":
    print(f"\n===== DVD 数据集 SelfCheck 多采样生成【{MODE}模式】 =====")
    print(f"处理范围：第 {START_IDX} ~ {END_IDX} 条 | 每个视频采样 {SAMPLE_COUNT} 次")
    all_results = []
    failed_videos = []

    for idx, q_dict in enumerate(tqdm(multimodal_questions, desc="生成进度"), 1):
        video_id = q_dict["video_id"]
        conv = build_conversation(q_dict)

        try:
            model_inputs = get_model_inputs(conv)
            sample_descs = []

            for _ in range(SAMPLE_COUNT):
                sample_descs.append(generate_from_inputs(model_inputs))

            if idx <= PRINT_MAX:
                print(f"\n【{idx}】video_id: {video_id}")
                print(f"采样预览: {sample_descs[0][:200]}...")

            all_results.append({
                "video_id": video_id,
                "mode": MODE,
                "prompt": FIXED_DESCRIBE_PROMPT,
                "reference_caption": q_dict["reference_caption"],
                "sampled_descriptions": sample_descs
            })

            del model_inputs
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            failed_videos.append({"video_id": video_id, "error": str(e)})
            print(f"\n⚠️  跳过失败视频：{video_id}，错误：{str(e)[:80]}")
            continue

    # 保存结果
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    failed_log = OUTPUT_JSON.replace(".json", "_failed.json")
    with open(failed_log, 'w', encoding='utf-8') as f:
        json.dump(failed_videos, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 生成完成！成功：{len(all_results)} 个 | 失败：{len(failed_videos)} 个")
    print(f"📄 结果文件：{OUTPUT_JSON}")
    print(f"📄 失败日志：{failed_log}")