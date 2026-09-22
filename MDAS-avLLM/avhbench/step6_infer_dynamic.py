import os
import gc
import json
import torch
import warnings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm
from qwen_omni_utils import process_mm_info

# ===================== 屏蔽所有冗余输出 =====================
warnings.filterwarnings("ignore")
import transformers
transformers.logging.set_verbosity_error()
import logging
logging.getLogger().setLevel(logging.CRITICAL)

# ===================== 1. 环境配置 =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

torch.cuda.empty_cache()
gc.collect()

# ===================== 路径配置 =====================
LOCAL_MODEL_PATH = "/home/srt32/activation-steering/Qwen2.5-Omni-3B"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

AVHBENCH_VIDEO_DIR = "/home/srt32/activation-steering/avhbench_data/videos"
AVHBENCH_QA_JSON = "/home/srt32/activation-steering/avhbench_data/QA.json"
OUTPUT_JSON = "/home/srt32/activation-steering/rh_qwen_avhbench_result_condition1.json"

CONDITION_VECTOR_PATH = "avh_hallucination_classify_vector.svec"
OPTIMAL_POINT_PATH = "optimal_condition_point_avh.json"
VECTOR_AUDIO2VIDEO = "hallucination_reduction_vector_avhbench_videonew2.svec"
VECTOR_VIDEO2AUDIO = "hallucination_reduction_vector_avhbench_audionew2.svec"

SAMPLE_NUM = 2000
PRINT_MAX = 10
USE_AUDIO_IN_VIDEO = True

# 权重过渡区间宽度（余弦相似度范围），阈值±半宽为线性过渡区
TRANSITION_WIDTH = 0.1

# 🔥 两组独立配置：向量 + 注入层 + 强度一一对应（原配置保持不变）
# 视频幻觉向量（音频驱动视频幻觉任务）
LAYERS_A2V = [25, 26, 27, 28]
STRENGTH_A2V = 2.0
# 音频幻觉向量（视频驱动音频幻觉任务）
LAYERS_V2A = [24, 25, 26, 27]
STRENGTH_V2A = 3.0

# ===================== 2. 核心补丁 =====================
from activation_steering.malleable_model import LeashLayer
original_leash_init = LeashLayer.__init__
def patched_leash_init(self, layer, *args, **kwargs):
    original_leash_init(self, layer, *args, **kwargs)
    if hasattr(layer, 'attention_type'):
        self.attention_type = layer.attention_type
    if hasattr(layer, 'layer_idx'):
        self.layer_idx = layer.layer_idx
LeashLayer.__init__ = patched_leash_init

from activation_steering import malleable_model as mm_module
from activation_steering import steering_vector as sv_module
def global_get_model_layer_list(model):
    if "Qwen2_5OmniForConditionalGeneration" in str(type(model)):
        return model.thinker.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
mm_module.get_model_layer_list = global_get_model_layer_list
sv_module.get_model_layer_list = global_get_model_layer_list

# ===================== 3. 加载模型 =====================
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from activation_steering import MalleableModel, SteeringVector

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    dtype=torch.float16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True
)

def qwen_text_forward_wrapper(input_ids, attention_mask=None, **kwargs):
    kwargs.pop('output_hidden_states', None)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model.thinker.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            **kwargs
        )
    lm_head_weight = model.thinker.lm_head.weight.to(torch.float16)
    logits = outputs.last_hidden_state @ lm_head_weight.t()
    setattr(outputs, 'logits', logits)
    setattr(outputs, 'hidden_states', outputs.hidden_states)
    return outputs
model.forward = qwen_text_forward_wrapper

model.layers = model.thinker.model.layers
model.config.num_hidden_layers = len(model.layers)
model.config.n_layer = len(model.layers)
model.config.hidden_size = 2048
model.config.model_type = "qwen2_5_omni"
model.disable_talker()
model.eval()

processor = Qwen2_5OmniProcessor.from_pretrained(
    LOCAL_MODEL_PATH, trust_remote_code=True, use_fast=False, local_files_only=True
)
tokenizer = processor.tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("✅ Qwen模型+Tokenizer加载完成，层数：", len(model.layers))

# ===================== 4. 加载向量与原配置参数 =====================
print("\n===== 加载向量与原配置参数 =====")
condition_vector = SteeringVector.load(CONDITION_VECTOR_PATH)
vec_a2v = SteeringVector.load(VECTOR_AUDIO2VIDEO)
vec_v2a = SteeringVector.load(VECTOR_VIDEO2AUDIO)

# 直接使用训练集得到的原配置，不做测试集搜索调优
with open(OPTIMAL_POINT_PATH, 'r', encoding='utf-8') as f:
    optimal_data = json.load(f)
best_layer = optimal_data["best_layers"][0]
best_threshold = optimal_data["best_threshold"]
best_direction = optimal_data["best_direction"]

print(f"📌 原配置分类参数：层={best_layer} | 阈值={best_threshold:.4f} | 方向={best_direction}")
print(f"📌 视频幻觉向量（A2V）：层={LAYERS_A2V} | 基础强度={STRENGTH_A2V}")
print(f"📌 音频幻觉向量（V2A）：层={LAYERS_V2A} | 基础强度={STRENGTH_V2A}")
print(f"📌 权重过渡区间宽度：{TRANSITION_WIDTH}")

# ===================== 5. 加载测试集 =====================
def load_avhbench_questions():
    with open(AVHBENCH_QA_JSON, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    qa_data = qa_data[:SAMPLE_NUM]
    multimodal_questions = []
    for sample in qa_data:
        video_id = sample["video_id"]
        video_path = os.path.join(AVHBENCH_VIDEO_DIR, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            continue
        multimodal_questions.append({
            "text": sample["text"],
            "multimodal": [{"type": "video", "video": video_path}],
            "video_id": video_id, "task": sample["task"], "label": sample["label"]
        })
    print(f"✅ 加载测试样本：{len(multimodal_questions)}")
    return multimodal_questions
multimodal_questions = load_avhbench_questions()

# ===================== 6. 对话构建 =====================
def build_conversation(question_dict):
    user_content = [{"type": "text", "text": question_dict["text"]}]
    for modal in question_dict["multimodal"]:
        user_content.append({"type": modal["type"], modal["type"]: modal[modal["type"]]})
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            }]
        },
        {"role": "user", "content": user_content}
    ]

# ===================== 条件向量连续权重计算 =====================
@torch.no_grad()
def calc_hallucination_weights(question_text):
    """
    根据条件向量计算两个幻觉方向的加权系数
    返回: w_a2v（视频幻觉向量权重）, w_v2a（音频幻觉向量权重）
    阈值处权重为 0.5:0.5，过渡区间内线性平滑
    """
    system_prompt = {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]}
    conversation = [system_prompt, {"role": "user", "content": [{"type": "text", "text": question_text}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    target_hidden = hidden_states[best_layer]
    last_token_hidden = target_hidden[:, -1, :].squeeze()

    direction = torch.tensor(condition_vector.directions[best_layer], dtype=model.dtype, device=model.device)
    score = torch.cosine_similarity(last_token_hidden, direction, dim=0).item()

    half_width = TRANSITION_WIDTH / 2

    # 根据方向映射：将分数归一化到 0=纯A2V, 1=纯V2A 的连续值
    if best_direction == "smaller":
        # 小于阈值 → 偏向A2V；大于阈值 → 偏向V2A
        normalized = (score - (best_threshold - half_width)) / TRANSITION_WIDTH
    else:
        # 大于阈值 → 偏向A2V；小于阈值 → 偏向V2A
        normalized = ((best_threshold + half_width) - score) / TRANSITION_WIDTH

    # 截断到 [0, 1]
    normalized = max(0.0, min(1.0, normalized))

    # 转换为两个方向的权重：normalized=0 → 纯A2V；normalized=1 → 纯V2A
    w_a2v = 1.0 - normalized
    w_v2a = normalized

    return w_a2v, w_v2a, score

# ===================== 7. 核心推理（双向量加权混合注入） =====================
def ask_qwen(conversation, question_text):
    # 计算连续权重
    w_a2v, w_v2a, raw_score = calc_hallucination_weights(question_text)

    # 屏蔽库输出
    with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
        malleable_model = MalleableModel(model=model, tokenizer=tokenizer)

        # 同时注入两路向量，分别作用在原配置层数，强度按权重缩放
        # 1. 视频幻觉向量（A2V）注入到对应层
        malleable_model.steer(
            behavior_vector=vec_a2v,
            behavior_layer_ids=LAYERS_A2V,
            behavior_vector_strength=STRENGTH_A2V * w_a2v,
        )
        # 2. 音频幻觉向量（V2A）注入到对应层
        malleable_model.steer(
            behavior_vector=vec_v2a,
            behavior_layer_ids=LAYERS_V2A,
            behavior_vector_strength=STRENGTH_V2A * w_v2a,
        )

        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        processor_kwargs = {"text": text, "return_tensors": "pt", "padding": True, "use_audio_in_video": USE_AUDIO_IN_VIDEO}
        
        if torch.is_tensor(audios) and audios.numel() > 0:
            processor_kwargs["audio"] = audios.detach().cpu().numpy()
        if isinstance(images, list) and len(images) > 0:
            processor_kwargs["images"] = images
        if isinstance(videos, list) and len(videos) > 0:
            processor_kwargs["videos"] = videos

        inputs = processor(**processor_kwargs).to(DEVICE).to(torch.float16)
        with torch.no_grad():
            text_ids = malleable_model.model.generate(
                **inputs, max_new_tokens=10, do_sample=False, num_beams=1,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
            )
    
    answer = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
    answer = answer.split("assistant")[-1].strip() if "assistant" in answer else answer
    return answer, w_a2v, w_v2a, raw_score

# ===================== 8. 主程序 =====================
if __name__ == "__main__":
    print("\n===== 开始测试（条件向量加权混合转向，无测试集调参） =====")
    all_results = []

    for idx, q_dict in enumerate(tqdm(multimodal_questions, desc="推理进度", ncols=80), 1):
        question_text = q_dict["text"]
        task = q_dict["task"]
        gt_label = q_dict["label"]

        conversation = build_conversation(q_dict)
        raw_answer, w_a2v, w_v2a, cond_score = ask_qwen(conversation, question_text)

        # 结果解析
        if task in ["Video-driven Audio Hallucination", "Audio-driven Video Hallucination", "AV Matching"]:
            pred_answer = "Yes" if "yes" in raw_answer.lower() else "No" if "no" in raw_answer.lower() else "Unknown"
        else:
            pred_answer = raw_answer

        # 前N条打印详情
        if idx <= PRINT_MAX:
            print(f"\n【{idx} | {task}】")
            print(f"条件向量得分: {cond_score:.4f} | A2V权重: {w_a2v:.3f} | V2A权重: {w_v2a:.3f}")
            print(f"Q: {question_text}")
            print(f"A: {pred_answer}")

        all_results.append({
            "video_id": q_dict["video_id"],
            "task": task,
            "text": question_text,
            "label": gt_label,
            "model_answer": pred_answer,
            "condition_score": round(cond_score, 6),
            "weight_a2v": round(w_a2v, 4),
            "weight_v2a": round(w_v2a, 4)
        })

    # 保存结果
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 测试完成！结果已保存至：{OUTPUT_JSON}")
    torch.cuda.empty_cache()
    gc.collect()