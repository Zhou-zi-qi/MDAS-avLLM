import os
import torch
import numpy as np
import json
import warnings
from tqdm import tqdm
from qwen_omni_utils import process_mm_info
warnings.filterwarnings("ignore")

# ====================== 1. 基础配置 ======================
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_PATH = "/home/srt32/activation-steering/Qwen2.5-Omni-3B"
USE_AUDIO_IN_VIDEO = True

# ====================== 2. 数据集配置 ======================
AVHBENCH_VIDEO_DIR = "/home/srt32/activation-steering/avhbench_data/videos"
AVHBENCH_QA_JSON = "/home/srt32/activation-steering/avhbench_data/QA.json"
OUTPUT_JSON = "/home/srt32/activation-steering/qwen_avhbench_result.json"
SAMPLE_NUM = 2000
PRINT_MAX = 10     # 仅打印前10个样本

# ====================== 3. 加载数据 ======================
def load_avhbench_questions():
    with open(AVHBENCH_QA_JSON, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    if SAMPLE_NUM:
        qa_data = qa_data[:SAMPLE_NUM]
    
    multimodal_questions = []
    for sample in qa_data:
        video_id = sample["video_id"]
        video_path = os.path.join(AVHBENCH_VIDEO_DIR, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            continue
        
        question = {
            "text": sample["text"],
            "multimodal": [{"type": "video", "video": video_path}],
            "video_id": video_id,
            "task": sample["task"],
            "label": sample["label"]
        }
        multimodal_questions.append(question)
    
    print(f"✅ 加载 AVHBench 有效测试样本：{len(multimodal_questions)}")
    return multimodal_questions

multimodal_questions = load_avhbench_questions()

# ====================== 4. 加载模型（✅ 修复类名错误） ======================
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto"
)
model.disable_talker()
model.eval()

processor = Qwen2_5OmniProcessor.from_pretrained(
    MODEL_PATH,
    use_fast=False
)

# ====================== 5. 核心函数（无修改） ======================
def build_conversation(question_dict):
    user_content = [{"type": "text", "text": question_dict["text"]}]
    for modal in question_dict["multimodal"]:
        modal_type = modal["type"]
        modal_path = modal[modal_type]
        user_content.append({
            "type": modal_type,
            modal_type: modal_path
        })
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": user_content}
    ]
    return conversation

def ask_qwen(conversation):
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    processor_kwargs = {
        "text": text,
        "return_tensors": "pt",
        "padding": True,
        "use_audio_in_video": USE_AUDIO_IN_VIDEO
    }
    
    if torch.is_tensor(audios) and audios.numel() > 0:
        audio_np = audios.detach().cpu().numpy()
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)
        elif audio_np.ndim > 2:
            audio_np = audio_np.squeeze()
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
        processor_kwargs["audio"] = audio_np
    elif isinstance(audios, list) and len(audios) > 0:
        processor_kwargs["audio"] = audios
    
    if isinstance(images, list) and len(images) > 0:
        processor_kwargs["images"] = images
    
    if isinstance(videos, list) and len(videos) > 0:
        processor_kwargs["videos"] = videos
    
    inputs = processor(**processor_kwargs)
    inputs = inputs.to(model.device).to(model.dtype)
    
    with torch.no_grad():
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            max_new_tokens=10,
            do_sample=False,
            num_beams=1,
            temperature=0.01,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    
    answer = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    answer = answer.split("assistant")[-1].strip() if "assistant" in answer else answer
    return answer

# ====================== 6. 主程序（进度条+仅打印前10） ======================
if __name__ == "__main__":
    print("===== Qwen2.5-Omni AVHBench 幻觉测试 =====")
    print(f"总计测试样本：{len(multimodal_questions)}")
    print(f"✅ 仅展示前 {PRINT_MAX} 个样本详细信息")
    print("-" * 80)
    
    all_results = []

    for idx, q_dict in enumerate(tqdm(multimodal_questions, desc="推理进度"), 1):
        question_text = q_dict["text"]
        video_id = q_dict["video_id"]
        task = q_dict["task"]
        gt_label = q_dict["label"]
        
        # 模型推理
        conversation = build_conversation(q_dict)
        raw_answer = ask_qwen(conversation)

        # 按任务分类处理答案
        if task in ["Video-driven Audio Hallucination", "Audio-driven Video Hallucination", "AV Matching"]:
            pred_answer = "Yes" if "yes" in raw_answer.lower() else "No" if "no" in raw_answer.lower() else "Unknown"
        elif task == "AV Captioning":
            pred_answer = raw_answer
        else:
            pred_answer = raw_answer

        # 仅打印前10个样本
        if idx <= PRINT_MAX:
            print(f"\n【{idx} | 任务：{task} | Video：{video_id}】")
            print(f"问题：{question_text}")
            if task in ["Video-driven Audio Hallucination", "Audio-driven Video Hallucination", "AV Matching"]:
                print(f"模型预测：{pred_answer}")
            else:
                print(f"模型输出：{pred_answer[:150]}..." if len(pred_answer) > 150 else f"模型输出：{pred_answer}")
            print("-" * 5)
        
        # 保存结果
        all_results.append({
            "video_id": video_id,
            "task": task,
            "text": question_text,
            "label": gt_label,
            "model_answer": pred_answer
        })
    
    # 保存文件
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 测试完成！结果已保存：{OUTPUT_JSON}")