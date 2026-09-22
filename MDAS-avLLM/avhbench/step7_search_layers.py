import os
import sys
import json
import gc
import multiprocessing as mp
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===================== 🔥 寻优核心配置（只改这里） =====================
# 指定使用哪几张显卡，比如 [0,1,2,3] 就是4卡并行
GPU_IDS = [0,1,2,3,5,6]

LOCAL_MODEL_PATH = "/home/srt32/activation-steering/Qwen2.5-Omni-3B"
AVHBENCH_VIDEO_DIR = "/home/srt32/activation-steering/avhbench_data/videos"
AVHBENCH_QA_JSON = "/home/srt32/activation-steering/avhbench_data/QA.json"
RESULT_DIR = "/home/srt32/activation-steering/layer_search_video_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 待测试的向量
HALLUCINATION_VECTOR = "hallucination_reduction_vector_dvd_audio.svec"
# 固定强度
FIXED_STRENGTH = 1.5
# 测试样本数
SAMPLE_NUM = 2000
# 目标任务
TARGET_TASK = "Video-driven Audio Hallucination"

# 所有待测试的层组合
LAYER_CANDIDATES = [
    [18, 19, 20, 21],
    [19, 20, 21, 22],
    [20, 21, 22, 23],
    [21, 22, 23, 24],
    [22, 23, 24, 25],
    [23, 24, 25, 26],
    [24, 25, 26, 27],
    [25, 26, 27, 28],
]
# =========================================================================

USE_AUDIO_IN_VIDEO = True


# ===================== 单卡工作进程函数 =====================
def worker(gpu_id, layer_batch, result_dir, return_dict):
    # 子进程独占指定显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    import warnings
    from contextlib import redirect_stdout, redirect_stderr
    from tqdm import tqdm
    from qwen_omni_utils import process_mm_info

    warnings.filterwarnings("ignore")
    import transformers
    transformers.logging.set_verbosity_error()
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------- 模型兼容补丁 ----------
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

    # ---------- 加载模型 ----------
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from activation_steering import MalleableModel, SteeringVector

    print(f"[GPU {gpu_id}] 正在加载模型...")
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
    print(f"[GPU {gpu_id}] 模型加载完成")

    # ---------- 加载向量 ----------
    steering_vector = SteeringVector.load(HALLUCINATION_VECTOR)
    print(f"[GPU {gpu_id}] 向量加载完成")

    # ---------- 加载测试集 ----------
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
        return multimodal_questions

    multimodal_questions = load_avhbench_questions()
    print(f"[GPU {gpu_id}] 测试集加载完成：{len(multimodal_questions)} 条，待处理 {len(layer_batch)} 组")

    # ---------- 评估函数 ----------
    def evaluate_result(gt_path, pred_path):
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)

        TASK_LIST = [
            "Audio-driven Video Hallucination",
            "Video-driven Audio Hallucination",
            "AV Matching"
        ]
        SKIP_TASKS = ["AV Captioning"]
        
        task_metrics = defaultdict(lambda: {"tp":0, "fp":0, "tn":0, "fn":0, "total":0, "correct":0})

        for gt, pred in zip(gt_data, pred_data):
            task = gt["task"]
            if task in SKIP_TASKS or task not in TASK_LIST:
                continue
            gt_ans = gt["label"].strip().upper()
            pred_ans = pred["model_answer"].strip().upper()

            if gt_ans == "YES":
                if pred_ans == "YES":
                    task_metrics[task]["tp"] += 1
                    task_metrics[task]["correct"] += 1
                else:
                    task_metrics[task]["fn"] += 1
            else:
                if pred_ans == "YES":
                    task_metrics[task]["fp"] += 1
                else:
                    task_metrics[task]["tn"] += 1
                    task_metrics[task]["correct"] += 1
            task_metrics[task]["total"] += 1

        result = {}
        total_correct = 0
        total_samples = 0
        for task in TASK_LIST:
            m = task_metrics[task]
            if m["total"] == 0:
                continue
            acc = m["correct"] / m["total"]
            precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"]+m["fp"]) > 0 else 0.0
            recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"]+m["fn"]) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0.0
            result[task] = {
                "acc": acc, "precision": precision, "recall": recall, "f1": f1,
                "total": m["total"], "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"]
            }
            total_correct += m["correct"]
            total_samples += m["total"]
        
        result["overall"] = {
            "acc": total_correct / total_samples if total_samples > 0 else 0,
            "total": total_samples
        }
        return result

    # ---------- 推理函数 ----------
    def build_conversation(question_dict):
        user_content = [{"type": "text", "text": question_dict["text"]}]
        for modal in question_dict["multimodal"]:
            user_content.append({"type": modal["type"], modal["type"]: modal[modal["type"]]})
        return [
            {
                "role": "system",
                "content": [{"type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                }]
            },
            {"role": "user", "content": user_content}
        ]

    def run_inference(layers, strength, output_path):
        all_results = []
        for q_dict in tqdm(multimodal_questions, desc=f"[GPU{gpu_id}] 层{layers}", ncols=80):
            conversation = build_conversation(q_dict)
            
            with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
                malleable_model = MalleableModel(model=model, tokenizer=tokenizer)
                malleable_model.steer(
                    behavior_vector=steering_vector,
                    behavior_layer_ids=layers,
                    behavior_vector_strength=strength,
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
            raw_answer = answer.split("assistant")[-1].strip() if "assistant" in answer else answer
            
            task = q_dict["task"]
            if task in ["Video-driven Audio Hallucination", "Audio-driven Video Hallucination", "AV Matching"]:
                pred_answer = "Yes" if "yes" in raw_answer.lower() else "No" if "no" in raw_answer.lower() else "Unknown"
            else:
                pred_answer = raw_answer

            all_results.append({
                "video_id": q_dict["video_id"], "task": task, "text": q_dict["text"],
                "label": q_dict["label"], "model_answer": pred_answer
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        return output_path

    # ---------- 处理分配给自己的层组合 ----------
    batch_results = []
    for layers in layer_batch:
        layer_str = "_".join(map(str, layers))
        output_file = os.path.join(result_dir, f"result_layers_{layer_str}.json")
        
        run_inference(layers, FIXED_STRENGTH, output_file)
        metrics = evaluate_result(AVHBENCH_QA_JSON, output_file)
        
        target_metric = metrics[TARGET_TASK]
        record = {
            "layers": layers,
            "target_acc": target_metric["acc"],
            "target_f1": target_metric["f1"],
            "target_precision": target_metric["precision"],
            "target_recall": target_metric["recall"],
            "overall_acc": metrics["overall"]["acc"],
            "result_file": output_file
        }
        batch_results.append(record)
        print(f"[GPU {gpu_id}] 完成 {layers} | 目标Acc: {target_metric['acc']:.4f} | 整体Acc: {metrics['overall']['acc']:.4f}")

    # 清理显存
    del model, steering_vector
    torch.cuda.empty_cache()
    gc.collect()

    # 返回结果
    return_dict[gpu_id] = batch_results


# ===================== 主进程：任务拆分 + 汇总 =====================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    print(f"{'='*80}")
    print(f"🚀 多卡并行层寻优启动")
    print(f"使用显卡：{GPU_IDS} | 待测试组合：{len(LAYER_CANDIDATES)} 组")
    print(f"目标任务：{TARGET_TASK} | 固定强度：{FIXED_STRENGTH}")
    print(f"{'='*80}\n")

    # 1. 均匀拆分任务到每张卡
    num_gpus = len(GPU_IDS)
    batches = [[] for _ in range(num_gpus)]
    for i, layers in enumerate(LAYER_CANDIDATES):
        batches[i % num_gpus].append(layers)

    # 2. 启动多进程
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for gpu_id, batch in zip(GPU_IDS, batches):
        if not batch:
            continue
        p = mp.Process(target=worker, args=(gpu_id, batch, RESULT_DIR, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 3. 合并所有结果
    all_results = []
    for res_list in return_dict.values():
        all_results.extend(res_list)

    # 4. 排序输出
    all_results_sorted = sorted(all_results, key=lambda x: x["target_acc"], reverse=True)
    best = all_results_sorted[0]

    print(f"\n\n{'='*80}")
    print(f"🏆 全部测试完成！按【{TARGET_TASK} 准确率】排序结果：")
    print(f"{'='*80}")
    print(f"{'排名':<4} {'层组合':<18} {'目标任务Acc':<12} {'目标任务F1':<12} {'整体Acc':<12}")
    print("-" * 60)
    for rank, item in enumerate(all_results_sorted, 1):
        layer_str = str(item["layers"])
        print(f"{rank:<4} {layer_str:<18} {item['target_acc']:<12.4f} {item['target_f1']:<12.4f} {item['overall_acc']:<12.4f}")

    print(f"\n🥇 最优层组合：{best['layers']}")
    print(f"   目标任务准确率：{best['target_acc']:.4f}")
    print(f"   目标任务F1：{best['target_f1']:.4f}")
    print(f"   整体准确率：{best['overall_acc']:.4f}")
    print(f"   结果文件：{best['result_file']}")
    print(f"{'='*80}")

    # 保存报告
    report_path = os.path.join(RESULT_DIR, "layer_search_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_results_sorted, f, indent=2, ensure_ascii=False)
    print(f"\n📄 完整寻优报告已保存：{report_path}")