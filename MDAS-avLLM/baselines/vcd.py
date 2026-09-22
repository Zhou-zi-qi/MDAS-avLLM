import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import json
import warnings
import subprocess
from tqdm import tqdm
from qwen_omni_utils import process_mm_info
warnings.filterwarnings("ignore")

# ====================== 1. 基础配置 ======================
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_PATH = "/home/srt32/activation-steering/Qwen2.5-Omni-3B"
USE_AUDIO_IN_VIDEO = True

# ====================== 2. 数据集 / 输出 ======================
AVHBENCH_VIDEO_DIR = "/home/srt32/activation-steering/avhbench_data/videos"
AVHBENCH_QA_JSON = "/home/srt32/activation-steering/avhbench_data/QA.json"
TASK_DIR = "/home/srt32/activation-steering/qwen_AVHDVDbench_task"
OUTPUT_JSON = os.path.join(TASK_DIR, "qwen_avhbench_vcd_result.json")
NOISY_DIR = os.path.join(TASK_DIR, "avhbench_vcd_noisy")
SAMPLE_NUM = 2000

# VCD alpha 扫描列表（跑完一次前向后，这里全部后处理评估）
ALPHA_LIST = [0.0, 0.05, 0.1, 0.2]

# 加噪强度（ffmpeg noise filter spatial/temporal strength）
NOISE_VIDEO_STRENGTH = 20      # ffmpeg noise=alls=...
NOISE_AUDIO_AMPLITUDE = 0.05   # white noise amplitude mixed in

os.makedirs(NOISY_DIR, exist_ok=True)

# ====================== 3. 数据 ======================
def load_questions():
    with open(AVHBENCH_QA_JSON, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    if SAMPLE_NUM:
        qa_data = qa_data[:SAMPLE_NUM]
    out = []
    for s in qa_data:
        vp = os.path.join(AVHBENCH_VIDEO_DIR, f"{s['video_id']}.mp4")
        if not os.path.exists(vp):
            continue
        out.append({
            "text": s["text"],
            "multimodal": [{"type": "video", "video": vp}],
            "video_id": s["video_id"],
            "task": s["task"],
            "label": s["label"],
            "video_path": vp,
        })
    print(f"loaded {len(out)} samples")
    return out

# ====================== 4. 加噪视频（ffmpeg） ======================
def has_audio_stream(path):
    try:
        out = subprocess.check_output([
            "ffprobe","-v","error","-select_streams","a","-show_entries","stream=codec_type",
            "-of","csv=p=0", path
        ], text=True).strip()
        return out != ""
    except Exception:
        return False

def get_duration(path):
    try:
        return float(subprocess.check_output([
            "ffprobe","-v","error","-show_entries","format=duration",
            "-of","default=noprint_wrappers=1:nokey=1", path
        ], text=True).strip())
    except Exception:
        return 10.0

def make_noisy_video(video_path, noisy_path):
    if os.path.exists(noisy_path) and os.path.getsize(noisy_path) > 0:
        return
    if has_audio_stream(video_path):
        dur = get_duration(video_path)
        fc = (f"[0:v]noise=alls={NOISE_VIDEO_STRENGTH}:allf=t+u[v];"
              f"[0:a]volume=0.7[a0];"
              f"anoisesrc=color=white:amplitude={NOISE_AUDIO_AMPLITUDE}:duration={dur}[n];"
              f"[a0][n]amix=inputs=2:duration=first:dropout_transition=0[a]")
        cmd = ["ffmpeg","-y","-i",video_path,"-filter_complex",fc,
               "-map","[v]","-map","[a]",
               "-c:v","libx264","-preset","veryfast","-crf","24",
               "-c:a","aac","-ar","16000", noisy_path]
    else:
        cmd = ["ffmpeg","-y","-i",video_path,
               "-vf", f"noise=alls={NOISE_VIDEO_STRENGTH}:allf=t+u",
               "-an","-c:v","libx264","-preset","veryfast","-crf","24",
               noisy_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

# ====================== 5. 模型 ======================
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype="auto", device_map="auto")
model.disable_talker()
model.eval()
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH, use_fast=False)
tokenizer = processor.tokenizer

# Yes / No 候选 token（首 token 决策）
def _last_ids(text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [ids[-1]] if ids else []

YES_IDS = list(set(_last_ids("Yes") + _last_ids(" Yes")))
NO_IDS = list(set(_last_ids("No") + _last_ids(" No")))
print("YES tokens:", YES_IDS, "NO tokens:", NO_IDS)

def build_conversation(question_dict, video_path=None):
    vp = video_path or question_dict["video_path"]
    user_content = [{"type": "text", "text": question_dict["text"]},
                    {"type": "video", "video": vp}]
    return [
        {"role":"system","content":[{"type":"text","text":
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role":"user","content":user_content},
    ]

def build_mm_tensor(conversation):
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    pk = {"text": text, "return_tensors":"pt", "padding":True, "use_audio_in_video":USE_AUDIO_IN_VIDEO}
    if torch.is_tensor(audios) and audios.numel() > 0:
        a = audios.detach().cpu().numpy()
        if a.ndim == 1: a = a.reshape(1,-1)
        elif a.ndim > 2:
            a = a.squeeze()
            if a.ndim == 1: a = a.reshape(1,-1)
        pk["audio"] = a
    if isinstance(images, list) and len(images): pk["images"] = images
    if isinstance(videos, list) and len(videos): pk["videos"] = videos
    return processor(**pk).to(model.device).to(model.dtype)

def clean_generate(tensor_in):
    """普通贪心生成（和原 Unsteered 完全一致）"""
    with torch.no_grad():
        out = model.generate(
            **tensor_in,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            max_new_tokens=10,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    gen = out[0, tensor_in["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()

def first_token_logits(conversation):
    inputs = build_mm_tensor(conversation)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            max_new_tokens=1,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    return out.scores[0][0].float()

def logprob_of(logits, ids):
    lp = torch.log_softmax(logits, dim=-1)
    sel = torch.stack([lp[i] for i in ids])
    return torch.logsumexp(sel, dim=0).item()

# ====================== 6. 官方指标（直接打印） ======================
def official_metrics(records):
    TASK_LIST = ["Audio-driven Video Hallucination","Video-driven Audio Hallucination","AV Matching"]
    from collections import defaultdict
    M = defaultdict(lambda: {"tp":0,"fp":0,"tn":0,"fn":0,"correct":0,"total":0})
    for r in records:
        task = r["task"]
        if task not in TASK_LIST: continue
        gt = r["label"].strip().upper()
        ans = r["model_answer"].strip().lower()
        if "yes" in ans: pred = "YES"
        elif "no" in ans: pred = "NO"
        else: pred = "UNKNOWN"
        m = M[task]
        m["total"] += 1
        if gt == "YES":
            if pred == "YES": m["tp"] += 1; m["correct"] += 1
            else: m["fn"] += 1
        else:
            if pred == "YES": m["fp"] += 1
            else: m["tn"] += 1; m["correct"] += 1
    print("="*70)
    all_c, all_t = 0, 0
    for task in TASK_LIST:
        m = M[task]
        if m["total"] == 0: continue
        acc = m["correct"]/m["total"]
        prec = m["tp"]/(m["tp"]+m["fp"]) if (m["tp"]+m["fp"])>0 else 0.0
        rec = m["tp"]/(m["tp"]+m["fn"]) if (m["tp"]+m["fn"])>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        all_c += m["correct"]; all_t += m["total"]
        print(f"{task}: n={m['total']} Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    if all_t:
        print(f"OVERALL Acc={all_c/all_t:.4f} (n={all_t})")
    # AVH + VAH pooled: F = 1 - Precision = FP / (TP+FP)
    hallu_tasks = ["Audio-driven Video Hallucination", "Video-driven Audio Hallucination"]
    tp = sum(M[t]["tp"] for t in hallu_tasks)
    fp = sum(M[t]["fp"] for t in hallu_tasks)
    pooled_prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    print(f"AVH+VAH pooled: n={sum(M[t]['total'] for t in hallu_tasks)} "
          f"Prec={pooled_prec:.4f}  F=1-Prec={1-pooled_prec:.4f}")
    print("="*70)

# ====================== 7. 主流程 ======================
if __name__ == "__main__":
    questions = load_questions()
    done = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            old = json.load(open(OUTPUT_JSON, encoding="utf-8"))
            for r in old:
                done[r["key"]] = r
            print(f"resume: {len(done)} already done")
        except Exception:
            done = {}

    for idx, q in enumerate(tqdm(questions, desc="VCD inference"), 1):
        vid = q["video_id"]
        key = f"{idx}_{vid}_{q['text'][:40]}"
        if key in done: continue
        # AV Captioning 任务官方不评估，直接记 Unknown
        if q["task"] == "AV Captioning":
            rec = {"key":key,"video_id":vid,"task":q["task"],"text":q["text"],
                   "label":q["label"]}
            for a in ALPHA_LIST: rec[f"ans_{a}"] = "Unknown"
            done[key] = rec
            continue

        clean_in = build_mm_tensor(build_conversation(q))
        # alpha=0: 完整贪心生成，和论文 Unsteered 完全一致
        ans0 = clean_generate(clean_in)

        # 首 token 对比 logits
        lc = first_token_logits(build_conversation(q))
        ly_c = logprob_of(lc, YES_IDS); ln_c = logprob_of(lc, NO_IDS)
        # 加噪
        noisy_path = os.path.join(NOISY_DIR, f"{vid}.mp4")
        try:
            make_noisy_video(q["video_path"], noisy_path)
            ln_lg = first_token_logits(build_conversation(q, video_path=noisy_path))
            ly_n = logprob_of(ln_lg, YES_IDS); ln_n = logprob_of(ln_lg, NO_IDS)
        except Exception as e:
            print(f"[warn] noisy failed for {vid}: {e}")
            ly_n, ln_n = 0.0, 0.0

        rec = {"key":key,"video_id":vid,"task":q["task"],"text":q["text"],"label":q["label"]}
        for a in ALPHA_LIST:
            if abs(a) < 1e-9:
                rec[f"ans_{a}"] = ans0
            else:
                sy = ly_c - a * ly_n
                sn = ln_c - a * ln_n
                rec[f"ans_{a}"] = "Yes" if sy > sn else "No"
        done[key] = rec
        # 增量保存
        json.dump(list(done.values()), open(OUTPUT_JSON,"w",encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        del clean_in
        torch.cuda.empty_cache()

    # 用不同 alpha 评估
    records = list(done.values())
    print("\n\n##### VCD alpha sweep #####")
    for alpha in ALPHA_LIST:
        print(f"\n----- alpha = {alpha} -----")
        eval_recs = []
        for r in records:
            ans = r.get(f"ans_{alpha}", "Unknown")
            eval_recs.append({"video_id":r["video_id"],"task":r["task"],
                              "label":r["label"],"model_answer":ans})
        official_metrics(eval_recs)
    print(f"\nraw scores saved to {OUTPUT_JSON}")
