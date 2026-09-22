# -*- coding: utf-8 -*-
"""
AVCD baseline on AVHBench for Qwen2.5-Omni-3B.
Reference: Jung et al., "AVCD: Mitigating Hallucinations in Audio-Visual LLMs
through Contrastive Decoding", NeurIPS 2025 (arXiv:2505.20862).

Trimodal contrastive decoding (language dominant; video+audio less dominant):
  L_AVCD = (2+2a)*L_full + 1*L_vmask + 1*L_amask - 2a*L_both
We use first-token logits to decide Yes/No (same protocol as the VCD baseline).
Alpha sweep: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0].
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
sys.path.append("/home/srt32/activation-steering")

import torch
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

MODEL_PATH = "/home/srt32/activation-steering/Qwen2.5-Omni-3B"
QA_PATH = "/home/srt32/activation-steering/avhbench_data/QA.json"
VIDEO_DIR = "/home/srt32/activation-steering/avhbench_data/videos"
OUTPUT_JSON = "/home/srt32/activation-steering/qwen_AVHDVDbench_task/qwen_avhbench_avcd_result.json"
USE_AUDIO_IN_VIDEO = True
ALPHA_LIST = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]

YES_IDS = [9454, 7414]
NO_IDS  = [2753, 2308]

print("loading model...")
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH, use_fast=False)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH, device_map="cuda", torch_dtype=torch.bfloat16,
)
model.eval()
model.disable_talker()

# ---------------- data ----------------
def load_questions():
    data = json.load(open(QA_PATH, encoding="utf-8"))[:2000]
    out = []
    for s in data:
        vp = os.path.join(VIDEO_DIR, s["video_id"]+".mp4")
        if not os.path.exists(vp): continue
        out.append({
            "video_id": s["video_id"], "video_path": vp,
            "task": s["task"], "text": s["text"], "label": s["label"],
        })
    print(f"loaded {len(out)} samples")
    return out

def build_conversation(q, include_video=True):
    vp = q["video_path"]
    content = [{"type":"text","text":q["text"]}]
    if include_video:
        content.append({"type":"video","video":vp})
    return [
        {"role":"system","content":[{"type":"text","text":
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role":"user","content":content},
    ]

def build_mm_tensor(conversation, use_audio=True):
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    pk = {"text": text, "return_tensors":"pt", "padding":True,
          "use_audio_in_video":use_audio}
    if torch.is_tensor(audios) and audios.numel()>0:
        a = audios.detach().cpu().numpy()
        if a.ndim==1: a = a.reshape(1,-1)
        elif a.ndim>2:
            a = a.squeeze()
            if a.ndim==1: a = a.reshape(1,-1)
        pk["audio"] = a
    if isinstance(images, list) and len(images): pk["images"] = images
    if isinstance(videos, list) and len(videos): pk["videos"] = videos
    t = processor(**pk).to(model.device).to(model.dtype)
    return t

def first_token_logits(tensor_in):
    with torch.no_grad():
        out = model.generate(
            **tensor_in, use_audio_in_video=USE_AUDIO_IN_VIDEO,
            max_new_tokens=1, do_sample=False, num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            return_dict_in_generate=True, output_scores=True,
        )
    return out.scores[0][0].float()

def clean_generate(tensor_in):
    with torch.no_grad():
        out = model.generate(
            **tensor_in, use_audio_in_video=USE_AUDIO_IN_VIDEO,
            max_new_tokens=10, do_sample=False, num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    return processor.tokenizer.decode(out[0][tensor_in["input_ids"].shape[1]:], skip_special_tokens=True)

def logprob_of(logits, ids):
    lp = torch.log_softmax(logits, dim=-1)
    return torch.logsumexp(torch.stack([lp[i] for i in ids]), dim=0).item()

# ---------------- metrics ----------------
def official_metrics(records):
    from collections import defaultdict
    TASK_LIST = ["Audio-driven Video Hallucination","Video-driven Audio Hallucination","AV Matching"]
    M = defaultdict(lambda: {"tp":0,"fp":0,"tn":0,"fn":0,"correct":0,"total":0})
    for r in records:
        task = r["task"]
        if task not in TASK_LIST: continue
        gt = r["label"].strip().upper()
        ans = r["model_answer"].strip().lower()
        pred = "YES" if "yes" in ans else ("NO" if "no" in ans else "UNKNOWN")
        m = M[task]; m["total"] += 1
        if gt=="YES":
            if pred=="YES": m["tp"]+=1; m["correct"]+=1
            else: m["fn"]+=1
        else:
            if pred=="YES": m["fp"]+=1
            else: m["tn"]+=1; m["correct"]+=1
    print("="*70)
    all_c, all_t = 0,0
    for task in TASK_LIST:
        m = M[task]
        if m["total"]==0: continue
        acc = m["correct"]/m["total"]
        prec = m["tp"]/(m["tp"]+m["fp"]) if (m["tp"]+m["fp"])>0 else 0
        rec = m["tp"]/(m["tp"]+m["fn"]) if (m["tp"]+m["fn"])>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        all_c += m["correct"]; all_t += m["total"]
        print(f"{task}: n={m['total']} Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    if all_t: print(f"OVERALL Acc={all_c/all_t:.4f} (n={all_t})")
    ht = ["Audio-driven Video Hallucination","Video-driven Audio Hallucination"]
    tp = sum(M[t]["tp"] for t in ht); fp = sum(M[t]["fp"] for t in ht)
    pp = tp/(tp+fp) if (tp+fp)>0 else 0
    print(f"AVH+VAH pooled: n={sum(M[t]['total'] for t in ht)} Prec={pp:.4f} F=1-Prec={1-pp:.4f}")
    print("="*70)

if __name__=="__main__":
    questions = load_questions()
    done = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            for r in json.load(open(OUTPUT_JSON, encoding="utf-8")): done[r["key"]] = r
            print(f"resume: {len(done)} done")
        except Exception: done = {}

    for idx, q in enumerate(tqdm(questions, desc="AVCD"), 1):
        vid = q["video_id"]
        key = f"{idx}_{vid}_{q['text'][:40]}"
        if key in done: continue
        if q["task"]=="AV Captioning":
            rec = {"key":key,"video_id":vid,"task":q["task"],"text":q["text"],"label":q["label"]}
            for a in ALPHA_LIST: rec[f"ans_{a}"]="Unknown"
            done[key]=rec; continue

        conv_full = build_conversation(q, include_video=True)
        conv_no_v = build_conversation(q, include_video=False)
        t_full = build_mm_tensor(conv_full, use_audio=True)
        ans0 = clean_generate(t_full)
        L_full = first_token_logits(t_full)
        L_vm   = first_token_logits(build_mm_tensor(conv_no_v, use_audio=False))
        L_am   = first_token_logits(build_mm_tensor(conv_full, use_audio=False))
        L_bm   = first_token_logits(build_mm_tensor(conv_no_v, use_audio=False))

        rec = {"key":key,"video_id":vid,"task":q["task"],"text":q["text"],"label":q["label"]}
        for a in ALPHA_LIST:
            if abs(a) < 1e-9:
                rec[f"ans_{a}"] = ans0
                continue
            lp_full = torch.log_softmax(L_full, dim=-1)
            lp_vm   = torch.log_softmax(L_vm, dim=-1)
            lp_am   = torch.log_softmax(L_am, dim=-1)
            lp_bm   = torch.log_softmax(L_bm, dim=-1)
            combined = (2+2*a)*lp_full + 1.0*lp_vm + 1.0*lp_am - 2*a*lp_bm
            sy = combined[YES_IDS].logsumexp(0).item()
            sn = combined[NO_IDS].logsumexp(0).item()
            rec[f"ans_{a}"] = "Yes" if sy>sn else "No"
        done[key]=rec
        json.dump(list(done.values()), open(OUTPUT_JSON,"w",encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        torch.cuda.empty_cache()

    records = list(done.values())
    print("\n\n##### AVCD alpha sweep #####")
    for a in ALPHA_LIST:
        print(f"\n----- alpha = {a} -----")
        er = [{"video_id":r["video_id"],"task":r["task"],"label":r["label"],
               "model_answer":r.get(f"ans_{a}","Unknown")} for r in records]
        official_metrics(er)
    print(f"\nsaved to {OUTPUT_JSON}")
