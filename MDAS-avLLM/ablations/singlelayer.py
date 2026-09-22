"""
Single-layer single-vector AVHBench eval, following bestlayer_search.py structure.
Usage:
  python rh_qwen_avhbench_singlelayer.py --vector visual
  python rh_qwen_avhbench_singlelayer.py --vector audio
"""
import os, sys, json, gc, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===================== config =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

LOCAL_MODEL_PATH = "/home/srt32/activation-steering/Qwen2.5-Omni-3B"
AVHBENCH_VIDEO_DIR = "/home/srt32/activation-steering/avhbench_data/videos"
AVHBENCH_QA_JSON = "/home/srt32/activation-steering/avhbench_data/QA.json"
BASE_DIR = "/home/srt32/activation-steering/qwen_AVHDVDbench_task"

VECTOR_PATH = {
    "visual": f"{BASE_DIR}/hallucination_reduction_vector_avhbench_videonew2.svec",
    "audio":  f"{BASE_DIR}/hallucination_reduction_vector_avhbench_audionew2.svec",
}
FIXED_STRENGTH = 1.5
FIXED_LAYERS = [25]
SAMPLE_NUM = 2000
USE_AUDIO_IN_VIDEO = True
PRINT_MAX = 3

import torch
import warnings
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm
from collections import defaultdict
warnings.filterwarnings("ignore")
import transformers
transformers.logging.set_verbosity_error()
import logging
logging.getLogger().setLevel(logging.CRITICAL)

from qwen_omni_utils import process_mm_info

# ---------- model compat patch ----------
from activation_steering.malleable_model import LeashLayer
_orig_init = LeashLayer.__init__
def _patched(self, layer, *a, **k):
    _orig_init(self, layer, *a, **k)
    if hasattr(layer, 'attention_type'): self.attention_type = layer.attention_type
    if hasattr(layer, 'layer_idx'): self.layer_idx = layer.layer_idx
LeashLayer.__init__ = _patched

from activation_steering import malleable_model as mm
from activation_steering import steering_vector as sv
def _gll(model):
    if "Qwen2_5OmniForConditionalGeneration" in str(type(model)):
        return model.thinker.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(str(type(model)))
mm.get_model_layer_list = _gll
sv.get_model_layer_list = _gll

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from activation_steering import MalleableModel, SteeringVector

torch.cuda.empty_cache(); gc.collect()
print("loading model ...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH, dtype=torch.float16, device_map={"":0},
    low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True)

def _fwd(input_ids, attention_mask=None, **kw):
    kw.pop('output_hidden_states', None)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = model.thinker.model(input_ids=input_ids, attention_mask=attention_mask,
                                  return_dict=True, output_hidden_states=True, **kw)
    hw = model.thinker.lm_head.weight.to(torch.float16)
    setattr(out, 'logits', out.last_hidden_state @ hw.t())
    setattr(out, 'hidden_states', out.hidden_states)
    return out
model.forward = _fwd
model.layers = model.thinker.model.layers
model.config.num_hidden_layers = len(model.layers)
model.config.n_layer = len(model.layers)
model.config.hidden_size = 2048
model.config.model_type = "qwen2_5_omni"
model.disable_talker(); model.eval()

processor = Qwen2_5OmniProcessor.from_pretrained(
    LOCAL_MODEL_PATH, trust_remote_code=True, use_fast=False, local_files_only=True)
tokenizer = processor.tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("model loaded")

# ---------- load questions ----------
def load_questions():
    with open(AVHBENCH_QA_JSON, 'r', encoding='utf-8') as f:
        qa = json.load(f)
    qa = qa[:SAMPLE_NUM]
    out = []
    for s in qa:
        vp = os.path.join(AVHBENCH_VIDEO_DIR, f"{s['video_id']}.mp4")
        if not os.path.exists(vp): continue
        out.append({"text": s["text"], "multimodal": [{"type":"video","video":vp}],
                    "video_id": s["video_id"], "task": s["task"], "label": s["label"]})
    print(f"loaded {len(out)} samples")
    return out

questions = load_questions()

# ---------- inference ----------
def build_conv(q):
    uc = [{"type":"text","text":q["text"]}]
    for m in q["multimodal"]: uc.append({"type":m["type"], m["type"]:m[m["type"]]})
    return [{"role":"system","content":[{"type":"text","text":
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role":"user","content":uc}]

def run(vector_name):
    steering_vector = SteeringVector.load(VECTOR_PATH[vector_name])
    print(f"vector loaded: {VECTOR_PATH[vector_name]}")
    out_json = f"{BASE_DIR}/rh_qwen_avhbench_singlelayer_{vector_name}_L{FIXED_LAYERS[0]}_s{FIXED_STRENGTH}.json"
    ckpt_json = out_json.replace(".json", ".ckpt.json")
    # ckpt is a list aligned with questions (positional, like conditionadd)
    ckpt = None
    if os.path.exists(ckpt_json):
        try:
            with open(ckpt_json,'r',encoding='utf-8') as f: ckpt = json.load(f)
            done = sum(1 for x in ckpt if x is not None)
            print(f"resume: {done}/{len(questions)}")
        except: ckpt = None
    if ckpt is None:
        ckpt = [None]*len(questions)

    for idx, q in enumerate(tqdm(questions, ncols=80)):
        if ckpt[idx] is not None: continue
        conv = build_conv(q)
        with redirect_stdout(open(os.devnull,'w')), redirect_stderr(open(os.devnull,'w')):
            mmdl = MalleableModel(model=model, tokenizer=tokenizer)
            mmdl.steer(behavior_vector=steering_vector, behavior_layer_ids=FIXED_LAYERS,
                       behavior_vector_strength=FIXED_STRENGTH)
            audios, images, videos = process_mm_info(conv, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            kw = {"text":text,"return_tensors":"pt","padding":True,"use_audio_in_video":USE_AUDIO_IN_VIDEO}
            if torch.is_tensor(audios) and audios.numel()>0:
                kw["audio"] = audios.detach().cpu().numpy()
            if isinstance(images, list) and len(images)>0:
                kw["images"] = images
            if isinstance(videos, list) and len(videos)>0:
                kw["videos"] = videos
            inputs = processor(**kw).to(model.device).to(torch.float16)
            with torch.no_grad():
                ids = mmdl.model.generate(**inputs, max_new_tokens=10, do_sample=False, num_beams=1,
                                          pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        ans = processor.batch_decode(ids, skip_special_tokens=True)[0]
        raw = ans.split("assistant")[-1].strip() if "assistant" in ans else ans
        task = q["task"]
        if task in ["Video-driven Audio Hallucination","Audio-driven Video Hallucination","AV Matching"]:
            pred = "Yes" if "yes" in raw.lower() else "No" if "no" in raw.lower() else "Unknown"
        else:
            pred = raw
        if idx+1 <= PRINT_MAX:
            print(f"[{idx+1}] {task}: {pred} | gt={q['label']}")
        ckpt[idx] = {"video_id":q["video_id"],"task":task,"text":q["text"],
                     "label":q["label"],"model_answer":pred}
        with open(ckpt_json,'w',encoding='utf-8') as f:
            json.dump(ckpt, f, ensure_ascii=False)

    with open(out_json,'w',encoding='utf-8') as f:
        json.dump(ckpt, f, indent=2, ensure_ascii=False)
    if os.path.exists(ckpt_json): os.remove(ckpt_json)
    print(f"saved: {out_json}")
    return ckpt

def do_eval(pred_list, tag):
    # exact same as avhbench_official_eval.py: zip(gt[:2000], pred)
    with open(AVHBENCH_QA_JSON,'r',encoding='utf-8') as f:
        gt_all = json.load(f)
    gt_zip = gt_all[:len(pred_list)]
    TASK_LIST = ["Audio-driven Video Hallucination","Video-driven Audio Hallucination","AV Matching"]
    SKIP = {"AV Captioning"}
    tm = defaultdict(lambda: {"tp":0,"fp":0,"tn":0,"fn":0,"total":0,"correct":0})
    for g, p in zip(gt_zip, pred_list):
        task = g["task"]
        if task in SKIP or task not in TASK_LIST: continue
        ga = g["label"].strip().upper()
        pa = p["model_answer"].strip().upper()
        if ga == "YES":
            if pa == "YES": tm[task]["tp"] += 1; tm[task]["correct"] += 1
            else: tm[task]["fn"] += 1
        else:
            if pa == "YES": tm[task]["fp"] += 1
            else: tm[task]["tn"] += 1; tm[task]["correct"] += 1
        tm[task]["total"] += 1

    print("\n" + "="*60)
    print(f"{tag}")
    print("="*60)
    tc = ts = 0
    for task in TASK_LIST:
        m = tm[task]
        if m["total"] == 0: continue
        acc = m["correct"]/m["total"]
        prec = m["tp"]/(m["tp"]+m["fp"]) if (m["tp"]+m["fp"])>0 else 0
        rec = m["tp"]/(m["tp"]+m["fn"]) if (m["tp"]+m["fn"])>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        tc += m["correct"]; ts += m["total"]
        print(f"{task}: n={m['total']} Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    print(f"OVERALL: n={ts} Acc={tc/ts:.4f}")

if __name__ == "__main__":
    for vn in ["visual", "audio"]:
        print("\n" + "#"*60)
        print(f"# running {vn}")
        print("#"*60)
        res = run(vn)
        do_eval(res, f"single-layer {vn} L{FIXED_LAYERS[0]} s{FIXED_STRENGTH}")
