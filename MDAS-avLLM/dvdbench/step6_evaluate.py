"""
Qwen DVDbench baseline evaluation with INSERTION RATE instead of WER.
Usage:
  python qwen_dvd_eval_insertion.py
Outputs: test_dorca_metrics_qwen_baseline_insertion_test.json
"""
import os, sys, gc, re, ast, json, random, torch, warnings
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_omni_utils import process_mm_info
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache(); gc.collect()

MODEL_PATH = "/home/srt32/activation-steering/Qwen2.5-Omni-3B"
USE_AUDIO_IN_VIDEO = True
INPUT_DATA = "/home/srt32/activation-steering/qwen_AVHDVDbench_task/grpo_selfYtb_words_train_v2_100words_reanntTime.json"
VIDEO_LOCAL_DIR = "/home/srt32/activation-steering/dvd_dataset/videos/en_transcoded"
BASE_DIR = "/home/srt32/activation-steering/qwen_AVHDVDbench_task"

DATA_SPLIT = "test"
SPLIT_SEED = 42
TEST_RATIO = 0.7
OUTPUT_JSON = f"{BASE_DIR}/test_dorca_metrics_qwen_baseline_insertion_{DATA_SPLIT}.json"

PRINT_MAX = 3
FIXED_PROMPT = """
Transcribe all dialogues in this video strictly line by line in the following format:
[start_seconds - end_seconds] Speaker_X: exact speech content
Rules:
1. Number speakers as Speaker_1, Speaker_2... in the order they first speak
2. Time must be in seconds (e.g. [0.0 - 7.5])
3. Output the exact spoken words, do not paraphrase
4. One line per utterance, no extra explanation
"""
GEN_CONFIG = {"max_new_tokens": 2048, "do_sample": False, "num_beams": 1,
              "pad_token_id": 151643, "eos_token_id": 151643}

# ---------- utils ----------
def time_str_to_sec(s):
    p = list(map(float, s.strip().split(':')))
    return p[0]*60+p[1] if len(p)==2 else p[0]*3600+p[1]*60+p[2] if len(p)==3 else 0.0

def parse_ref(ref_str):
    try:
        d = ast.literal_eval(ref_str)
    except: return [], {}
    chars = d.get("Character", []); dials = d.get("Dialogue", [])
    smap = {c: f"Speaker_{i+1}" for i, c in enumerate(chars)}
    out = []
    for dial in dials:
        s, e = dial.get("time", ["0","0"])
        out.append({"start": time_str_to_sec(s), "end": time_str_to_sec(e),
                    "speaker": smap.get(dial.get("speaker",""),"Speaker_1").lower(),
                    "text": dial.get("content","")})
    return out, smap

def edit_counts(ref_words, hyp_words):
    """Return (S, D, I) substitution/deletion/insertion counts via edit distance backtracking."""
    n, m = len(ref_words), len(hyp_words)
    d = [[0]*(m+1) for _ in range(n+1)]
    op = [[None]*(m+1) for _ in range(n+1)]  # 'S'ub, 'D'el, 'I'ns, 'M'atch
    for i in range(n+1): d[i][0] = i; op[i][0] = 'D' if i>0 else None
    for j in range(m+1): d[0][j] = j; op[0][j] = 'I' if j>0 else None
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]; op[i][j] = 'M'
            else:
                vals = [(d[i-1][j]+1,'D'), (d[i][j-1]+1,'I'), (d[i-1][j-1]+1,'S')]
                d[i][j], op[i][j] = min(vals)
    S=D=I=0
    i, j = n, m
    while i>0 or j>0:
        c = op[i][j]
        if c == 'M': i-=1; j-=1
        elif c == 'S': S+=1; i-=1; j-=1
        elif c == 'D': D+=1; i-=1
        elif c == 'I': I+=1; j-=1
        else: break
    return S, D, I

def calc_iou(a, b):
    inter = max(0, min(a[1],b[1]) - max(a[0],b[0]))
    union = max(a[1],b[1]) - min(a[0],b[0])
    return inter/union if union>0 else 0.0

def parse_pred_dialogue(text):
    lines = text.strip().split('\n'); out = []; sc = 1
    p_s = r'\[?\s*(\d+\.?\d*)\s*[-–~]\s*(\d+\.?\d*)\s*\]?\s*(speaker_\d+)\s*[:：]\s*(.+)'
    p_n = r'\[?\s*(\d+\.?\d*)\s*[-–~]\s*(\d+\.?\d*)\s*\]?\s*(.+)'
    for line in lines:
        line = line.strip()
        if not line: continue
        m = re.search(p_s, line, re.IGNORECASE)
        if m:
            out.append({"start":float(m.group(1)),"end":float(m.group(2)),
                        "speaker":m.group(3).lower(),"text":m.group(4).strip()}); continue
        m = re.search(p_n, line, re.IGNORECASE)
        if m:
            out.append({"start":float(m.group(1)),"end":float(m.group(2)),
                        "speaker":f"speaker_{sc}","text":m.group(3).strip()}); sc+=1
    return out

def compute_metrics(ref_dialogue, pred_text):
    pred = parse_pred_dialogue(pred_text)
    if not pred or not ref_dialogue:
        return {"speaker_acc":0.0,"insertion_rate":1.0,"temporal_iou":0.0,"parsed_num":0}
    rs = sorted(ref_dialogue, key=lambda x:x["start"])
    ps = sorted(pred, key=lambda x:x["start"])
    ml = min(len(rs), len(ps))
    correct = sum(1 for i in range(ml) if rs[i]["speaker"]==ps[i]["speaker"])
    spk = correct/len(rs)
    ref_full = " ".join(s["text"] for s in rs).lower().split()
    pred_full = " ".join(s["text"] for s in ps).lower().split()
    S,D,I = edit_counts(ref_full, pred_full)
    ins_rate = I/len(ref_full) if ref_full else 0.0
    iou_sum = sum(calc_iou((rs[i]["start"],rs[i]["end"]),(ps[i]["start"],ps[i]["end"])) for i in range(ml))
    return {"speaker_acc":round(spk,4),"insertion_rate":round(ins_rate,4),
            "temporal_iou":round(iou_sum/len(rs),4),"parsed_num":len(pred)}

def is_abnormal(m):
    return m["speaker_acc"]==0.0 and m["insertion_rate"]==1.0 and m["temporal_iou"]==0.0

def load_samples():
    with open(INPUT_DATA,'r',encoding='utf-8') as f: all_data = json.load(f)
    vids = {f for f in os.listdir(VIDEO_LOCAL_DIR) if f.endswith(".mp4")}
    samples = []
    for item in all_data:
        vn = os.path.basename(item["video"])
        if vn not in vids: continue
        rd, sm = parse_ref(item.get("ref",""))
        if not rd: continue
        samples.append({"video_path":os.path.join(VIDEO_LOCAL_DIR,vn),"video_id":vn,
                        "ref_dialogue":rd,"speaker_map":sm})
    rng = random.Random(SPLIT_SEED); sh = samples.copy(); rng.shuffle(sh)
    n_test = int(len(sh)*TEST_RATIO)
    sel = sh[:n_test] if DATA_SPLIT=="test" else sh[n_test:]
    print(f"📊 split seed={SPLIT_SEED} {DATA_SPLIT}={len(sel)}")
    return sel

# ---------- model ----------
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
print("\n🔄 loading Qwen2.5-Omni-3B ...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH, dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True)
model.disable_talker(); model.eval()
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True, local_files_only=True)

def build_conv(vp):
    return [{"role":"system","content":[{"type":"text","text":
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role":"user","content":[{"type":"text","text":FIXED_PROMPT},{"type":"video","video":vp}]}]

def get_inputs(conv):
    au, im, vi = process_mm_info(conv, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    kw = {"text":text,"return_tensors":"pt","padding":True,"use_audio_in_video":USE_AUDIO_IN_VIDEO}
    if im: kw["images"]=im
    if vi: kw["videos"]=vi
    if au is not None: kw["audio"]=au.cpu().numpy() if torch.is_tensor(au) else au
    return processor(**kw).to(model.device, model.dtype)

def generate(inputs):
    with torch.no_grad(): ids = model.generate(**inputs, **GEN_CONFIG)
    ans = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return ans.split("assistant")[-1].strip() if "assistant" in ans else ans.strip()

if __name__ == "__main__":
    samples = load_samples()
    print(f"\n===== baseline insertion eval ({len(samples)} videos) =====")
    ckpt_json = OUTPUT_JSON.replace(".json", ".ckpt.json")
    ckpt = {}
    if os.path.exists(ckpt_json):
        try:
            with open(ckpt_json, 'r', encoding='utf-8') as f:
                ckpt = json.load(f)
            print(f"📂 resume: {len(ckpt)}/{len(samples)} done")
        except:
            ckpt = {}

    def save_ckpt():
        with open(ckpt_json, 'w', encoding='utf-8') as f:
            json.dump(ckpt, f, ensure_ascii=False)

    for idx, s in enumerate(tqdm(samples, desc="baseline", ncols=80), 1):
        vid = s["video_id"]
        if vid in ckpt:
            continue
        try:
            conv = build_conv(s["video_path"])
            inputs = get_inputs(conv)
            pred = generate(inputs)
            m = compute_metrics(s["ref_dialogue"], pred)
            ckpt[vid] = {"video_id": vid, "pred_text": pred, "metrics": m,
                          "ref_dialogue": s["ref_dialogue"],
                          "status": "abnormal" if is_abnormal(m) else "valid"}
            if idx <= PRINT_MAX:
                print(f"[{idx}] {vid} ins={m['insertion_rate']:.4f} spk={m['speaker_acc']:.4f} iou={m['temporal_iou']:.4f}")
            del inputs; torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            ckpt[vid] = {"video_id": vid, "status": "failed", "error": str(e)[:100]}
            print(f"⚠️ fail {vid}: {str(e)[:80]}")
            torch.cuda.empty_cache(); gc.collect()
        save_ckpt()

    valid=[]; abnormal=[]; failed=[]
    s_acc=s_ins=s_iou=0.0
    for vid, e in ckpt.items():
        if e["status"]=="valid":
            valid.append(e); m=e["metrics"]; s_acc+=m["speaker_acc"]; s_ins+=m["insertion_rate"]; s_iou+=m["temporal_iou"]
        elif e["status"]=="abnormal": abnormal.append(e)
        else: failed.append({"video_id":vid,"error":e.get("error","")})

    vc = len(valid)
    avg = {"avg_speaker_acc":round(s_acc/vc,4) if vc else 0,
           "avg_insertion_rate":round(s_ins/vc,4) if vc else 0,
           "avg_temporal_iou":round(s_iou/vc,4) if vc else 0}
    print(f"\n📈 baseline avg: spk={avg['avg_speaker_acc']:.4f} ins={avg['avg_insertion_rate']:.4f} iou={avg['avg_temporal_iou']:.4f}")
    out = {"total":len(samples),"valid":vc,"abnormal":len(abnormal),"failed":len(failed),
           "average_metrics":avg,"valid_results":valid,"abnormal_results":abnormal,"failed_list":failed}
    with open(OUTPUT_JSON,'w',encoding='utf-8') as f: json.dump(out,f,indent=2,ensure_ascii=False)
    if os.path.exists(ckpt_json): os.remove(ckpt_json)
    print(f"📄 saved: {OUTPUT_JSON}")
