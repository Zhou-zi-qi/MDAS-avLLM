"""
Qwen DVDbench layer+strength search with INSERTION RATE (not WER).
Specific configs from the ablation table:
  visual: strengths [0.5,1.0,1.5,2.0] at layers [18,19,20,21]
  audio:  strength 0.5 at [18,19,20,21]; strengths [1.0,1.5,2.0] at [22,23,24,25]
Usage:
  python qwen_dvd_layer_insertion.py --mode visual
  python qwen_dvd_layer_insertion.py --mode audio
"""
import os, sys, gc, re, ast, json, random, argparse
import multiprocessing as mp
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")
import transformers
transformers.logging.set_verbosity_error()
import logging
logging.getLogger().setLevel(logging.CRITICAL)

os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
os.environ["PYTORCH_ALLOC_CONF"]="expandable_segments:True,max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"]="false"

GPU_IDS = [0,1,2,3,4,5]
MODEL_PATH = "/home/srt32/activation-steering/Qwen2.5-Omni-3B"
USE_AUDIO_IN_VIDEO = True
INPUT_DATA = "/home/srt32/activation-steering/qwen_AVHDVDbench_task/grpo_selfYtb_words_train_v2_100words_reanntTime.json"
VIDEO_LOCAL_DIR = "/home/srt32/activation-steering/dvd_dataset/videos/en_transcoded"
DATA_SPLIT = "test"; SPLIT_SEED = 42; TEST_RATIO = 0.7
BASE_DIR = "/home/srt32/activation-steering/qwen_AVHDVDbench_task"

# Specific (strength, layers) configs per modality
CONFIGS = {
    "visual": [(0.5,[18,19,20,21]), (1.0,[18,19,20,21]),
               (1.5,[18,19,20,21]), (2.0,[18,19,20,21])],
    "audio":  [(0.5,[18,19,20,21]), (1.0,[22,23,24,25]),
               (1.5,[22,23,24,25]), (2.0,[22,23,24,25])],
}
VECTOR_PATH = {
    "visual": f"{BASE_DIR}/hallucination_reduction_vector_dvd_visual.svec",
    "audio":  f"{BASE_DIR}/hallucination_reduction_vector_dvd_audio.svec",
}

GEN_CONFIG = {"max_new_tokens":2048,"do_sample":False,"num_beams":1,
              "pad_token_id":151643,"eos_token_id":151643}
FIXED_PROMPT = """
Transcribe all dialogues in this video strictly line by line in the following format:
[start_seconds - end_seconds] Speaker_X: exact speech content
Rules:
1. Number speakers as Speaker_1, Speaker_2... in the order they first speak
2. Time must be in seconds (e.g. [0.0 - 7.5])
3. Output the exact spoken words, do not paraphrase
4. One line per utterance, no extra explanation
"""

# ---------- utils ----------
def time_str_to_sec(s):
    p = list(map(float, s.strip().split(':')))
    return p[0]*60+p[1] if len(p)==2 else p[0]*3600+p[1]*60+p[2] if len(p)==3 else 0.0

def parse_ref(ref_str):
    try: d = ast.literal_eval(ref_str)
    except: return [], {}
    chars=d.get("Character",[]); dials=d.get("Dialogue",[])
    smap={c:f"Speaker_{i+1}" for i,c in enumerate(chars)}
    out=[]
    for dial in dials:
        s,e=dial.get("time",["0","0"])
        out.append({"start":time_str_to_sec(s),"end":time_str_to_sec(e),
                    "speaker":smap.get(dial.get("speaker",""),"Speaker_1").lower(),
                    "text":dial.get("content","")})
    return out, smap

def edit_counts(ref_words, hyp_words):
    n,m=len(ref_words),len(hyp_words)
    d=[[0]*(m+1) for _ in range(n+1)]
    op=[[None]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0]=i; op[i][0]='D' if i>0 else None
    for j in range(m+1): d[0][j]=j; op[0][j]='I' if j>0 else None
    for i in range(1,n+1):
        for j in range(1,m+1):
            if ref_words[i-1]==hyp_words[j-1]:
                d[i][j]=d[i-1][j-1]; op[i][j]='M'
            else:
                v=[(d[i-1][j]+1,'D'),(d[i][j-1]+1,'I'),(d[i-1][j-1]+1,'S')]
                d[i][j],op[i][j]=min(v)
    S=D=I=0; i,j=n,m
    while i>0 or j>0:
        c=op[i][j]
        if c=='M': i-=1; j-=1
        elif c=='S': S+=1; i-=1; j-=1
        elif c=='D': D+=1; i-=1
        elif c=='I': I+=1; j-=1
        else: break
    return S,D,I

def calc_iou(a,b):
    inter=max(0,min(a[1],b[1])-max(a[0],b[0]))
    union=max(a[1],b[1])-min(a[0],b[0])
    return inter/union if union>0 else 0.0

def parse_pred(text):
    lines=text.strip().split('\n'); out=[]; sc=1
    p_s=r'\[?\s*(\d+\.?\d*)\s*[-–~]\s*(\d+\.?\d*)\s*\]?\s*(speaker_\d+)\s*[:：]\s*(.+)'
    p_n=r'\[?\s*(\d+\.?\d*)\s*[-–~]\s*(\d+\.?\d*)\s*\]?\s*(.+)'
    for line in lines:
        line=line.strip()
        if not line: continue
        m=re.search(p_s,line,re.IGNORECASE)
        if m:
            out.append({"start":float(m.group(1)),"end":float(m.group(2)),
                        "speaker":m.group(3).lower(),"text":m.group(4).strip()}); continue
        m=re.search(p_n,line,re.IGNORECASE)
        if m:
            out.append({"start":float(m.group(1)),"end":float(m.group(2)),
                        "speaker":f"speaker_{sc}","text":m.group(3).strip()}); sc+=1
    return out

def compute_metrics(rd, pred_text):
    pd=parse_pred(pred_text)
    if not pd or not rd:
        return {"speaker_acc":0.0,"insertion_rate":1.0,"temporal_iou":0.0,"parsed_num":0}
    rs=sorted(rd,key=lambda x:x["start"]); ps=sorted(pd,key=lambda x:x["start"])
    ml=min(len(rs),len(ps))
    correct=sum(1 for i in range(ml) if rs[i]["speaker"]==ps[i]["speaker"])
    ref_full=" ".join(s["text"] for s in rs).lower().split()
    hyp_full=" ".join(s["text"] for s in ps).lower().split()
    S,D,I=edit_counts(ref_full,hyp_full)
    ins=I/len(ref_full) if ref_full else 0.0
    iou_sum=sum(calc_iou((rs[i]["start"],rs[i]["end"]),(ps[i]["start"],ps[i]["end"])) for i in range(ml))
    return {"speaker_acc":round(correct/len(rs),4),"insertion_rate":round(ins,4),
            "temporal_iou":round(iou_sum/len(rs),4),"parsed_num":len(pd)}

def is_abnormal(m):
    return m["speaker_acc"]==0.0 and m["insertion_rate"]==1.0 and m["temporal_iou"]==0.0

def load_samples():
    with open(INPUT_DATA,'r',encoding='utf-8') as f: all_data=json.load(f)
    vids={f for f in os.listdir(VIDEO_LOCAL_DIR) if f.endswith(".mp4")}
    samples=[]
    for item in all_data:
        vn=os.path.basename(item["video"])
        if vn not in vids: continue
        rd,sm=parse_ref(item.get("ref",""))
        if not rd: continue
        samples.append({"video_path":os.path.join(VIDEO_LOCAL_DIR,vn),"video_id":vn,
                        "ref_dialogue":rd,"speaker_map":sm})
    rng=random.Random(SPLIT_SEED); sh=samples.copy(); rng.shuffle(sh)
    n_test=int(len(sh)*TEST_RATIO)
    sel=sh[:n_test] if DATA_SPLIT=="test" else sh[n_test:]
    print(f"📊 split {DATA_SPLIT}={len(sel)}")
    return sel

# ---------- worker ----------
def worker(gpu_id, task_batch, mode, return_dict):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from activation_steering import MalleableModel, SteeringVector
    from activation_steering.malleable_model import LeashLayer
    from activation_steering import malleable_model as mm
    from activation_steering import steering_vector as sv

    torch.cuda.empty_cache(); gc.collect()
    orig=LeashLayer.__init__
    def patch(self, layer, *a, **k):
        orig(self,layer,*a,**k)
        if hasattr(layer,'attention_type'): self.attention_type=layer.attention_type
        if hasattr(layer,'layer_idx'): self.layer_idx=layer.layer_idx
    LeashLayer.__init__=patch
    def gll(model):
        if "Qwen2_5OmniForConditionalGeneration" in str(type(model)): return model.thinker.model.layers
        elif hasattr(model,"model") and hasattr(model.model,"layers"): return model.model.layers
        elif hasattr(model,"transformer") and hasattr(model.transformer,"h"): return model.transformer.h
        raise ValueError(str(type(model)))
    mm.get_model_layer_list=gll; sv.get_model_layer_list=gll

    print(f"[GPU{gpu_id}] loading model...")
    model=Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,dtype=torch.float16,device_map={"":0},low_cpu_mem_usage=True,
        trust_remote_code=True,local_files_only=True)
    def fw(input_ids, attention_mask=None, **kw):
        kw.pop('output_hidden_states',None)
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            out=model.thinker.model(input_ids=input_ids,attention_mask=attention_mask,
                                     return_dict=True,output_hidden_states=True,**kw)
        hw=model.thinker.lm_head.weight.to(torch.float16)
        setattr(out,'logits',out.last_hidden_state@hw.t())
        setattr(out,'hidden_states',out.hidden_states)
        return out
    model.forward=fw
    model.layers=model.thinker.model.layers
    model.config.num_hidden_layers=len(model.layers); model.config.hidden_size=2048
    model.disable_talker(); model.eval()
    proc=Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH,trust_remote_code=True,use_fast=False,local_files_only=True)
    tok=proc.tokenizer; tok.pad_token=tok.eos_token; tok.padding_side="left"
    sv_path=VECTOR_PATH[mode]
    vec=SteeringVector.load(sv_path)
    samples=load_samples()
    print(f"[GPU{gpu_id}] ready, {len(task_batch)} tasks, {len(samples)} samples")

    def build_conv(vp):
        return [{"role":"system","content":[{"type":"text","text":
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role":"user","content":[{"type":"text","text":FIXED_PROMPT},{"type":"video","video":vp}]}]

    results=[]
    for strength, layers in task_batch:
        s_a=s_i=s_n=0.0; valid=0; abn=0; failed=[]
        for s in tqdm(samples, desc=f"[GPU{gpu_id}] {mode} s{strength} L{layers}", ncols=80):
            vid=s["video_id"]
            try:
                conv=build_conv(s["video_path"])
                with redirect_stdout(open(os.devnull,'w')), redirect_stderr(open(os.devnull,'w')):
                    mmdl=MalleableModel(model=model, tokenizer=tok)
                    mmdl.steer(behavior_vector=vec, behavior_layer_ids=layers,
                               behavior_vector_strength=strength)
                    au,im,vi=process_mm_info(conv,use_audio_in_video=USE_AUDIO_IN_VIDEO)
                    text=proc.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
                    kw={"text":text,"return_tensors":"pt","padding":True,"use_audio_in_video":USE_AUDIO_IN_VIDEO}
                    if torch.is_tensor(au) and au.numel()>0: kw["audio"]=au.detach().cpu().numpy()
                    if isinstance(vi,list) and len(vi)>0: kw["videos"]=vi
                    inputs=proc(**kw).to(model.device).to(torch.float16)
                    with torch.no_grad(): ids=mmdl.model.generate(**inputs,**GEN_CONFIG)
                ans=proc.batch_decode(ids,skip_special_tokens=True)[0]
                pred=ans.split("assistant")[-1].strip() if "assistant" in ans else ans.strip()
                m=compute_metrics(s["ref_dialogue"], pred)
                if is_abnormal(m): abn+=1
                else:
                    valid+=1; s_a+=m["speaker_acc"]; s_i+=m["insertion_rate"]; s_n+=m["temporal_iou"]
                del inputs, mmdl; torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                failed.append({"video_id":vid,"error":str(e)[:80]})
        res={"mode":mode,"layers":layers,"strength":strength,
             "avg_speaker_acc":round(s_a/valid,4) if valid else 0,
             "avg_insertion_rate":round(s_i/valid,4) if valid else 0,
             "avg_temporal_iou":round(s_n/valid,4) if valid else 0,
             "valid":valid,"abnormal":abn,"failed":len(failed)}
        print(f"[GPU{gpu_id}] done s={strength} L={layers} | spk={res['avg_speaker_acc']:.4f} ins={res['avg_insertion_rate']:.4f} iou={res['avg_temporal_iou']:.4f}")
        results.append(res)
    del model, vec; torch.cuda.empty_cache(); gc.collect()
    return_dict[gpu_id]=results

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["visual","audio"], required=True)
    args=ap.parse_args()
    mp.set_start_method("spawn", force=True)
    tasks=CONFIGS[args.mode]
    print(f"🚀 mode={args.mode}, tasks={tasks}")
    nb=len(GPU_IDS)
    batches=[[] for _ in range(nb)]
    for i,t in enumerate(tasks): batches[i%nb].append(t)
    mgr=mp.Manager(); rd=mgr.dict(); procs=[]
    for g,b in zip(GPU_IDS,batches):
        if not b: continue
        p=mp.Process(target=worker,args=(g,b,args.mode,rd)); procs.append(p); p.start()
    for p in procs: p.join()
    all_res=[]
    for v in rd.values(): all_res.extend(v)
    all_res.sort(key=lambda x:(x["strength"], str(x["layers"])))
    out=f"{BASE_DIR}/qwen_dvd_layer_insertion_{args.mode}_{DATA_SPLIT}.json"
    with open(out,'w',encoding='utf-8') as f: json.dump(all_res,f,indent=2,ensure_ascii=False)
    print(f"\n📄 saved: {out}")
    for r in all_res:
        print(f"  s={r['strength']} L={r['layers']} spk={r['avg_speaker_acc']:.4f} ins={r['avg_insertion_rate']:.4f} iou={r['avg_temporal_iou']:.4f}")
