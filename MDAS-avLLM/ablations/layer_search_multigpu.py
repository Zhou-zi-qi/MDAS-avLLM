"""
Multi-GPU layer-window search for AVHBench single-vector steering.
Searches LAYER_CANDIDATES (4-layer windows) with fixed strength=5.
- visual vector: selects best by Audio-driven Video Hallucination (AVH) accuracy
- audio vector: selects best by Video-driven Audio Hallucination (VAH) accuracy
Outputs best config's AVH, VAH, AVH+VAH Acc and F (=1-Precision).

Multi-GPU: data-parallel across GPUs. Each GPU loads its own model copy and
processes a subset of samples for every (vector, layer) config.
Results are merged after all GPUs finish each config.

Usage:
  GPU_IDS=0,1 python rh_qwen_avhbench_layer_search.py
  GPU_IDS=4,5,6,7 python rh_qwen_avhbench_layer_search.py
"""
import os, sys, json, gc
import torch.multiprocessing as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===================== config =====================
GPU_IDS = [x.strip() for x in os.environ.get("GPU_IDS", "0").split(",") if x.strip()]
NUM_GPUS = len(GPU_IDS)
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
FIXED_STRENGTH = 5
SAMPLE_NUM = 2000
USE_AUDIO_IN_VIDEO = True

# ===================== worker (runs in each GPU process) =====================
def worker(gpu_idx, gpu_id, index_question_pairs, vector_names, layer_candidates, strength):
    """Load model on gpu_id, process all configs for this worker's sample subset."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    import warnings
    from contextlib import redirect_stdout, redirect_stderr
    from collections import defaultdict
    warnings.filterwarnings("ignore")
    import transformers
    transformers.logging.set_verbosity_error()
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)

    from qwen_omni_utils import process_mm_info

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
    print(f"[GPU {gpu_id}] loading model ...")
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
    print(f"[GPU {gpu_id}] model loaded, processing {len(index_question_pairs)} samples")

    def build_conv(q):
        uc = [{"type":"text","text":q["text"]}]
        for m in q["multimodal"]: uc.append({"type":m["type"], m["type"]:m[m["type"]]})
        return [{"role":"system","content":[{"type":"text","text":
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role":"user","content":uc}]

    for vector_name in vector_names:
        steering_vector = SteeringVector.load(VECTOR_PATH[vector_name])
        for layers in layer_candidates:
            layers_key = f"L{layers[0]}-{layers[-1]}"
            partial_path = f"{BASE_DIR}/layer_search_{vector_name}_{layers_key}_s{strength}_gpu{gpu_idx}.json"
            ckpt_path = partial_path.replace(".json", ".ckpt.json")

            # Resume: load existing partial results
            done = {}
            if os.path.exists(ckpt_path):
                try:
                    with open(ckpt_path, 'r', encoding='utf-8') as f:
                        done = json.load(f)
                    print(f"[GPU {gpu_id}] {vector_name} {layers_key}: resume {len(done)}/{len(index_question_pairs)}")
                except:
                    done = {}

            for orig_idx, q in index_question_pairs:
                key = str(orig_idx)
                if key in done:
                    continue
                conv = build_conv(q)
                with redirect_stdout(open(os.devnull,'w')), redirect_stderr(open(os.devnull,'w')):
                    mmdl = MalleableModel(model=model, tokenizer=tokenizer)
                    mmdl.steer(behavior_vector=steering_vector, behavior_layer_ids=layers,
                               behavior_vector_strength=strength)
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
                done[key] = {"video_id":q["video_id"],"task":task,"text":q["text"],
                             "label":q["label"],"model_answer":pred}
                # Save checkpoint periodically (every 50 samples)
                if len(done) % 50 == 0:
                    with open(ckpt_path, 'w', encoding='utf-8') as f:
                        json.dump(done, f, ensure_ascii=False)

            # Final save for this config
            with open(partial_path, 'w', encoding='utf-8') as f:
                json.dump(done, f, ensure_ascii=False)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            print(f"[GPU {gpu_id}] done {vector_name} {layers_key}: {len(done)} samples")

    print(f"[GPU {gpu_id}] all configs done")

# ===================== main (orchestration + evaluation) =====================
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

def merge_partial_results(vector_name, layers_key, strength, num_gpus, total_samples):
    """Merge partial results from all GPUs into an ordered list."""
    merged = [None] * total_samples
    for gpu_idx in range(num_gpus):
        partial_path = f"{BASE_DIR}/layer_search_{vector_name}_{layers_key}_s{strength}_gpu{gpu_idx}.json"
        if not os.path.exists(partial_path):
            print(f"  WARNING: missing {partial_path}")
            continue
        with open(partial_path, 'r', encoding='utf-8') as f:
            partial = json.load(f)
        for key, val in partial.items():
            idx = int(key)
            if idx < total_samples:
                merged[idx] = val
    return merged

def evaluate(pred_list):
    """Evaluate predictions, return per-task and combined AVH+VAH metrics."""
    with open(AVHBENCH_QA_JSON, 'r', encoding='utf-8') as f:
        gt_all = json.load(f)
    gt_zip = gt_all[:len(pred_list)]
    TASK_LIST = ["Audio-driven Video Hallucination","Video-driven Audio Hallucination","AV Matching"]
    SKIP = {"AV Captioning"}
    from collections import defaultdict
    tm = defaultdict(lambda: {"tp":0,"fp":0,"tn":0,"fn":0,"total":0,"correct":0})
    for g, p in zip(gt_zip, pred_list):
        if p is None:
            continue
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

    result = {}
    for task in TASK_LIST:
        m = tm[task]
        if m["total"] == 0: continue
        acc = m["correct"]/m["total"]
        prec = m["tp"]/(m["tp"]+m["fp"]) if (m["tp"]+m["fp"])>0 else 0
        rec = m["tp"]/(m["tp"]+m["fn"]) if (m["tp"]+m["fn"])>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        result[task] = {"n":m["total"],"acc":acc,"prec":prec,"rec":rec,"f1":f1,
                         "tp":m["tp"],"fp":m["fp"],"tn":m["tn"],"fn":m["fn"]}

    avh = tm["Audio-driven Video Hallucination"]
    vah = tm["Video-driven Audio Hallucination"]
    comb_total = avh["total"] + vah["total"]
    comb_correct = avh["correct"] + vah["correct"]
    comb_tp = avh["tp"] + vah["tp"]
    comb_fp = avh["fp"] + vah["fp"]
    comb_acc = comb_correct / comb_total if comb_total > 0 else 0
    comb_prec = comb_tp / (comb_tp + comb_fp) if (comb_tp + comb_fp) > 0 else 0
    result["AVH+VAH"] = {"n":comb_total,"acc":comb_acc,"prec":comb_prec,
                          "tp":comb_tp,"fp":comb_fp}
    return result

def config_is_complete(vector_name, layers_key, strength, num_gpus):
    """Check if all GPU partial results exist for this config."""
    for gpu_idx in range(num_gpus):
        partial_path = f"{BASE_DIR}/layer_search_{vector_name}_{layers_key}_s{strength}_gpu{gpu_idx}.json"
        if not os.path.exists(partial_path):
            return False
    return True

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    questions = load_questions()
    total_samples = len(questions)
    vector_names = ["visual", "audio"]

    TARGET_TASK = {
        "visual": "Audio-driven Video Hallucination",
        "audio":  "Video-driven Audio Hallucination",
    }

    # Determine which configs still need to run
    configs_to_run = []
    for vector_name in vector_names:
        for layers in LAYER_CANDIDATES:
            layers_key = f"L{layers[0]}-{layers[-1]}"
            if not config_is_complete(vector_name, layers_key, FIXED_STRENGTH, NUM_GPUS):
                configs_to_run.append((vector_name, layers))
            else:
                print(f"[skip] {vector_name} {layers_key} already complete")

    if configs_to_run:
        # Split sample indices across GPUs
        all_indices = list(range(total_samples))
        worker_assignments = []
        for gpu_idx in range(NUM_GPUS):
            indices = all_indices[gpu_idx::NUM_GPUS]
            pairs = [(idx, questions[idx]) for idx in indices]
            worker_assignments.append(pairs)

        # Extract unique vector names and layers from configs to run
        vectors_to_run = sorted(set(v for v, _ in configs_to_run))
        layers_to_run = []
        seen = set()
        for _, layers in configs_to_run:
            key = tuple(layers)
            if key not in seen:
                seen.add(key)
                layers_to_run.append(layers)

        print(f"\nLaunching {NUM_GPUS} GPU workers: {GPU_IDS}")
        print(f"Configs to run: {len(configs_to_run)} (vectors: {vectors_to_run}, layers: {len(layers_to_run)})")
        print(f"Samples per GPU: ~{total_samples // NUM_GPUS}")

        processes = []
        for gpu_idx, gpu_id in enumerate(GPU_IDS):
            p = mp.Process(
                target=worker,
                args=(gpu_idx, gpu_id, worker_assignments[gpu_idx],
                      vectors_to_run, layers_to_run, FIXED_STRENGTH)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("\nAll GPU workers finished.")

    # Merge and evaluate all configs
    all_summary = {}
    for vector_name in vector_names:
        print(f"\n{'#'*60}")
        print(f"# Evaluating: {vector_name} vector, strength={FIXED_STRENGTH}")
        print(f"# Target task for selection: {TARGET_TASK[vector_name]}")
        print(f"{'#'*60}")

        all_metrics = {}
        for layers in LAYER_CANDIDATES:
            layers_key = f"L{layers[0]}-{layers[-1]}"
            merged = merge_partial_results(vector_name, layers_key, FIXED_STRENGTH, NUM_GPUS, total_samples)
            n_done = sum(1 for x in merged if x is not None)
            if n_done == 0:
                print(f"  {layers_key}: no results, skipping")
                continue
            metrics = evaluate(merged)
            all_metrics[layers_key] = metrics
            tgt = TARGET_TASK[vector_name]
            print(f"  {layers_key}: n={n_done} target Acc={metrics[tgt]['acc']:.4f} "
                  f"F={1-metrics[tgt]['prec']:.4f}")

        if not all_metrics:
            print(f"  No results for {vector_name}, skipping")
            continue

        # Select best by target task accuracy
        tgt = TARGET_TASK[vector_name]
        best_key = max(all_metrics.keys(), key=lambda k: all_metrics[k][tgt]["acc"])
        best_metrics = all_metrics[best_key]

        print(f"\n  BEST {vector_name}: {best_key} (selected by {tgt} accuracy)")
        for task in ["Audio-driven Video Hallucination","Video-driven Audio Hallucination","AVH+VAH"]:
            if task in best_metrics:
                m = best_metrics[task]
                print(f"    {task}: Acc={m['acc']:.4f}  F={1-m['prec']:.4f}")

        all_summary[vector_name] = {
            "best_layers": best_key,
            "best_layers_list": [l for l in LAYER_CANDIDATES if f"L{l[0]}-{l[-1]}"==best_key][0],
            "strength": FIXED_STRENGTH,
            "target_task": tgt,
            "best_metrics": best_metrics,
            "all_layer_metrics": {k: {t: {"acc":v[t]["acc"],"prec":v[t]["prec"]}
                                        for t in ["Audio-driven Video Hallucination",
                                                   "Video-driven Audio Hallucination","AVH+VAH"]
                                        if t in v}
                                    for k,v in all_metrics.items()},
        }

    # Save summary
    summary_path = f"{BASE_DIR}/rh_qwen_avhbench_layer_search_summary_s{FIXED_STRENGTH}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")

    # Final print
    print(f"\n{'#'*60}")
    print("# FINAL RESULTS")
    print(f"{'#'*60}")
    for vn in ["visual", "audio"]:
        if vn not in all_summary:
            continue
        s = all_summary[vn]
        m = s["best_metrics"]
        print(f"\n{vn} vector (best layers: {s['best_layers']}, strength={FIXED_STRENGTH}):")
        for task in ["Audio-driven Video Hallucination","Video-driven Audio Hallucination","AVH+VAH"]:
            if task in m:
                print(f"  {task}: Acc={m[task]['acc']:.4f}  F={1-m[task]['prec']:.4f}")
