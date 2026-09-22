import os
import gc
import json
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ===================== 🔥 核心配置（只改这里） =====================
# 切换模式："visual" = 训练视觉幻觉抑制向量；"audio" = 训练音频幻觉抑制向量
MODE = "audio"
# 训练使用的对比对数量（按分差从高到低取，保证样本质量）
TRAIN_PAIRS_NUM = 400

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

torch.cuda.empty_cache()
gc.collect()

LOCAL_MODEL_PATH = "/home/srt32/activation-steering/Qwen2.5-Omni-3B"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===================== 路径与Prompt自动匹配 =====================
INPUT_CSV_MAP = {
    "visual": "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_visual_hallucination_pairs.csv",
    "audio":  "/home/srt32/activation-steering/qwen_AVHDVDbench_task/dvd_audio_hallucination_pairs.csv"
}

VECTOR_SAVE_MAP = {
    "visual": "hallucination_reduction_vector_dvd_visual",
    "audio":  "hallucination_reduction_vector_dvd_audio"
}

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

PAIRED_CSV_PATH = INPUT_CSV_MAP[MODE]
VECTOR_SAVE_PATH = VECTOR_SAVE_MAP[MODE]
FIXED_PROMPT = PROMPT_MAP[MODE]

# ===================== 2. 模型兼容补丁 =====================
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

# ===================== 3. 模型加载 =====================
from transformers import (
    Qwen2_5OmniForConditionalGeneration, 
    Qwen2_5OmniProcessor
)
from activation_steering import SteeringVector, SteeringDataset

print(f"===== DVD 数据集转向向量训练【{MODE}模式】 =====")
print("🔄 正在加载 Qwen2.5-Omni 模型...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    dtype=torch.float16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True
)

model_layers = model.thinker.model.layers
model.layers = model_layers
model.config.num_hidden_layers = len(model_layers)
model.config.n_layer = len(model_layers)
model.config.hidden_size = 2048
model.config.model_type = "qwen2_5_omni"

def model_forward_wrapper(*args, **kwargs):
    kwargs.pop('output_hidden_states', None)
    kwargs.pop('return_dict', None)
    input_ids = kwargs.pop('input_ids', None)
    attention_mask = kwargs.pop('attention_mask', None)
    labels = kwargs.pop('labels', None)
    
    if input_ids is not None:
        input_ids = input_ids.to(model.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    if labels is not None:
        labels = labels.to(model.device)
    
    outputs = model.thinker.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        output_hidden_states=True,
        return_dict=True,
        **kwargs
    )
    lm_head_weight = model.thinker.lm_head.weight.to(torch.float16)
    logits = outputs.last_hidden_state @ lm_head_weight.t()
    setattr(outputs, 'logits', logits)
    return outputs

model.forward = model_forward_wrapper
model.disable_talker()
model.eval()

# ===================== 4. 加载对比对 + 高质量采样 =====================
processor = Qwen2_5OmniProcessor.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    use_fast=False,
    local_files_only=True
)
tokenizer = processor.tokenizer

print(f"\n🔍 正在加载对比对文件：{PAIRED_CSV_PATH}")
df = pd.read_csv(PAIRED_CSV_PATH).fillna("")
print(f"📊 原始对比对总数：{len(df)}")

# 🔥 核心：按正负分差降序排序，取对比度最强的N对
df = df.sort_values("score_gap", ascending=False).head(TRAIN_PAIRS_NUM).reset_index(drop=True)
print(f"📊 选取分差最大的 {len(df)} 对用于训练（样本对比度最高，效果最优）")
print(f"📊 采样后平均分差：{df['score_gap'].mean():.4f}")

# 构建训练样本
train_examples = []
train_suffixes = []

for idx, row in df.iterrows():
    pos_sent = str(row["positive"]).strip()
    neg_sent = str(row["negative"]).strip()
    
    if len(pos_sent) < 5 or len(neg_sent) < 5:
        continue
    
    train_examples.append((FIXED_PROMPT, FIXED_PROMPT))
    train_suffixes.append((pos_sent, neg_sent))

print(f"✅ 最终有效训练样本数：{len(train_examples)}")
print(f"📝 使用Prompt：{FIXED_PROMPT[:80]}...")

# ===================== 5. 构建数据集 =====================
print("\n🔄 正在构建转向数据集...")
hallucination_reduction_dataset = SteeringDataset(
    tokenizer=tokenizer,
    examples=train_examples,
    suffixes=train_suffixes
)

# ===================== 6. 训练转向向量 =====================
print("\n===== 开始训练幻觉抑制向量 =====")
try:
    hallucination_reduction_vector = SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=hallucination_reduction_dataset,
        method="pca_pairwise",          
        accumulate_last_x_tokens="suffix-only",
        batch_size=4                    
    )
    
    hallucination_reduction_vector.save(VECTOR_SAVE_PATH)
    print(f"✅ 向量训练完成！已保存：{VECTOR_SAVE_PATH}.svec")

except Exception as e:
    print(f"\n❌ 训练失败：{e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ===================== 7. 清理资源 =====================
torch.cuda.empty_cache()
gc.collect()
print("\n✅ 训练流程完成，显存已清理！")