import os
import gc
import json
import torch
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 环境配置 =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

torch.cuda.empty_cache()
gc.collect()

# 路径配置
LOCAL_MODEL_PATH = "/home/srt32/activation-steering/Qwen2.5-Omni-3B"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONDITION_DATA_PATH = "/home/srt32/activation-steering/avh_condition_data.json"
VECTOR_SAVE_PATH = "avh_hallucination_classify_vector"
OPTIMAL_POINT_PATH = "optimal_condition_point_avh.json"

# ===================== 2. LeashLayer 兼容补丁（沿用你稳定版本） =====================
from activation_steering.malleable_model import LeashLayer
original_leash_init = LeashLayer.__init__
def patched_leash_init(self, layer, *args, **kwargs):
    original_leash_init(self, layer, *args, **kwargs)
    if hasattr(layer, 'attention_type'):
        self.attention_type = layer.attention_type
    if hasattr(layer, 'layer_idx'):
        self.layer_idx = layer.layer_idx
LeashLayer.__init__ = patched_leash_init

# ===================== 3. 模型层获取函数 =====================
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

# ===================== 4. 加载Qwen2.5-Omni模型 + 8bit兼容 =====================
from transformers import (
    Qwen2_5OmniForConditionalGeneration, 
    Qwen2_5OmniProcessor, 
    BitsAndBytesConfig
)
from activation_steering import MalleableModel, SteeringVector, SteeringDataset

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

# 加载模型
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    dtype=torch.float16,
    device_map={"": 0},
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True
)

# 修复8bit模型设备迁移方法
def no_op_to(self, *args, **kwargs):
    return self
model.to = no_op_to.__get__(model)
model.thinker.to = no_op_to.__get__(model.thinker)
model.thinker.model.to = no_op_to.__get__(model.thinker.model)

# 自定义前向传播
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

# ===================== 🔥 Qwen2.5专属配置字段补齐（解决核心报错） =====================
model.layers = model.thinker.model.layers
layer_num = len(model.layers)
# 手动补充 steering 库依赖的层数属性，适配Qwen2.5配置结构
model.config.num_hidden_layers = layer_num
model.config.n_layer = layer_num
model.config.hidden_size = 2048
model.config.model_type = "qwen2_5_omni"

model.disable_talker()
model.eval()

# ===================== 5. 加载Tokenizer =====================
processor = Qwen2_5OmniProcessor.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    use_fast=False,
    local_files_only=True
)
tokenizer = processor.tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.model_max_length = 512
print("✅ Qwen2.5-Omni模型+Tokenizer加载完成，层数：", len(model.layers))

# ===================== 6. 训练AVH分类条件向量 =====================
print("\n===== 开始训练 音频/视频幻觉分类条件向量 =====")
with open(CONDITION_DATA_PATH, "r", encoding="utf-8") as f:
    condition_data = json.load(f)

audio_samples = []
video_samples = []
for pair in condition_data["train"]:
    audio_samples.append(pair["audio_driven_video_hallucination"].strip())
    video_samples.append(pair["video_driven_audio_hallucination"].strip())

print(f"样本统计：音频驱动{len(audio_samples)}条 | 视频驱动{len(video_samples)}条")

dataset = SteeringDataset(
    tokenizer=tokenizer,
    examples=list(zip(audio_samples, video_samples)),
    disable_suffixes=True
)

# 向量训练参数适配Qwen2.5
vec = SteeringVector.train(
    model=model,
    tokenizer=tokenizer,
    steering_dataset=dataset,
    method="pca_pairwise",
    accumulate_last_x_tokens=1,
    batch_size=2
)
vec.save(VECTOR_SAVE_PATH)
print(f"✅ 条件向量保存成功：{VECTOR_SAVE_PATH}.svec")

# ===================== 7. 完整搜索最优判断参数 =====================
print("\n===== 搜索最优层、阈值、方向参数 =====")
malleable_model = MalleableModel(model=model, tokenizer=tokenizer)

best_layer, best_threshold, best_direction, best_score = malleable_model.find_best_condition_point(
    positive_strings=audio_samples[:150],
    negative_strings=video_samples[:150],
    condition_vector=vec,
    layer_range=(8, 18),
    threshold_range=(0.0, 0.10),
    threshold_step=0.001,
    save_analysis=True,
    file_path=OPTIMAL_POINT_PATH
)

# 解析单层参数
best_layer = best_layer[0] if isinstance(best_layer, list) else best_layer

# ===================== 8. 结果输出 =====================
print("\n🎉 Qwen2.5模型训练&寻参全部完成")
print(f"最佳判定层：{best_layer}")
print(f"最优相似度阈值：{best_threshold:.4f}")
print(f"判定对比方向：{best_direction}")
print(f"综合匹配得分：{best_score:.4f}")
print(f"参数文件已写入：{OPTIMAL_POINT_PATH}")

# 显存清理
torch.cuda.empty_cache()
gc.collect()