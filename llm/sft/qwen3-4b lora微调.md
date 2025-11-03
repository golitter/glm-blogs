[self-llm/models/Qwen3/05-Qwen3-8B-LoRA及SwanLab可视化记录.md at master · datawhalechina/self-llm](https://github.com/datawhalechina/self-llm/blob/master/models/Qwen3/05-Qwen3-8B-LoRA及SwanLab可视化记录.md)

[实战 | Qwen3大模型微调入门实战（完整代码）-阿里云开发者社区 (aliyun.com)](https://developer.aliyun.com/article/1663178)

autodl：rtx3090 24G

用了大概13G显存。

```shell
conda create -n sft_qwen4b python=3.10 -y
conda activate sft_qwen4b
```



```shell
# 换清华镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.25.0
pip install transformers==4.51.3
pip install accelerate==1.6.0
pip install datasets==3.5.1
pip install peft==0.15.2
pip install swanlab==0.5.7

```



下载模型

```python
# model_download.py
# 注意修改cache_dir为保存的路径
from modelscope import snapshot_download
cache_dir = "./my_model_cache"  # 请修改我！！！
model_dir = snapshot_download('Qwen/Qwen3-4B', cache_dir=cache_dir, revision='master')

print(f"模型下载完成，保存路径为：{model_dir}")
```



lora微调：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_path = './my_model_cache/Qwen/Qwen3-4B'
lora_path = 'lora_path'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
def process_func(example):
    MAX_LENGTH = 1024 # 设置最大序列长度为1024个token
    input_ids, attention_mask, labels = [], [], [] # 初始化返回值
    # 适配chat_template
    instruction = tokenizer(
        f"<s><|im_start|>system\n现在你要扮演皇帝身边的女人--甄嬛<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    # 将instructio部分和response部分的input_ids拼接，并在末尾添加eos token作为标记结束的token
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # 注意力掩码，表示模型需要关注的位置
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # 对于instruction，使用-100表示这些位置不计算loss（即模型不需要预测这部分）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 超出最大序列长度截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# messages = [
#     {"role": "system", "content": "===system_message_test==="},
#     {"role": "user", "content": "===user_message_test==="},
#     {"role": "assistant", "content": "===assistant_message_test==="},
# ]

# # 不开 thinking mode
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=False
# )
# print("-----不开 thinking mode -----")
# print(text)


# # 开启 thinking mode
# messages = [
#     {"role": "system", "content": "===system_message_test==="},
#     {"role": "user", "content": "===user_message_test==="},
#     {"role": "assistant", "content": "===assistant_message_test==="},
# ]

# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=True
# )
# print("-----开启 thinking mode -----")
# print(text)


# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

print("Qwen4b 模型加载完成")


from peft import LoraConfig, TaskType, get_peft_model
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alpha
    lora_dropout=0.1 # Dropout 比例
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
args = TrainingArguments(
    output_dir="./output/Qwen3_4B_LoRA", # 注意修改
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    run_name="qwen3-4B",
)

# 处理数据集
from datasets import Dataset
import pandas as pd

# 假设 df 是原始数据（如 json/csv）
df = pd.read_json("huanhuan.json")
ds = Dataset.from_pandas(df)
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

import swanlab
from swanlab.integration.transformers import SwanLabCallback

# 实例化SwanLabCallback
swanlab_callback = SwanLabCallback(
    project="Qwen3-Lora",  # 注意修改
    experiment_name="Qwen3-4B-LoRA-experiment"  # 注意修改
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback] # 传入之前的swanlab_callback
)
trainer.train()
```

![image-20251103093230011](qwen3-4b%20lora%E5%BE%AE%E8%B0%83.assets/image-20251103093230011.png)

保存并进行简易测试：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_path = './my_model_cache/Qwen/Qwen3-4B'
lora_path = './output/Qwen3_4B_LoRA/checkpoint-702'

# --- 1. 加载 Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_path)

# --- 2. 加载基础模型和 LoRA 权重 ---
base_model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, model_id=lora_path)

# --- 3. 【关键步骤】合并 LoRA 权重 ---
print("正在合并 LoRA 权重...")
merged_model = model.merge_and_unload()
print("LoRA 权重合并完成！")

# --- 4. 【可选但推荐】保存合并后的模型 ---
# 你可以选择一个新目录来保存合并后的模型，以免覆盖原始模型
merged_model_path = "./Qwen3-4B-HuanHuan-Merged"
print(f"正在将合并后的模型保存到: {merged_model_path}")
merged_model.save_pretrained(merged_model_path)
# 也要保存 tokenizer，以便后续使用
tokenizer.save_pretrained(merged_model_path)
print("模型保存完成！")


# --- 5. 使用合并后的模型进行推理 ---
# 注意：现在 merged_model 是一个标准的 AutoModelForCausalLM，不再是 PeftModel
prompt = "你是谁？"
inputs = tokenizer.apply_chat_template(
    [{"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。"}, {"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
    enable_thinking=False
)

# 将输入移动到设备
device = merged_model.device
inputs_on_gpu = {k: v.to(device) for k, v in inputs.items()}

# 采样参数设置
# 注意：这里使用 max_new_tokens 而不是 max_length，更安全
gen_kwargs = {"max_new_tokens": 512, "do_sample": True, "top_p": 0.8, "temperature": 0.7}

with torch.no_grad():
    outputs = merged_model.generate(**inputs_on_gpu, **gen_kwargs)
    # 只解码新生成的部分
    response = outputs[0][inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))


```

![image-20251103093328097](qwen3-4b%20lora%E5%BE%AE%E8%B0%83.assets/image-20251103093328097.png)

合并完之后进行推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 直接加载合并后的模型
model_path = "./Qwen3-4B-HuanHuan-Merged"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

# 使用合并后的模型进行推理
prompt = "你是谁？"
inputs = tokenizer.apply_chat_template(
    [{"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。"}, {"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
    enable_thinking=False
)

# 将输入移动到设备
device = model.device
inputs_on_gpu = {k: v.to(device) for k, v in inputs.items()}

# 采样参数设置
# 注意：这里使用 max_new_tokens 而不是 max_length，更安全
gen_kwargs = {"max_new_tokens": 512, "do_sample": True, "top_p": 0.8, "temperature": 0.7}

with torch.no_grad():
    outputs = model.generate(**inputs_on_gpu, **gen_kwargs)
    # 只解码新生成的部分
    response = outputs[0][inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))


```

vllm

```shell
    vllm serve ./Qwen3-4B-HuanHuan-Merged \
        --served-model-name Huan-Qwen3-4B \
        --max_model_len 8192
```





查看模型：

```shell
curl http://localhost:8000/v1/models
```



推理：

```shell
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Huan-Qwen3-4B",
        "prompt": "我想问你，5的阶乘是多少",
        "max_tokens": 1024,
        "temperature": 0
    }'
```

