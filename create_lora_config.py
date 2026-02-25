#!/usr/bin/env python3
"""
LoRA 微调配置生成器
为 Chameleons 数据集生成 LoRA 训练配置
"""

import json
from pathlib import Path

# 基础配置
base_config = {
    "model": {
        "name": "Qwen3-30B-A3B-Instruct-2507",
        "path": "/mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507",
        "hf_model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "checkpoint_dir": "/mnt/parallel/checkpoints",
        
        # LoRA 配置
        "use_lora": True,
        "lora_config": {
            "r": 64,                    # LoRA rank (32-128)
            "lora_alpha": 128,          # LoRA alpha (通常 = 2*r)
            "lora_dropout": 0.05,       # Dropout
            "target_modules": [         # 要应用 LoRA 的模块
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    },
    "data": {
        "train_path": "sampled_data/Chameleons/train_di3.json"
    },
    "training": {
        "batch_size": 2,              # LoRA 可以用更大的 batch size
        "eval_batch_size": 2,
        "gradient_accumulation_steps": 4,  # 有效 batch = 2*4*8 = 64
        "learning_rate": 2e-4,        # LoRA 通常用更高的学习率
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "max_length": 1024,
        "max_context_turns": 15,
        "logging_steps": 10,
        "save_steps": 200,            # 更频繁保存
        "save_total_limit": 5
    },
    "ablation_configs": {
        "profile_and_history_and_context": {
            "use_profile": True,
            "use_history": True,
            "use_context": True,
            "name": "profile_and_history_and_context"
        },
        "profile_and_history": {
            "use_profile": True,
            "use_history": True,
            "use_context": False,
            "name": "profile_and_history"
        },
        "profile_and_context": {
            "use_profile": True,
            "use_history": False,
            "use_context": True,
            "name": "profile_and_context"
        },
        "history_and_context": {
            "use_profile": False,
            "use_history": True,
            "use_context": True,
            "name": "history_and_context"
        },
        "profile_only": {
            "use_profile": True,
            "use_history": False,
            "use_context": False,
            "name": "profile_only"
        },
        "history_only": {
            "use_profile": False,
            "use_history": True,
            "use_context": False,
            "name": "history_only"
        },
        "context_only": {
            "use_profile": False,
            "use_history": False,
            "use_context": True,
            "name": "context_only"
        }
    }
}

# 保存配置
output_path = Path("config_Chameleons_30B_lora_di3.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(base_config, f, indent=2, ensure_ascii=False)

print(f"✅ LoRA 配置已保存到: {output_path}")
print("\n配置要点:")
print(f"  - LoRA rank: {base_config['model']['lora_config']['r']}")
print(f"  - Target modules: {len(base_config['model']['lora_config']['target_modules'])} 个")
print(f"  - Batch size: {base_config['training']['batch_size']} (有效 batch size = {base_config['training']['batch_size'] * base_config['training']['gradient_accumulation_steps'] * 8})")
print(f"  - Learning rate: {base_config['training']['learning_rate']}")
print(f"  - 数据集: {base_config['data']['train_path']}")
print("\n预期改进:")
print("  ⚡ 训练速度: 3-5x 加速")
print("  💾 显存占用: 50-70% 减少")
print("  ⏱️  训练时间: 从 12.5 天 → 2-3 天")
