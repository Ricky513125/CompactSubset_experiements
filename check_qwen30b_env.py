#!/usr/bin/env python3
"""
Qwen 30B 环境和配置验证脚本
运行此脚本检查训练环境是否配置正确
"""

import os
import sys
import json
from pathlib import Path

def check_file_exists(filepath, name):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✓ {name}: {filepath}")
        return True
    else:
        print(f"✗ {name} 不存在: {filepath}")
        return False

def check_directory_exists(dirpath, name):
    """检查目录是否存在"""
    if os.path.isdir(dirpath):
        print(f"✓ {name}: {dirpath}")
        return True
    else:
        print(f"✗ {name} 不存在: {dirpath}")
        return False

def check_gpu_availability():
    """检查GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ GPU可用: {gpu_count} 张")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
            return True
        else:
            print("✗ GPU不可用")
            return False
    except ImportError:
        print("✗ PyTorch未安装")
        return False

def check_deepspeed():
    """检查DeepSpeed是否安装"""
    try:
        import deepspeed
        version = deepspeed.__version__
        print(f"✓ DeepSpeed已安装: v{version}")
        return True
    except ImportError:
        print("✗ DeepSpeed未安装")
        return False

def check_transformers():
    """检查Transformers是否安装"""
    try:
        import transformers
        version = transformers.__version__
        print(f"✓ Transformers已安装: v{version}")
        return True
    except ImportError:
        print("✗ Transformers未安装")
        return False

def check_config_file(config_path):
    """检查配置文件内容"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"\n配置文件内容:")
        print(f"  模型名称: {config['model']['name']}")
        print(f"  模型路径: {config['model']['path']}")
        print(f"  训练数据: {config['data']['train_path']}")
        print(f"  测试数据: {config['data']['test_path']}")
        print(f"  批次大小: {config['training']['batch_size']}")
        print(f"  梯度累积: {config['training']['gradient_accumulation_steps']}")
        print(f"  学习率: {config['training']['learning_rate']}")
        print(f"  最大长度: {config['training']['max_length']}")
        
        # 检查模型路径
        model_path = config['model']['path']
        if os.path.isdir(model_path):
            print(f"✓ 模型目录存在")
        else:
            print(f"✗ 模型目录不存在: {model_path}")
            return False
        
        # 检查数据路径
        train_path = config['data']['train_path']
        test_path = config['data']['test_path']
        
        if os.path.exists(train_path):
            print(f"✓ 训练数据存在")
        else:
            print(f"✗ 训练数据不存在: {train_path}")
            return False
            
        if os.path.exists(test_path):
            print(f"✓ 测试数据存在")
        else:
            print(f"✗ 测试数据不存在: {test_path}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ 配置文件读取失败: {e}")
        return False

def main():
    print("=" * 60)
    print("Qwen 30B 训练环境检查")
    print("=" * 60)
    
    all_checks_passed = True
    
    # 1. 检查Python版本
    print("\n1. Python 版本")
    python_version = sys.version.split()[0]
    print(f"  Python {python_version}")
    
    # 2. 检查GPU
    print("\n2. GPU 状态")
    if not check_gpu_availability():
        all_checks_passed = False
    
    # 3. 检查依赖包
    print("\n3. Python 依赖包")
    if not check_deepspeed():
        all_checks_passed = False
    if not check_transformers():
        all_checks_passed = False
    
    # 4. 检查配置文件
    print("\n4. 配置文件")
    config_files = [
        ("config_RealPersonaChat_Qwen30B.json", "Qwen30B配置文件"),
        ("ds_config_zero3_30b.json", "DeepSpeed配置文件"),
    ]
    
    for filepath, name in config_files:
        if not check_file_exists(filepath, name):
            all_checks_passed = False
    
    # 5. 检查训练脚本
    print("\n5. 训练脚本")
    scripts = [
        ("train_distributed_RealPersonaChat.py", "训练脚本"),
        ("run_qwen30b_train.sh", "训练启动脚本"),
        ("run_qwen30b_inference.sh", "推理启动脚本"),
    ]
    
    for filepath, name in scripts:
        if not check_file_exists(filepath, name):
            all_checks_passed = False
    
    # 6. 详细检查配置文件
    print("\n6. 配置文件详细信息")
    if not check_config_file("config_RealPersonaChat_Qwen30B.json"):
        all_checks_passed = False
    
    # 7. 检查模型文件
    print("\n7. 模型文件")
    model_dir = "/mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507"
    if check_directory_exists(model_dir, "模型目录"):
        # 检查关键文件
        key_files = ["config.json", "generation_config.json"]
        for filename in key_files:
            filepath = os.path.join(model_dir, filename)
            if not check_file_exists(filepath, f"  {filename}"):
                all_checks_passed = False
        
        # 统计模型分片数量
        safetensors_files = list(Path(model_dir).glob("*.safetensors"))
        print(f"  模型分片数量: {len(safetensors_files)}")
    else:
        all_checks_passed = False
    
    # 8. 检查输出目录是否可写
    print("\n8. 输出目录")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isdir(output_dir) and os.access(output_dir, os.W_OK):
        print(f"✓ 输出目录可写: {output_dir}")
    else:
        print(f"✗ 输出目录不可写: {output_dir}")
        all_checks_passed = False
    
    # 9. 检查环境变量
    print("\n9. 环境变量")
    env_vars = {
        "CUDA_VISIBLE_DEVICES": "可见的GPU设备",
        "RANK": "分布式训练rank",
        "WORLD_SIZE": "总进程数",
        "LOCAL_RANK": "本地rank",
    }
    
    for var, desc in env_vars.items():
        value = os.environ.get(var, "未设置")
        print(f"  {var}: {value}")
    
    # 总结
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✓ 所有检查通过！环境配置正确")
        print("\n可以开始训练:")
        print("  bash run_qwen30b_train.sh profile_and_context v1")
    else:
        print("✗ 部分检查失败，请修复上述问题后再开始训练")
    print("=" * 60)
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())
