"""
Emoji 检测和过滤模块
用于训练时过滤包含 emoji 的样本，以及推理时获取 emoji token IDs
"""
import re
import unicodedata
from typing import List, Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def contains_emoji(text: str) -> bool:
    """
    检测文本中是否包含 emoji 表情符号
    
    Args:
        text: 要检测的文本
        
    Returns:
        bool: 如果包含emoji返回True，否则返回False
    """
    if not text:
        return False
    
    # Emoji Unicode 范围（涵盖大多数常见 emoji）
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号
        "\U0001F300-\U0001F5FF"  # 符号和图标
        "\U0001F680-\U0001F6FF"  # 交通和地图符号
        "\U0001F1E0-\U0001F1FF"  # 国旗（iOS）
        "\U0001F900-\U0001F9FF"  # 补充符号和图标
        "\U0001FA00-\U0001FA6F"  # 扩展A
        "\U0001FA70-\U0001FAFF"  # 扩展B
        "\U00002600-\U000026FF"  # 杂项符号（包含常见符号如❤️⭐）
        "\U00002700-\U000027BF"  # 装饰符号（包含✨✅✔️等）
        "\U0000FE00-\U0000FE0F"  # 变体选择器（emoji变体）
        "\U0001F004-\U0001F0CF"  # 麻将和扑克牌
        "\U0001F170-\U0001F251"  # 封闭字符（血型、按钮等）
        "]+",
        flags=re.UNICODE
    )
    
    # 检测是否匹配 emoji 模式
    return bool(emoji_pattern.search(text))


def filter_samples_with_emoji(samples: List[dict], target_key: str = 'continuation') -> tuple:
    """
    过滤包含 emoji 的训练样本
    
    Args:
        samples: 训练样本列表
        target_key: 要检测的字段名（默认为 'continuation'，也可以是 'next_question' 等）
        
    Returns:
        tuple: (过滤后的样本列表, 统计信息字典)
    """
    filtered_samples = []
    emoji_count = 0
    total_count = len(samples)
    
    for sample in samples:
        # 获取目标文本
        target_text = sample.get(target_key, '')
        
        # 如果没有找到 target_key，尝试其他可能的字段
        if not target_text and target_key == 'continuation':
            target_text = sample.get('next_question', '')
        
        # 检测是否包含 emoji
        if contains_emoji(target_text):
            emoji_count += 1
            continue  # 跳过包含 emoji 的样本
        
        filtered_samples.append(sample)
    
    # 统计信息
    stats = {
        'total_samples': total_count,
        'emoji_samples': emoji_count,
        'filtered_samples': len(filtered_samples),
        'emoji_ratio': emoji_count / total_count if total_count > 0 else 0.0,
        'kept_ratio': len(filtered_samples) / total_count if total_count > 0 else 0.0
    }
    
    return filtered_samples, stats


def get_emoji_token_ids(tokenizer) -> Set[int]:
    """
    获取 tokenizer 中所有 emoji 相关的 token IDs
    用于推理时的 logit bias
    
    Args:
        tokenizer: Hugging Face tokenizer (PreTrainedTokenizer)
        
    Returns:
        Set[int]: emoji token IDs 的集合
    """
    emoji_token_ids = set()
    
    # 常见的 emoji 列表（可以根据需要扩展）
    common_emojis = [
        # 表情符号
        '😀', '😁', '😂', '🤣', '😃', '😄', '😅', '😆', '😉', '😊',
        '😋', '😎', '😍', '😘', '🥰', '😗', '😙', '😚', '🙂', '🤗',
        '🤩', '🤔', '🤨', '😐', '😑', '😶', '🙄', '😏', '😣', '😥',
        '😮', '🤐', '😯', '😪', '😫', '🥱', '😴', '😌', '😛', '😜',
        '😝', '🤤', '😒', '😓', '😔', '😕', '🙃', '🤑', '😲', '☹️',
        '🙁', '😖', '😞', '😟', '😤', '😢', '😭', '😦', '😧', '😨',
        '😩', '🤯', '😬', '😰', '😱', '🥵', '🥶', '😳', '🤪', '😵',
        '😡', '😠', '🤬', '😷', '🤒', '🤕', '🤢', '🤮', '🤧', '😇',
        '🥳', '🥺', '🤠', '🤡', '🤥', '🤫', '🤭', '🧐', '🤓',
        
        # 手势符号
        '👍', '👎', '👌', '✌️', '🤞', '🤟', '🤘', '🤙', '👈', '👉',
        '👆', '👇', '☝️', '✋', '🤚', '🖐️', '🖖', '👋', '🤝', '🙏',
        '💪', '🦾', '🦿', '🦵', '🦶', '👂', '🦻', '👃', '🧠', '🦷',
        
        # 心形和情感符号
        '❤️', '🧡', '💛', '💚', '💙', '💜', '🖤', '🤍', '🤎', '💔',
        '❣️', '💕', '💞', '💓', '💗', '💖', '💘', '💝', '💟',
        
        # 其他常见符号
        '✨', '⭐', '🌟', '💫', '✅', '❌', '⚠️', '🔥', '💯', '🎉',
        '🎊', '🎈', '🎁', '🏆', '🥇', '🥈', '🥉', '👏', '🙌',
    ]
    
    for emoji in common_emojis:
        try:
            # 编码 emoji 并获取 token IDs
            token_ids = tokenizer.encode(emoji, add_special_tokens=False)
            emoji_token_ids.update(token_ids)
        except Exception:
            continue
    
    # 扫描 tokenizer 的词汇表，查找可能的 emoji tokens
    # 这个过程可能较慢，但更全面
    try:
        vocab = tokenizer.get_vocab()
        for token, token_id in vocab.items():
            # 解码 token 看是否包含 emoji
            try:
                decoded = tokenizer.decode([token_id], skip_special_tokens=True)
                if contains_emoji(decoded):
                    emoji_token_ids.add(token_id)
            except Exception:
                continue
    except Exception as e:
        print(f"警告: 无法扫描完整词汇表: {e}")
    
    return emoji_token_ids


def create_emoji_suppression_bias(
    tokenizer, 
    bias_value: float = -100.0
) -> dict:
    """
    创建用于推理时抑制 emoji 的 logit bias 字典
    
    Args:
        tokenizer: Hugging Face tokenizer (PreTrainedTokenizer)
        bias_value: 负值越大，抑制越强（推荐 -10.0 到 -100.0）
        
    Returns:
        dict: {token_id: bias_value} 的字典，可直接传给 model.generate()
    """
    emoji_token_ids = get_emoji_token_ids(tokenizer)
    
    # 创建 bias 字典
    logit_bias = {token_id: bias_value for token_id in emoji_token_ids}
    
    print(f"✓ 创建 emoji 抑制 bias: {len(logit_bias)} 个 token 将被抑制 (bias={bias_value})")
    
    return logit_bias


# 测试函数
if __name__ == "__main__":
    # 测试 emoji 检测
    test_cases = [
        ("这是一段正常文本", False),
        ("我爱你❤️", True),
        ("太棒了！😊", True),
        ("谢谢谢谢谢谢❤️❤️❤️❤️❤️", True),
        ("Hello world", False),
        ("好的👌", True),
        ("🎉🎉🎉", True),
    ]
    
    print("=" * 80)
    print("测试 Emoji 检测:")
    print("=" * 80)
    
    for text, expected in test_cases:
        result = contains_emoji(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{text}' -> {result} (期望: {expected})")
    
    print("\n" + "=" * 80)
    print("测试样本过滤:")
    print("=" * 80)
    
    test_samples = [
        {'continuation': '好的，我知道了'},
        {'continuation': '谢谢❤️'},
        {'continuation': '太棒了！😊😊😊'},
        {'continuation': '这是正常文本'},
        {'continuation': '明白了👌'},
    ]
    
    filtered, stats = filter_samples_with_emoji(test_samples)
    
    print(f"总样本数: {stats['total_samples']}")
    print(f"包含 emoji 样本数: {stats['emoji_samples']}")
    print(f"过滤后样本数: {stats['filtered_samples']}")
    print(f"Emoji 比例: {stats['emoji_ratio']:.2%}")
    print(f"保留比例: {stats['kept_ratio']:.2%}")
    
    print("\n过滤后的样本:")
    for i, sample in enumerate(filtered, 1):
        print(f"  {i}. {sample['continuation']}")
