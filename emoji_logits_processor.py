"""
Emoji Logits Processor
用于推理时抑制 emoji token 的生成
"""
import torch
from typing import List, Optional
from transformers import LogitsProcessor


class EmojiSuppressionLogitsProcessor(LogitsProcessor):
    """
    自定义 LogitsProcessor，用于抑制 emoji token 的生成
    通过对 emoji token 的 logits 添加负偏置来降低其生成概率
    """
    
    def __init__(self, emoji_token_ids: List[int], bias_value: float = -100.0):
        """
        Args:
            emoji_token_ids: 需要抑制的 emoji token ID 列表
            bias_value: 负偏置值，越小抑制越强（推荐 -10.0 到 -100.0）
                       -10.0: 轻度抑制
                       -50.0: 中度抑制
                       -100.0: 强力抑制（几乎不可能生成）
        """
        self.emoji_token_ids = set(emoji_token_ids)
        self.bias_value = bias_value
        
        if not self.emoji_token_ids:
            print("警告: emoji_token_ids 为空，Emoji 抑制将不起作用")

    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        对 emoji tokens 应用负偏置
        
        Args:
            input_ids: 当前已生成的 token IDs [batch_size, seq_len]
            scores: 下一个 token 的 logits [batch_size, vocab_size]
            
        Returns:
            修改后的 logits
        """
        # 对所有 emoji token IDs 应用负偏置
        for token_id in self.emoji_token_ids:
            if token_id < scores.shape[-1]:  # 确保 token_id 在有效范围内
                scores[:, token_id] += self.bias_value
        
        return scores


class AdaptiveEmojiSuppressionLogitsProcessor(LogitsProcessor):
    """
    自适应 Emoji 抑制 Logits Processor
    根据已生成内容中 emoji 的数量动态调整抑制强度
    """
    
    def __init__(
        self, 
        emoji_token_ids: List[int], 
        base_bias: float = -50.0,
        max_bias: float = -200.0,
        emoji_threshold: int = 2
    ):
        """
        Args:
            emoji_token_ids: 需要抑制的 emoji token ID 列表
            base_bias: 基础负偏置值（当未检测到emoji时）
            max_bias: 最大负偏置值（当检测到多个emoji时）
            emoji_threshold: emoji 数量阈值，超过此值将应用最大抑制
        """
        self.emoji_token_ids = set(emoji_token_ids)
        self.base_bias = base_bias
        self.max_bias = max_bias
        self.emoji_threshold = emoji_threshold
        
        print(f"✓ 自适应 Emoji Logits Processor 已初始化:")
        print(f"  - 抑制 {len(self.emoji_token_ids)} 个 emoji tokens")
        print(f"  - 基础 bias: {self.base_bias}")
        print(f"  - 最大 bias: {self.max_bias}")
        print(f"  - Emoji 阈值: {self.emoji_threshold}")
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        根据已生成的 emoji 数量动态调整抑制强度
        """
        batch_size = input_ids.shape[0]
        
        for i in range(batch_size):
            # 统计当前序列中已生成的 emoji 数量
            emoji_count = sum(1 for token_id in input_ids[i].tolist() if token_id in self.emoji_token_ids)
            
            # 根据 emoji 数量计算 bias
            if emoji_count == 0:
                bias = self.base_bias
            elif emoji_count < self.emoji_threshold:
                # 线性插值
                ratio = emoji_count / self.emoji_threshold
                bias = self.base_bias + (self.max_bias - self.base_bias) * ratio
            else:
                bias = self.max_bias
            
            # 应用 bias
            for token_id in self.emoji_token_ids:
                if token_id < scores.shape[-1]:
                    scores[i, token_id] += bias
        
        return scores


def create_emoji_suppression_processor(
    tokenizer, 
    mode: str = "normal",
    bias_value: float = -100.0,
    **kwargs
) -> Optional[LogitsProcessor]:
    """
    创建 Emoji 抑制 Logits Processor
    
    Args:
        tokenizer: Hugging Face tokenizer
        mode: 抑制模式
            - "normal": 标准抑制（固定 bias）
            - "adaptive": 自适应抑制（根据已生成 emoji 数量调整）
            - "off": 关闭抑制
        bias_value: 负偏置值（仅用于 normal 模式）
        **kwargs: 其他参数（用于 adaptive 模式）
            - base_bias: 基础 bias（默认 -50.0）
            - max_bias: 最大 bias（默认 -200.0）
            - emoji_threshold: emoji 数量阈值（默认 2）
    
    Returns:
        LogitsProcessor 实例，如果 mode="off" 则返回 None
    """
    if mode == "off":
        print("ℹ️  Emoji 抑制已关闭")
        return None
    
    # 导入 emoji_filter 获取 token IDs
    try:
        from emoji_filter import get_emoji_token_ids
    except ImportError:
        print("❌ 错误: 无法导入 emoji_filter，Emoji 抑制功能不可用")
        return None
    
    # ✅ 关键修复：获取 emoji token IDs
    # print("🔍 正在扫描 tokenizer 词汇表以获取 emoji token IDs...")
    emoji_token_ids = get_emoji_token_ids(tokenizer)
    
    if not emoji_token_ids:
        print("⚠️  警告: 未找到任何 emoji tokens，抑制功能将不起作用")
        return None
    
    # print(f"✓ 找到 {len(emoji_token_ids)} 个 emoji tokens")
    # print("=" * 80)
    
    # 根据模式创建 processor
    if mode == "normal":
        return EmojiSuppressionLogitsProcessor(
            emoji_token_ids=list(emoji_token_ids),
            bias_value=bias_value
        )
    elif mode == "adaptive":
        return AdaptiveEmojiSuppressionLogitsProcessor(
            emoji_token_ids=list(emoji_token_ids),
            base_bias=kwargs.get('base_bias', -50.0),
            max_bias=kwargs.get('max_bias', -200.0),
            emoji_threshold=kwargs.get('emoji_threshold', 2)
        )
    else:
        print(f"❌ 错误: 未知的抑制模式 '{mode}'")
        return None


# 测试代码
if __name__ == "__main__":
    print("="*80)
    print("Emoji Logits Processor 测试")
    print("="*80)
    
    # 模拟测试
    emoji_token_ids = [100, 200, 300]  # 假设的 emoji token IDs
    
    print("\n1. 测试标准抑制模式:")
    processor = EmojiSuppressionLogitsProcessor(emoji_token_ids, bias_value=-50.0)
    
    # 创建模拟 logits
    import torch
    batch_size = 2
    vocab_size = 500
    seq_len = 10
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    scores = torch.randn(batch_size, vocab_size)
    
    print(f"  原始 scores[0, {emoji_token_ids[0]}]: {scores[0, emoji_token_ids[0]]:.4f}")
    
    modified_scores = processor(input_ids, scores)
    
    print(f"  修改后 scores[0, {emoji_token_ids[0]}]: {modified_scores[0, emoji_token_ids[0]]:.4f}")
    print(f"  差值: {modified_scores[0, emoji_token_ids[0]] - scores[0, emoji_token_ids[0]]:.4f}")
    
    print("\n2. 测试自适应抑制模式:")
    adaptive_processor = AdaptiveEmojiSuppressionLogitsProcessor(
        emoji_token_ids, 
        base_bias=-50.0,
        max_bias=-200.0,
        emoji_threshold=2
    )
    
    # 创建包含 emoji 的输入序列
    input_with_emoji = torch.tensor([[100, 50, 200, 80, 150, 200, 90, 110, 120, 130]])  # 包含3个emoji
    scores_new = torch.randn(1, vocab_size)
    
    print(f"  输入序列中包含 {sum(1 for id in input_with_emoji[0].tolist() if id in emoji_token_ids)} 个 emoji")
    print(f"  原始 scores[0, {emoji_token_ids[0]}]: {scores_new[0, emoji_token_ids[0]]:.4f}")
    
    modified_scores_new = adaptive_processor(input_with_emoji, scores_new)
    
    print(f"  修改后 scores[0, {emoji_token_ids[0]}]: {modified_scores_new[0, emoji_token_ids[0]]:.4f}")
    print(f"  差值: {modified_scores_new[0, emoji_token_ids[0]] - scores_new[0, emoji_token_ids[0]]:.4f}")
    
    print("\n" + "="*80)
    print("✓ 测试完成")
    print("="*80)
