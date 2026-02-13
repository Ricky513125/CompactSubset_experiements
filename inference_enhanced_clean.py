"""
增强版推理清理模块
专门解决：
1. 日文中夹杂英文
2. 重复生成
3. Assistant/GS等元数据
4. 语言混乱
5. Emoji 表情符号清除 ✅
"""
import re
from typing import List

# 导入 emoji 检测和清除功能
try:
    from emoji_filter import contains_emoji
except ImportError:
    # 如果无法导入，提供一个简单的备用实现
    def contains_emoji(text):
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF"
            "\U00002600-\U000026FF\U00002700-\U000027BF\U0000FE00-\U0000FE0F"
            "\U0001F004-\U0001F0CF\U0001F170-\U0001F251]+"
        )
        return bool(emoji_pattern.search(text))


def remove_excessive_interjections(text: str, max_repeats: int = 2) -> str:
    """
    清理过度重复的语气词和拟声词
    
    Args:
        text: 输入文本
        max_repeats: 允许的最大连续重复次数（默认2次）
        
    Returns:
        清理后的文本
    """
    if not text:
        return text
    
    # 第一步：处理任何单字符的过度重复（通用规则）
    # 将 3个或更多的连续相同字符 -> 最多2个
    cleaned = re.sub(r'(.)\1{2,}', r'\1\1', text)  # ✅ 哈哈哈哈哈 -> 哈哈
    
    # 第二步：处理特定语气词（更严格）
    interjection_patterns = [
        (r'(哈哈)+哈', '哈哈'),     # 哈哈哈 -> 哈哈（更严格）
        (r'(嘻){2,}', ''),          # 嘻嘻 -> 删除
        (r'(嘿){2,}', ''),          # 嘿嘿 -> 删除
        (r'(呜){2,}', ''),          # 呜呜 -> 删除
        (r'(哇){2,}', ''),          # 哇哇 -> 删除
        (r'(啊){2,}', ''),          # 啊啊 -> 删除
        (r'(噜){1,}', ''),          # 噜 -> 删除
        (r'(嗷){1,}', ''),          # 嗷 -> 删除
        (r'(嘟){1,}', ''),          # 嘟 -> 删除
        (r'(拉){2,}', ''),          # 拉拉 -> 删除
        (r'(啦){2,}', ''),          # 啦啦 -> 删除
        (r'(哦){2,}', ''),          # 哦哦 -> 删除
        (r'(呵){2,}', ''),          # 呵呵 -> 删除
        (r'(嗯){2,}', ''),          # 嗯嗯 -> 删除
        (r'(哼){2,}', ''),          # 哼哼 -> 删除
        (r'(嗨){2,}', ''),          # 嗨嗨 -> 删除
        (r'(哟){2,}', ''),          # 哟哟 -> 删除
        (r'(喂){2,}', ''),          # 喂喂 -> 删除
        (r'(哎){2,}', ''),          # 哎哎 -> 删除
        (r'(呦){2,}', ''),          # 呦呦 -> 删除
    ]
    
    for pattern, replacement in interjection_patterns:
        cleaned = re.sub(pattern, replacement, cleaned)
    
    # 第三步：删除整个无意义的拟声词序列
    meaningless_sequences = [
        r'呜哇\w*',
        r'嗷嗷\w*',
        r'嘟噜\w*',
        r'啦啦\w*',
        r'拉拉\w*',
        r'呜呜\w*',
        r'我的天\w*',
    ]
    
    for pattern in meaningless_sequences:
        cleaned = re.sub(pattern, '', cleaned)
    
    # 第四步：清理英文重复（如 "hahahaha", "hihihi"）
    cleaned = re.sub(r'\b([a-z]{2,})\1{2,}\b', r'\1', cleaned, flags=re.IGNORECASE)  # hahaha -> ha
    cleaned = re.sub(r'\b(ha|hi|hey|lol|hehe){3,}\b', '', cleaned, flags=re.IGNORECASE)  # 删除过度重复的英文笑声
    
    # 清理可能产生的多余空格和标点
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'\s+([，。！？、])', r'\1', cleaned)  # 标点前的空格
    
    return cleaned


def remove_unicode_replacement_chars(text: str) -> str:
    """
    清除Unicode替换字符（�）和其他无效字符
    这些字符通常是由于emoji token解码失败产生的
    
    Args:
        text: 输入文本
        
    Returns:
        清除后的文本
    """
    if not text:
        return text
    
    # 移除 Unicode 替换字符（U+FFFD: �）
    cleaned = text.replace('\ufffd', '')
    cleaned = cleaned.replace('�', '')
    
    # 移除其他常见的无效字符
    invalid_chars = [
        '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
        '\x08', '\x0b', '\x0c', '\x0e', '\x0f', '\x10', '\x11', '\x12',
        '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a',
        '\x1b', '\x1c', '\x1d', '\x1e', '\x1f'
    ]
    for char in invalid_chars:
        cleaned = cleaned.replace(char, '')
    
    # 清理多余空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def remove_all_emoji(text: str) -> str:
    """
    强制清除文本中的所有 emoji 表情符号
    这是最后一道防线，确保输出中完全没有 emoji
    
    Args:
        text: 输入文本
        
    Returns:
        清除 emoji 后的文本
    """
    if not text:
        return text
    
    # Emoji Unicode 范围（扩展覆盖）
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 表情符号（😀-🙏）
        "\U0001F300-\U0001F5FF"  # 符号和图标
        "\U0001F680-\U0001F6FF"  # 交通和地图符号
        "\U0001F1E0-\U0001F1FF"  # 国旗
        "\U0001F900-\U0001F9FF"  # 补充符号和图标
        "\U0001FA00-\U0001FA6F"  # 扩展A
        "\U0001FA70-\U0001FAFF"  # 扩展B
        "\U00002600-\U000026FF"  # 杂项符号（包含❤️⭐）
        "\U00002700-\U000027BF"  # 装饰符号（包含✨✅等）
        "\U0000FE00-\U0000FE0F"  # 变体选择器（emoji变体）
        "\U0001F004-\U0001F0CF"  # 麻将和扑克牌
        "\U0001F170-\U0001F251"  # 封闭字符
        "]+",
        flags=re.UNICODE
    )
    
    # 清除所有匹配的 emoji
    cleaned_text = emoji_pattern.sub('', text)
    
    # 清理可能产生的多余空格
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def detect_language(text: str) -> str:
    """检测文本的主要语言"""
    # 日文字符（平假名、片假名、汉字）
    japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text))
    # 中文字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 英文单词
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
    
    total = len(text)
    if total == 0:
        return "unknown"
    
    if japanese_chars > total * 0.3:  # 30%以上日文字符
        return "japanese"
    elif chinese_chars > total * 0.3:
        return "chinese"
    elif english_words > 5:
        return "english"
    return "mixed"


def remove_english_from_japanese(text: str) -> str:
    """
    从日语文本中移除英文单词（保留必要的英文缩写和专有名词）
    """
    # 保留的英文词（常见缩写和专有名词）
    keep_words = {
        'AI', 'IT', 'PC', 'TV', 'DVD', 'CD', 'USB', 'WiFi', 'LINE', 'Twitter', 
        'Facebook', 'YouTube', 'Google', 'iOS', 'Android', 'OK', 'NG',
        'SNS', 'DM', 'PM', 'AM', 'vs', 'etc', 'App', 'Web', 'A', 'B', 'C'
    }
    
    def replace_english(match):
        word = match.group(0)
        # 如果是保留词，不替换
        if word in keep_words or word.upper() in keep_words:
            return word
        # 如果是单个字母，可能是缩写，保留
        if len(word) <= 1:
            return word
        # 否则移除
        return ''
    
    # 匹配所有英文单词（包括2个字母的）
    text = re.sub(r'\b[a-zA-Z]+\b', replace_english, text)
    
    # 清理多余的空格
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    
    return text


def remove_duplicates(text: str) -> str:
    """
    移除重复的句子或短语
    """
    # 按句子分割（日语句号、问号、感叹号）
    sentences = re.split(r'([。！？\n])', text)
    
    # 重组句子（包含标点）
    full_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            full_sentences.append(sentences[i] + sentences[i + 1])
        else:
            full_sentences.append(sentences[i])
    
    # 如果没有分割出句子（即原文没有。！？\n），直接返回原文
    if not full_sentences and sentences:
        full_sentences = [sentences[0]]
    
    # 去重（保留第一次出现）
    seen = set()
    unique_sentences = []
    for sent in full_sentences:
        sent_clean = sent.strip()
        if sent_clean and sent_clean not in seen:
            seen.add(sent_clean)
            unique_sentences.append(sent)
    
    result = ''.join(unique_sentences)
    
    # 如果结果为空，返回原文
    if not result.strip():
        return text
    
    # 额外处理：检测短语级别的重复（如"おはよう！おはよう！"）
    # 检测2-10个字的重复模式
    for length in range(10, 1, -1):
        pattern = r'(.{' + str(length) + r'})\1+'
        result = re.sub(pattern, r'\1', result)
    
    return result


def remove_metadata_and_roles(text: str) -> str:
    """
    移除Assistant、GS、user等元数据和角色标识
    """
    # 移除角色标识模式
    role_patterns = [
        r'\b[Aa]ssistant\s*[:\：]?\s*',  # Assistant: 或 assistant:
        r'\b[Uu]ser\s*[:\：]?\s*',       # User: 或 user:
        r'\bGS\s*[:\：]?\s*',            # GS:
        r'\b[Bb]ot\s*[:\：]?\s*',        # Bot:
        r'\b[Aa][Ii]\s*[:\：]?\s*',      # AI:
        r'\b回答者\s*[:\：]?\s*',         # 回答者:
        r'\b質問者\s*[:\：]?\s*',         # 質問者:
        r'^\s*[>\-\*]\s*',               # 开头的 >, -, * 等
    ]
    
    for pattern in role_patterns:
        text = re.sub(pattern, '', text)
    
    # 移除特殊标记
    special_markers = [
        r'<\|im_start\|>.*?\n',
        r'<\|im_end\|>',
        r'<\|user\|>',
        r'<\|assistant\|>',
        r'<think>.*?</think>',
        r'\[INST\].*?\[/INST\]',
        r'<<SYS>>.*?<</SYS>>',
    ]
    
    for pattern in special_markers:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    # 移除元数据括号内容
    metadata_patterns = [
        r'\(以下.*?\)',
        r'（以下.*?）',
        r'\(注[：:].*?\)',
        r'（注[：:].*?）',
        r'\([Nn]ote[：:].*?\)',
        r'\\Assistant.*?:',  # 反斜杠转义的Assistant
    ]
    
    for pattern in metadata_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    return text.strip()


def clean_language_contamination(text: str, target_language: str = "japanese") -> str:
    """
    清理语言污染，确保目标语言的纯净度
    注意：只清理明显的污染，不要过度清理
    """
    if target_language == "japanese":
        # 移除明显的纯中文短语（不包括日语汉字）
        # 只删除明显的中文解释性词汇
        chinese_only_phrases = [
            '分析如下', '建议如下', '回答如下', '问题是', '如果有机会',
            '这个地方', '那个地方', '什么时候', '怎么样', '为什么',  
        ]
        
        for phrase in chinese_only_phrases:
            text = text.replace(phrase, '')
        
        # 不再使用激进的正则删除，避免误删
        # 只删除明显是纯英文的长句（但保留日文内容）
        # 这里暂时禁用，因为太容易误删
        
    return text.strip()


def extract_first_valid_sentence(text: str, max_length: int = 512) -> str:
    """
    提取第一句有效的、干净的回复
    """
    # 按句子分割
    sentences = re.split(r'([。！？\n])', text)
    
    # 找到第一句有效的句子（长度合理且不是垃圾）
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
        else:
            sentence = sentences[i]
        
        sentence = sentence.strip()
        
        # 过滤条件
        if len(sentence) < 3:  # 太短
            continue
        if re.match(r'^[\s\*\-\.]+$', sentence):  # 只有符号
            continue
        if 'absence' in sentence.lower():  # 包含明显错误
            continue
        
        # 找到有效句子
        return sentence[:max_length]
    
    # 如果没找到，返回原文的前一部分
    return text[:max_length].strip()


def enhanced_clean_model_output(
    text: str,
    max_length: int = 512,
    is_japanese_task: bool = False,
    remove_english: bool = True,
    remove_repeats: bool = True,
    remove_emoji: bool = True,  # ✅ 新增：是否强制清除 emoji（默认True）
    debug: bool = False
) -> str:
    """
    增强版输出清理函数
    
    Args:
        text: 原始模型输出
        max_length: 最大输出长度
        is_japanese_task: 是否为日语任务
        remove_english: 是否移除英文（仅在日语任务中）
        remove_repeats: 是否移除重复
        remove_emoji: 是否强制清除 emoji（默认True，作为最后一道防线）
        debug: 是否打印调试信息
    
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    original_text = text
    if debug:
        print(f"[DEBUG] 原始输入: {text[:100]}...")
    
    # 1. 移除元数据和角色标识
    text = remove_metadata_and_roles(text)
    
    # 2. 基础清理（移除思考过程等）
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<think>', '').replace('</think>', '')
    
    # 3. 移除停止标记后的内容
    stop_markers = [
        r'\n（', r'\n\(',
        r'\n[*]{2,}',
        r'\n※',
        r'\n問題[：:]',
        r'\n最终生成',
        r'\n分析',
        r'\n建议',
        r'\n\s*---',
        r'\n[A-Z]\)',
        r'RealPersonaChat',
        r'\\Assistant',  # 特别处理
    ]
    
    combined_stop_pattern = '|'.join(stop_markers)
    match = re.search(combined_stop_pattern, text)
    if match:
        text = text[:match.start()]
    if debug:
        print(f"[DEBUG] Step 3 - 移除停止标记后: {repr(text[:100])}")
        print(f"[DEBUG] Step 3 - 长度: {len(text)}")
    
    # 4. 移除重复（在提取第一句之前）
    if remove_repeats:
        text = remove_duplicates(text)
    if debug:
        print(f"[DEBUG] Step 4 - 移除重复后: {repr(text[:100])}")
        print(f"[DEBUG] Step 4 - 长度: {len(text)}")
    
    # 5. 提取第一句有效内容
    text = extract_first_valid_sentence(text, max_length)
    if debug:
        print(f"[DEBUG] Step 5 - 提取第一句: {repr(text)}")
        print(f"[DEBUG] Step 5 - 长度: {len(text)}")
    
    # 6. 针对日语任务的特殊处理（在提取第一句之后）
    if is_japanese_task:
        if debug:
            print(f"[DEBUG] Step 6a - 开始日语处理, remove_english={remove_english}")
        # 移除英文单词
        if remove_english:
            before_remove = text
            text = remove_english_from_japanese(text)
            if debug:
                print(f"[DEBUG] Step 6b - 移除英文前: {before_remove}")
                print(f"[DEBUG] Step 6c - 移除英文后: {text}")
                print(f"[DEBUG] Step 6c - 包含experience: {'experience' in text}")
        
        # 移除中文和英文污染
        before_clean = text
        text = clean_language_contamination(text, "japanese")
        if debug:
            print(f"[DEBUG] Step 6d - 清理污染前: {before_clean}")
            print(f"[DEBUG] Step 6e - 清理污染后: {text}")
            print(f"[DEBUG] Step 6e - 包含experience: {'experience' in text}")
    
    # 7. 最后清理
    text = text.strip()
    if debug:
        print(f"[DEBUG] Step 7 - 最后清理: {text}")
        print(f"[DEBUG] Step 7 - text长度: {len(text)}")
    
    # 8. ✅ 清理 Unicode 替换字符（乱码 �）
    before_unicode_cleanup = text
    text = remove_unicode_replacement_chars(text)
    if debug and text != before_unicode_cleanup:
        print(f"[DEBUG] Step 8 - 清理乱码字符")
        print(f"[DEBUG] Step 8a - 清理前: {before_unicode_cleanup}")
        print(f"[DEBUG] Step 8b - 清理后: {text}")
    
    # 9. ✅ 清理过度重复的语气词
    before_interjection_removal = text
    text = remove_excessive_interjections(text)
    if debug and text != before_interjection_removal:
        print(f"[DEBUG] Step 9 - 清理语气词")
        print(f"[DEBUG] Step 9a - 清理前: {before_interjection_removal}")
        print(f"[DEBUG] Step 9b - 清理后: {text}")
    
    # 10. ✅ 强制清除 emoji（最后一道防线）
    if remove_emoji:
        before_emoji_removal = text
        had_emoji = contains_emoji(text)
        text = remove_all_emoji(text)
        if debug and had_emoji:
            print(f"[DEBUG] Step 10 - 检测到 emoji！")
            print(f"[DEBUG] Step 10a - 清除前: {before_emoji_removal}")
            print(f"[DEBUG] Step 10b - 清除后: {text}")
        elif debug:
            print(f"[DEBUG] Step 10 - 未检测到 emoji，跳过清除")
    
    # 11. 兜底：如果清理后太短，返回原文的一部分（也清除乱码、语气词和emoji）
    if len(text) < 3 and len(original_text) > 5:
        if debug:
            print(f"[DEBUG] Step 11 - 触发兜底逻辑！返回原文")
        fallback_text = original_text[:max_length].strip()
        # 即使是兜底逻辑，也要清除乱码、语气词和 emoji
        fallback_text = remove_unicode_replacement_chars(fallback_text)
        fallback_text = remove_excessive_interjections(fallback_text)
        if remove_emoji:
            fallback_text = remove_all_emoji(fallback_text)
        return fallback_text
    
    if debug:
        print(f"[DEBUG] Step 11 - 正常返回清理后的text")
    return text


def test_enhanced_clean():
    """测试增强版清理函数"""
    
    test_cases = [
        {
            "input": "おわあ、声優さんの視点からの舞台 experience 考えてみたら、よりプロフェッショナルな感じに聞こえますよね…",
            "expected": "おわあ、声優さんの視点からの舞台考えてみたら、よりプロフェッショナルな感じに聞こえますよね…",
            "desc": "移除英文单词"
        },
        {
            "input": "おはようございました！！\nおはようございました！！\nおはようございました！",
            "expected": "おはようございました！！",
            "desc": "移除重复句子"
        },
        {
            "input": "おわあ、千葉 really 熱かったですね…",
            "expected": "おわあ、千葉熱かったですね…",
            "desc": "移除日文中的英文"
        },
        {
            "input": "徳島県には阿波舞伎とかあるじゃないですか。それ食わないと損だよ！\\Assistant GS: よく耳にしましたけど...",
            "expected": "徳島県には阿波舞伎とかあるじゃないですか。",
            "desc": "移除Assistant标识"
        },
        {
            "input": "德島県には行ってみたいと思います。四国へのアクセスもわりわいなので、そちら方面へ行きたいなぁと思ったりします。",
            "expected": "德島県には行ってみたいと思います。",
            "desc": "提取第一句"
        },
        {
            "input": "谢谢你的帮助！❤️❤️❤️",
            "expected": "谢谢你的帮助！",
            "desc": "✅ 清除 emoji 表情符号"
        },
        {
            "input": "太棒了😊😊😊非常好👍👍",
            "expected": "太棒了非常好",
            "desc": "✅ 清除多个不同的 emoji"
        },
        {
            "input": "おはようございます😀✨",
            "expected": "おはようございます",
            "desc": "✅ 清除日文中的 emoji"
        }
    ]
    
    print("=" * 80)
    print("增强版清理函数测试")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        result = enhanced_clean_model_output(
            case["input"],
            max_length=512,
            is_japanese_task=True,
            remove_english=True,
            remove_repeats=True
        )
        
        print(f"\n测试 {i}: {case['desc']}")
        print(f"输入: {case['input'][:80]}...")
        print(f"输出: {result}")
        print(f"预期: {case['expected']}")
        print(f"通过: {'✅' if case['expected'] in result or result in case['expected'] else '❌'}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    test_enhanced_clean()
