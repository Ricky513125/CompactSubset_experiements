"""
日语文本规范化工具
修正常见的中英文混用问题
"""
import re

# 简体中文到日语繁体的映射（常见汉字）
CHINESE_TO_JAPANESE = {
    # 常见简体字 → 日语繁体/正字
    '桥': '橋',
    '户': '戸',
    '国': '国',  # 注意：日语也用"国"
    '岛': '島',
    '湾': '湾',  # 日语也用"湾"
    '门': '門',
    '车': '車',
    '边': '辺',
    '东': '東',
    '西': '西',
    '南': '南',
    '北': '北',
    '关': '関',
    '连': '連',
    '络': '絡',
    '线': '線',
    '铁': '鉄',
    '电': '電',
    '话': '話',
    '说': '説',
    '对': '対',
    '时': '時',
    '间': '間',
    '问': '問',
    '题': '題',
    '学': '学',  # 日语也用"学"
    '会': '会',  # 日语也用"会"
    '场': '場',
    '馆': '館',
    '饭': '飯',
    '厅': '廳',
    '楼': '楼',  # 日语也用"楼"
    '层': '階',
    '号': '号',  # 日语也用"号"
    '码': '碼',
    '买': '買',
    '卖': '売',
    '价': '価',
    '钱': '銭',
    '币': '幣',
    '银': '銀',
    '行': '行',  # 日语也用"行"
    '医': '医',  # 日语也用"医"
    '院': '院',  # 日语也用"院"
    '药': '薬',
    '店': '店',  # 日语也用"店"
    '书': '書',
    '图': '図',
    '馆': '館',
    '阅': '閲',
    '读': '読',
    '写': '書',
    '听': '聴',
    '观': '観',
    '视': '視',
    '见': '見',
    '现': '現',
    '实': '実',
    '际': '際',
    '经': '経',
    '验': '験',
    '历': '歴',
    '史': '史',  # 日语也用"史"
    '记': '記',
    '录': '録',
    '报': '報',
    '纸': '紙',
    '杂': '雑',
    '志': '誌',
    '传': '伝',
    '统': '統',
    '继': '継',
    '续': '続',
    '结': '結',
    '构': '構',
    '组': '組',
    '织': '織',
    '团': '団',
    '队': '隊',
    '员': '員',
}

# 英文数字词到日语的映射
ENGLISH_NUMBERS_TO_JAPANESE = {
    'one': '一',
    'two': '二',
    'three': '三',
    'four': '四',
    'five': '五',
    'six': '六',
    'seven': '七',
    'eight': '八',
    'nine': '九',
    'ten': '十',
}

# 常见英文单词到日语的映射
ENGLISH_TO_JAPANESE = {
    'tourist': '観光客',
    'tourism': '観光',
    'experience': '経験',
    'professional': 'プロフェッショナル',
    'station': '駅',
    'hotel': 'ホテル',
    'restaurant': 'レストラン',
    'cafe': 'カフェ',
    'coffee': 'コーヒー',
    'tea': '茶',
    'shop': '店',
    'store': '店',
    'market': '市場',
    'park': '公園',
    'museum': '美術館',
    'temple': '寺',
    'shrine': '神社',
    'castle': '城',
    'bridge': '橋',
    'river': '川',
    'mountain': '山',
    'sea': '海',
    'lake': '湖',
    'island': '島',
    'city': '市',
    'town': '町',
    'village': '村',
    # 添加更多常见词
    'temperature': '温度',
    'sleeve': '袖',
    'season': '季節',
    'weather': '天気',
    'time': '時間',
    'day': '日',
    'week': '週',
    'month': '月',
    'year': '年',
    'morning': '朝',
    'afternoon': '午後',
    'evening': '夕方',
    'night': '夜',
    'today': '今日',
    'tomorrow': '明日',
    'yesterday': '昨日',
    'people': '人々',
    'person': '人',
    'friend': '友達',
    'family': '家族',
    'food': '食べ物',
    'drink': '飲み物',
    'water': '水',
    'place': '場所',
    'room': '部屋',
    'house': '家',
    'building': '建物',
    'street': '通り',
    'road': '道',
    'car': '車',
    'bus': 'バス',
    'train': '電車',
    'plane': '飛行機',
    'book': '本',
    'movie': '映画',
    'music': '音楽',
    'art': '芸術',
    'sport': 'スポーツ',
    'game': 'ゲーム',
    'work': '仕事',
    'study': '勉強',
    'school': '学校',
    'university': '大学',
    'company': '会社',
}


def english_to_katakana(word: str) -> str:
    """
    将英文单词转换为片假名（简化版）
    对于未知的英文单词，转换成片假名是日语的标准做法
    """
    # 简化的英文到片假名映射
    katakana_map = {
        'a': 'ア', 'i': 'イ', 'u': 'ウ', 'e': 'エ', 'o': 'オ',
        'ka': 'カ', 'ki': 'キ', 'ku': 'ク', 'ke': 'ケ', 'ko': 'コ',
        'sa': 'サ', 'shi': 'シ', 'su': 'ス', 'se': 'セ', 'so': 'ソ',
        'ta': 'タ', 'chi': 'チ', 'tsu': 'ツ', 'te': 'テ', 'to': 'ト',
        'na': 'ナ', 'ni': 'ニ', 'nu': 'ヌ', 'ne': 'ネ', 'no': 'ノ',
        'ha': 'ハ', 'hi': 'ヒ', 'fu': 'フ', 'he': 'ヘ', 'ho': 'ホ',
        'ma': 'マ', 'mi': 'ミ', 'mu': 'ム', 'me': 'メ', 'mo': 'モ',
        'ya': 'ヤ', 'yu': 'ユ', 'yo': 'ヨ',
        'ra': 'ラ', 'ri': 'リ', 'ru': 'ル', 're': 'レ', 'ro': 'ロ',
        'wa': 'ワ', 'wo': 'ヲ', 'n': 'ン',
        'ga': 'ガ', 'gi': 'ギ', 'gu': 'グ', 'ge': 'ゲ', 'go': 'ゴ',
        'za': 'ザ', 'ji': 'ジ', 'zu': 'ズ', 'ze': 'ゼ', 'zo': 'ゾ',
        'da': 'ダ', 'di': 'ヂ', 'du': 'ヅ', 'de': 'デ', 'do': 'ド',
        'ba': 'バ', 'bi': 'ビ', 'bu': 'ブ', 'be': 'ベ', 'bo': 'ボ',
        'pa': 'パ', 'pi': 'ピ', 'pu': 'プ', 'pe': 'ペ', 'po': 'ポ',
    }
    
    # 对于复杂的英文单词，使用简化规则
    word_lower = word.lower()
    
    # 常见模式的直接映射
    direct_map = {
        'sleeve': 'スリーブ',
        'temperature': 'テンペラチャー',
        'experience': 'エクスペリエンス',
        'professional': 'プロフェッショナル',
        'service': 'サービス',
        'system': 'システム',
        'computer': 'コンピューター',
        'internet': 'インターネット',
        'message': 'メッセージ',
        'image': 'イメージ',
        'chance': 'チャンス',
        'change': 'チェンジ',
        'challenge': 'チャレンジ',
        'member': 'メンバー',
        'user': 'ユーザー',
        'level': 'レベル',
        'style': 'スタイル',
        'sense': 'センス',
        'balance': 'バランス',
        'point': 'ポイント',
        'percent': 'パーセント',
    }
    
    if word_lower in direct_map:
        return direct_map[word_lower]
    
    # 如果没有直接映射，保持原样（避免错误转换）
    return word


def normalize_japanese_text(text: str, fix_chinese: bool = True, fix_english: bool = True, convert_unknown_to_katakana: bool = True) -> str:
    """
    规范化日语文本，修正中英文混用问题
    
    Args:
        text: 输入文本
        fix_chinese: 是否修正简体中文汉字
        fix_english: 是否修正英文单词
        convert_unknown_to_katakana: 是否将未知英文单词转换为片假名
    
    Returns:
        规范化后的文本
    """
    if not text:
        return text
    
    result = text
    
    # 1. 修正英文数字词
    if fix_english:
        for eng, jpn in ENGLISH_NUMBERS_TO_JAPANESE.items():
            # 在日语环境中，英文单词前后可能是日语字符，不能用\b
            # 改用更宽松的匹配：前后不是字母
            pattern = r'(?<![a-zA-Z])' + eng + r'(?![a-zA-Z])'
            result = re.sub(pattern, jpn, result, flags=re.IGNORECASE)
    
    # 2. 修正常见英文单词（只替换明显的错误）
    if fix_english:
        # 首先转换已知单词（包括前后的空格）
        for eng, jpn in ENGLISH_TO_JAPANESE.items():
            # 匹配英文单词及其前后的空格
            pattern = r'\s*(?<![a-zA-Z])' + eng + r'(?![a-zA-Z])\s*'
            result = re.sub(pattern, jpn, result, flags=re.IGNORECASE)
        
        # 然后转换未知的英文单词为片假名
        if convert_unknown_to_katakana:
            def replace_unknown_english(match):
                word = match.group(0).strip()  # 去掉空格
                # 如果是保留词（大写缩写等），不转换
                if len(word) <= 2 and word.isupper():
                    return word
                # 转换为片假名
                return english_to_katakana(word)
            
            # 匹配剩余的英文单词（3个字母以上），包括前后空格
            result = re.sub(r'\s*(?<![a-zA-Z])[a-zA-Z]{3,}(?![a-zA-Z])\s*', replace_unknown_english, result)
    
    # 3. 修正简体中文汉字
    if fix_chinese:
        for cn, jp in CHINESE_TO_JAPANESE.items():
            result = result.replace(cn, jp)
    
    # 4. 清理多余的空格（替换后可能产生）
    result = re.sub(r'\s{2,}', ' ', result)  # 多个空格变成一个
    result = re.sub(r'\s+([。、！？，])', r'\1', result)  # 标点前的空格删除
    
    return result


def detect_mixed_characters(text: str) -> dict:
    """
    检测文本中的混用字符
    
    Returns:
        {
            'has_chinese': bool,
            'has_english_words': bool,
            'chinese_chars': list,
            'english_words': list
        }
    """
    result = {
        'has_chinese': False,
        'has_english_words': False,
        'chinese_chars': [],
        'english_words': []
    }
    
    # 检测简体中文字符
    for cn in CHINESE_TO_JAPANESE.keys():
        if cn in text:
            result['has_chinese'] = True
            result['chinese_chars'].append(cn)
    
    # 检测英文单词
    for eng in list(ENGLISH_NUMBERS_TO_JAPANESE.keys()) + list(ENGLISH_TO_JAPANESE.keys()):
        pattern = r'\b' + eng + r'\b'
        if re.search(pattern, text, flags=re.IGNORECASE):
            result['has_english_words'] = True
            result['english_words'].append(eng)
    
    return result


def test_normalizer():
    """测试规范化函数"""
    test_cases = [
        "本州four国連络桥って結構長いですよね。",
        "本州four国連络桥、確かに瀬户内海側から見ると結構大きな構造物ですよね。",
        "外国人touristも多かったですよね。",
        "それはprofessionalな感じですね。",
    ]
    
    print("=" * 80)
    print("日语文本规范化测试")
    print("=" * 80)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"原文: {text}")
        
        # 检测混用
        detection = detect_mixed_characters(text)
        if detection['has_chinese']:
            print(f"检测到简体中文: {detection['chinese_chars']}")
        if detection['has_english_words']:
            print(f"检测到英文单词: {detection['english_words']}")
        
        # 规范化
        normalized = normalize_japanese_text(text)
        print(f"修正: {normalized}")
        
        if text != normalized:
            print("✓ 已修正")
        else:
            print("✗ 无需修正")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_normalizer()
