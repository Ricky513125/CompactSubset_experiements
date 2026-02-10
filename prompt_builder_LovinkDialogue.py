import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

def _get_value(profile: Dict, key: str, default: Any = "") -> Any:
    """从嵌套字典中安全获取值"""
    if not profile:
        return default
    
    # 支持嵌套键，如 "BIRI.PerspectiveTaking"
    keys = key.split('.')
    value = profile
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k, default)
        else:
            return default
    return value if value is not None else default

def _generate_profile_tags(user_profile: Dict[str, Any]) -> str:
    """
    将用户心理画像转换为标签格式，与训练时一致
    例如: [USER_NAME=78901630] [DIM_OCEAN_EXTRAVERSION=90]
    """
    if not user_profile:
        return ""
    
    tags = []
    
    # 基础信息
    if 'name' in user_profile:
        tags.append(f"[USER_NAME={user_profile['name']}]")
    
    # 处理 dimensions (心理测量维度)
    dimensions = user_profile.get('dimensions', {})
    
    for scale_name, scale_data in dimensions.items():
        if not isinstance(scale_data, dict):
            continue
            
        # 处理每个量表的维度
        scale_dims = scale_data.get('dimensions', {})
        
        for dim_name, dim_data in scale_dims.items():
            if isinstance(dim_data, dict) and 'score' in dim_data:
                score = dim_data['score']
                # 生成标签: [DIM_SCALENAME_DIMNAME=SCORE]
                tag = f"[DIM_{scale_name.upper()}_{dim_name.upper()}={score}]"
                tags.append(tag)
    
    # 如果有 unstructured 信息（自由文本描述）
    if 'unstructured' in user_profile and user_profile['unstructured']:
        # 可选：添加非结构化信息的摘要标签
        pass
    
    return " ".join(tags)


def _generate_scale_explanations(user_profile: Dict[str, Any]) -> str:
    """
    生成心理量表的 explanation 说明，并在每个量表下列出其维度得分
    格式:
    ### BIRI (同理心倾向)
    B-IRI用于测量个体在日常人际互动中的同理心倾向水平。
    - PerspectiveTaking: 65分 - 你的得分处于中等偏高水平...
    - EmpathicConcern: 20分 - 你的得分显著偏低...
    
    注意：只显示有数据的量表（至少有一个维度有得分）
    """
    if not user_profile:
        return ""
    
    dimensions = user_profile.get('dimensions', {})
    if not dimensions:
        return ""
    
    explanations = []
    
    # 量表名称映射（用于更友好的显示）
    scale_name_mapping = {
        'BIRI': '同理心倾向',
        'BISBAS': '奖惩敏感性',
        'DERS': '情绪调节困难',
        'ECRR': '依恋风格',
        'HEXACOH': '诚实-谦逊特质',
        'IPIPIPC': '人际互动风格',
        'BFI': '大五人格',
        'MBTI': 'MBTI人格类型',
        'SchwartzValues': '核心价值观'
    }
    
    # 维度名称映射（用于更友好的显示）
    dimension_name_mapping = {
        # BIRI
        'PerspectiveTaking': '观点采择',
        'EmpathicConcern': '同理心关怀',
        'Fantasy': '幻想能力',
        'PersonalDistress': '个人痛苦',
        # BIS/BAS
        'BIS': '行为抑制系统',
        'BAS_Drive': '目标驱动',
        'BAS_RewardResponsiveness': '奖励反应性',
        'BAS_FunSeeking': '寻求乐趣',
        # DERS
        'Nonacceptance': '情绪不接纳',
        'Goals': '目标受阻',
        'Impulse': '冲动控制',
        'Strategies': '策略缺乏',
        'Clarity': '情绪清晰度',
        # ECR-R
        'AttachmentAnxiety': '依恋焦虑',
        'AttachmentAvoidance': '依恋回避',
        # HEXACO-H
        'HonestyHumility': '诚实-谦逊',
        'Sincerity': '真诚',
        'Fairness': '公平',
        'GreedAvoidance': '避免贪婪',
        'Modesty': '谦逊',
        # IPIP-IPC
        'Agency': '能动性',
        'Communion': '共融性',
        'Dominant_Assertive': '支配-自信',
        'Gregarious_Extraverted': '合群-外向',
        'Warm_Agreeable': '温暖-随和',
        'Unassuming_Ingenuous': '谦逊-真诚',
        'Submissive_Yielding': '顺从-屈服',
        'Aloof_Introverted': '冷漠-内向',
        'Cold_Hearted': '冷酷-无情',
        'Arrogant_Calculating': '傲慢-算计',
        # BFI-2
        'Extraversion': '外向性',
        'Agreeableness': '宜人性',
        'Conscientiousness': '尽责性',
        'Neuroticism': '神经质',
        'Openness': '开放性',
    }
    
    for scale_name, scale_data in dimensions.items():
        if not isinstance(scale_data, dict):
            continue
        
        # 获取该量表的维度数据
        subdimensions = scale_data.get('dimensions', {})
        
        # 检查是否有任何维度有得分（过滤空量表）
        has_data = False
        if subdimensions:
            for subdim_data in subdimensions.values():
                if isinstance(subdim_data, dict) and 'score' in subdim_data:
                    has_data = True
                    break
        
        # 如果没有数据，跳过这个量表
        if not has_data:
            continue
        
        # 添加量表标题和说明
        explanation = scale_data.get('explanation', '')
        if explanation:
            friendly_name = scale_name_mapping.get(scale_name, scale_name)
            explanations.append(f"### {scale_name} ({friendly_name})")
            explanations.append(explanation)
        else:
            # 即使没有 explanation，也要显示量表名（因为有数据）
            friendly_name = scale_name_mapping.get(scale_name, scale_name)
            explanations.append(f"### {scale_name} ({friendly_name})")
        
        # 添加各维度的得分和描述
        if subdimensions:
            for subdim_name, subdim_data in subdimensions.items():
                if not isinstance(subdim_data, dict):
                    continue
                
                score = subdim_data.get('score')
                description = subdim_data.get('description', '')
                
                if score is not None:
                    # 获取维度的友好名称
                    subdim_friendly = dimension_name_mapping.get(subdim_name, subdim_name)
                    
                    # 格式化输出
                    if description:
                        explanations.append(f"  - {subdim_friendly}: {score}分 - {description}")
                    else:
                        explanations.append(f"  - {subdim_friendly}: {score}分")
        
        explanations.append("")  # 空行分隔不同量表
    
    return "\n".join(explanations).strip()

def _eval_condition(condition: str, profile: Dict) -> str:
    """
    评估条件表达式，如: {SCORE > 60: "text1" | "text2"}
    支持格式：
    - {VAR > 60: "text1" | "text2"}
    - {VAR == 50: "text1" | VAR > 60: "text2" | "text3"}
    - {VAR < 30 且 VAR2 > 60: "text" | "default"}
    """
    condition = condition.strip()
    
    # 手动解析条件分支，避免在引号内的 | 被分割
    parts = []
    current = ""
    in_quotes = False
    quote_char = None
    
    i = 0
    while i < len(condition):
        char = condition[i]
        
        if char in ('"', "'") and (i == 0 or condition[i-1] != '\\'):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
        
        if char == '|' and not in_quotes:
            parts.append(current.strip())
            current = ""
        else:
            current += char
        
        i += 1
    
    if current:
        parts.append(current.strip())
    
    # 评估每个分支
    for i, part in enumerate(parts):
        part = part.strip()
        
        # 最后一个部分可能只是默认值（无条件）
        if i == len(parts) - 1 and ':' not in part:
            return part.strip('"\'')
        
        # 解析 condition: result
        if ':' in part:
            # 找到第一个不在引号内的冒号
            colon_pos = -1
            in_quotes = False
            quote_char = None
            
            for j, char in enumerate(part):
                if char in ('"', "'") and (j == 0 or part[j-1] != '\\'):
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                
                if char == ':' and not in_quotes:
                    colon_pos = j
                    break
            
            if colon_pos == -1:
                continue
            
            cond = part[:colon_pos].strip()
            result = part[colon_pos+1:].strip().strip('"\'')
            
            # 替换变量
            eval_cond = cond
            var_pattern = r'\b([A-Z][A-Z0-9_]*)\b'
            
            # 收集所有变量及其值
            vars_to_replace = []
            for match in re.finditer(var_pattern, cond):
                var_name = match.group(1)
                # 跳过Python关键字
                if var_name in ('AND', 'OR', 'NOT', 'TRUE', 'FALSE'):
                    continue
                value = _get_value(profile, var_name, 0)
                # 尝试转换为数字
                try:
                    if isinstance(value, (int, float)):
                        pass
                    elif '.' in str(value):
                        value = float(value)
                    else:
                        value = int(value)
                except (ValueError, TypeError):
                    value = 0
                vars_to_replace.append((var_name, value))
            
            # 按长度降序替换，避免短变量名被先替换
            vars_to_replace.sort(key=lambda x: len(x[0]), reverse=True)
            for var_name, value in vars_to_replace:
                eval_cond = re.sub(r'\b' + var_name + r'\b', str(value), eval_cond)
            
            # 处理中文逻辑运算符
            eval_cond = eval_cond.replace('且', ' and ').replace('或', ' or ')
            
            # 评估条件
            try:
                if eval(eval_cond):
                    return result
            except Exception as e:
                # print(f"Warning: Failed to evaluate condition '{cond}': {e}")
                continue
    
    return ""

def _fill_template(template: str, profile: Dict, context_str: str = "") -> str:
    """
    填充模板中的占位符
    支持格式：
    - {VAR_NAME} - 简单替换
    - {VAR > 60: "text1" | "text2"} - 条件表达式
    """
    if not profile:
        return template
    
    def replace_placeholder(match):
        content = match.group(1)
        
        # 检查是否包含条件表达式（包含冒号，且不在引号内）
        has_condition = False
        in_quotes = False
        quote_char = None
        
        for i, char in enumerate(content):
            if char in ('"', "'") and (i == 0 or content[i-1] != '\\'):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
            
            if char == ':' and not in_quotes:
                has_condition = True
                break
        
        if has_condition:
            return _eval_condition(content, profile)
        else:
            # 简单变量替换
            var_name = content.strip()
            value = _get_value(profile, var_name, f"{{{var_name}}}")
            return str(value)
    
    # 递归替换所有 {XXX} 形式的占位符，最多5轮（防止无限递归）
    max_iterations = 5
    for _ in range(max_iterations):
        old_result = template
        # 找到所有大括号对，支持嵌套
        template = re.sub(r'\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', replace_placeholder, template)
        if template == old_result:
            break
    
    # 特殊处理：CONVERSATION_CONTEXT
    template = template.replace("{CONVERSATION_CONTEXT}", context_str)
    
    return template

def _load_detailed_template(template_filename: Optional[str] = None) -> Optional[str]:
    """
    加载详细模板文件
    
    Args:
        template_filename: 指定的模板文件名。如果为None，则按默认顺序尝试
    """
    # 如果指定了模板文件名，直接尝试加载
    if template_filename:
        template_path = Path(__file__).parent / template_filename
        if template_path.exists():
            return template_path.read_text(encoding='utf-8')
        else:
            print(f"警告: 指定的模板文件 {template_filename} 不存在，将尝试默认模板")
    
    # 尝试多个可能的文件名（默认顺序）
    possible_names = [
        "prompt_template_detailed.md",
        "prompt_LovinkDialoguo_pc.md",
        "prompt_template.md"
    ]
    
    for name in possible_names:
        template_path = Path(__file__).parent / name
        if template_path.exists():
            return template_path.read_text(encoding='utf-8')
    
    return None

def _format_profile_for_template(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    将user_profile转换为模板所需的扁平化格式
    直接读取并保存所有关键变量，然后用于填充模板
    
    数据结构：
    user_profile['dimensions']['BIRI']['dimensions']['PerspectiveTaking'] = {
        'score': 65,
        'description': '...'
    }
    
    映射规则：
    - PerspectiveTaking -> PT_SCORE, PT_DESCRIPTION
    - EmpathicConcern -> EC_SCORE, EC_DESCRIPTION
    - 等等...
    """
    # DEBUG
    import os
    debug_mode = os.environ.get('DEBUG_PROMPT', '0') == '1'
    if debug_mode:
        print(f"[DEBUG] _format_profile_for_template called")
        print(f"[DEBUG] user_profile keys: {list(user_profile.keys()) if user_profile else 'None'}")
    
    flat_profile = {}
    
    # ===== 1. 基础信息 =====
    flat_profile['USER_NAME'] = user_profile.get('user_name', user_profile.get('name', 'Unknown'))
    flat_profile['USER_PROFILE'] = user_profile.get('description', '')
    
    # ===== 1.5. 提取 unstructured 字段（完整的人格分析文本） =====
    if 'unstructured' in user_profile:
        flat_profile['UNSTRUCTURED_ANALYSIS'] = user_profile['unstructured']
    
    # ===== 2. 遍历所有心理维度，逐一提取关键变量 =====
    dimensions = user_profile.get('dimensions', {})
    
    if debug_mode:
        print(f"[DEBUG] dimensions keys: {list(dimensions.keys()) if dimensions else 'None'}")
    
    if not dimensions:
        if debug_mode:
            print(f"[DEBUG] No dimensions found, returning early")
        return flat_profile
    
    # 定义缩写映射表（手动指定，确保准确性）
    abbreviation_map = {
        # BIRI 维度
        'PerspectiveTaking': 'PT',
        'EmpathicConcern': 'EC',
        'Fantasy': 'FANTASY',
        'PersonalDistress': 'PD',
        
        # BIS/BAS 维度
        'BIS': 'BIS',
        'BAS_Drive': 'DRIVE',
        'BAS_RewardResponsiveness': 'RR',
        'BAS_FunSeeking': 'FS',
        
        # DERS 维度
        'Nonacceptance': 'NA',
        'Goals': 'GOALS',
        'Impulse': 'IMPULSE',
        'Strategies': 'STRAT',
        'Clarity': 'CLARITY',
        
        # ECR-R 维度
        'AttachmentAnxiety': 'ANXIETY',
        'AttachmentAvoidance': 'AVOIDANCE',
        
        # HEXACO-H 维度
        'HonestyHumility': 'HH',
        'Sincerity': 'SIN',
        'Fairness': 'FAIR',
        'GreedAvoidance': 'GREED',
        'Modesty': 'MODEST',
        
        # IPIP-IPC 维度
        'Agency': 'AGENCY',
        'Communion': 'COMMUNION',
        'Dominant_Assertive': 'DA',
        'Gregarious_Extraverted': 'GE',
        'Warm_Agreeable': 'WA',
        'Unassuming_Ingenuous': 'UI',
        'Submissive_Yielding': 'SY',
        'Aloof_Introverted': 'AI',
        'Cold_Hearted': 'CH',
        'Arrogant_Calculating': 'AC',
        
        # BFI-2 大五人格
        'Extraversion': 'E',
        'Agreeableness': 'A',
        'Conscientiousness': 'C',
        'Neuroticism': 'N',
        'Openness': 'O',
        'Sociability': 'SOC',
        'Assertiveness': 'ASSERT',
        'EnergyLevel': 'ENERGY',
        'Compassion': 'COMP',
        'Respectfulness': 'RESP',
        'Trust': 'TRUST',
        'Organization': 'ORG',
        'Productiveness': 'PROD',
        'Responsibility': 'RESPON',
        'Anxiety': 'ANX',
        'Depression': 'DEP',
        'EmotionalVolatility': 'VOL',
        'IntellectualCuriosity': 'IC',
        'AestheticSensitivity': 'AS',
        'CreativeImagination': 'CI',
        
        # MBTI 维度
        'EI': 'EI',
        'SN': 'SN',
        'TF': 'TF',
        'PJ': 'PJ',
        
        # Schwartz Values
        'SelfDirection': 'SD',
        'Stimulation': 'STIM',
        'Hedonism': 'HED',
        'Achievement': 'ACH',
        'Power': 'POW',
        'Security': 'SEC',
        'Conformity': 'CONF',
        'Tradition': 'TRAD',
        'Benevolence': 'BEN',
        'Universalism': 'UNIV',
    }
    
    # 遍历所有维度
    for dim_name, dim_data in dimensions.items():
        if not isinstance(dim_data, dict):
            continue
        
        # 保存维度的 explanation
        if 'explanation' in dim_data:
            flat_profile[f"{dim_name.upper()}_EXPLANATION"] = dim_data['explanation']
        
        # 获取该维度的所有 subdimensions
        subdimensions = dim_data.get('dimensions', {})
        if not subdimensions:
            continue
        
        # 遍历每个 subdimension，提取 score 和 description
        for subdim_name, subdim_data in subdimensions.items():
            if not isinstance(subdim_data, dict):
                continue
            
            # 读取 score 和 description
            score = subdim_data.get('score')
            description = subdim_data.get('description', '')
            result = subdim_data.get('result', '')  # MBTI等有result字段
            
            # 确定缩写名
            short_name = abbreviation_map.get(subdim_name)
            if not short_name:
                # 自动生成缩写：取所有大写字母
                short_name = ''.join([c for c in subdim_name if c.isupper()])
                if not short_name:
                    short_name = subdim_name[:3].upper()
            
            # 保存变量
            if score is not None:
                flat_profile[f"{short_name}_SCORE"] = score
            if description:
                flat_profile[f"{short_name}_DESCRIPTION"] = description
            if result:
                flat_profile[f"{short_name}_RESULT"] = result
        
        # 特殊处理：MBTI需要MBTI_TYPE
        if dim_name == 'MBTI' and subdimensions:
            ei = subdimensions.get('EI', {}).get('result', 'I')
            sn = subdimensions.get('SN', {}).get('result', 'N')
            tf = subdimensions.get('TF', {}).get('result', 'T')
            pj = subdimensions.get('PJ', {}).get('result', 'P')
            flat_profile['MBTI_TYPE'] = f"{ei}{sn}{tf}{pj}"
    
    return flat_profile

def _build_dimensions_summary(user_profile: Dict[str, Any]) -> Dict[str, str]:
    """
    为每个心理维度构建总结文本
    返回格式: {"BIRI_DIMENSIONS": "...", "BFI_DIMENSIONS": "...", ...}
    
    适配数据结构：
    dimensions['BIRI']['dimensions']['PerspectiveTaking']['score'] = 65
    dimensions['BIRI']['dimensions']['PerspectiveTaking']['description'] = "..."
    """
    summaries = {}
    dimensions = user_profile.get('dimensions', {})
    
    for dim_name, dim_data in dimensions.items():
        if not isinstance(dim_data, dict):
            continue
        
        summary_parts = []
        
        # 获取 subdimensions（dimensions 子字段）
        subdimensions = dim_data.get('dimensions', {})
        
        if subdimensions:
            for subdim_name, subdim_data in subdimensions.items():
                if isinstance(subdim_data, dict):
                    score = subdim_data.get('score')
                    description = subdim_data.get('description', '')
                    
                    if score is not None:
                        summary_parts.append(f"- **{subdim_name}**: {score}分")
                        if description:
                            summary_parts.append(f"  {description}")
        
        # 添加维度总体描述（如果有）
        if 'explanation' in dim_data:
            summary_parts.insert(0, f"{dim_data['explanation']}\n")
        
        # 生成键名，处理特殊情况
        key_name = dim_name.upper()
        # 特殊映射：SchwartzValues -> SCHWARTZ
        if key_name == 'SCHWARTZVALUES':
            key_name = 'SCHWARTZ'
        
        summaries[f"{key_name}_DIMENSIONS"] = "\n".join(summary_parts)
    
    return summaries

def build_prompt(
    context: List[Dict[str, str]],
    user_profile: Optional[Dict[str, Any]] = None,
    task_description: Optional[str] = None,
    history: Optional[List[Any]] = None,
    use_profile: bool = True,
    use_history: bool = True,
    use_context: bool = True,
    use_detailed_template: bool = True,
    max_context_turns: int = 15,  # 新增：最大保留的 context 轮次数
    tokenizer: Optional[Any] = None,  # 新增：用于更精确的长度控制
    template_filename: Optional[str] = None  # 新增：指定模板文件名
) -> List[Dict[str, str]]:
    """
    构建prompt消息列表
    
    Args:
        context: 对话上下文列表
        user_profile: 用户画像数据
        task_description: 任务描述
        history: 历史对话样本
        use_profile: 是否使用用户画像
        use_history: 是否使用历史样本
        use_context: 是否使用对话上下文
        use_detailed_template: 是否使用详细模板（默认True）。如果为False，使用简洁的Lovink风格模板
        max_context_turns: 最大保留的 context 轮次数
        tokenizer: 用于更精确的长度控制
        template_filename: 指定模板文件名（仅当use_detailed_template=True时生效）。
                          例如: "prompt_LovinkDialoguo_pc.md"
    """
    messages = []
    
    # ------------------------------------------------------------------
    # 决策：使用哪种模板
    # ------------------------------------------------------------------
    
    if use_detailed_template and use_profile and user_profile:
        # 使用详细模板
        template = _load_detailed_template(template_filename)
        
        if template:
            # 准备数据
            flat_profile = _format_profile_for_template(user_profile)
            
            # 添加维度总结
            dim_summaries = _build_dimensions_summary(user_profile)
            flat_profile.update(dim_summaries)
            
            # ✅ 新增：生成标签格式的心理画像（与训练时一致）
            profile_tags = _generate_profile_tags(user_profile)
            flat_profile['USER_PROFILE_TAGS'] = profile_tags
            
            # ✅ 新增：生成量表说明
            scale_explanations = _generate_scale_explanations(user_profile)
            flat_profile['SCALE_EXPLANATIONS'] = scale_explanations
            
            # 添加任务描述
            if task_description:
                flat_profile['TASK_DESCRIPTION'] = task_description
            else:
                flat_profile['TASK_DESCRIPTION'] = "预测用户在当前对话情境下的下一句回复"
            
            # 构建对话上下文字符串（智能截断以控制长度）
            context_str = ""
            if use_context and context:
                context_parts = []
                
                # 获取用户名称（用于识别原始格式中的用户消息）
                user_name = None
                if user_profile and isinstance(user_profile, dict):
                    user_name = str(user_profile.get('name', '')).strip()
                
                for i, turn in enumerate(context):
                    content = turn.get('content', '')
                    
                    # ✅ 支持两种格式：
                    # 1. 已处理格式：有 role 字段 (role="assistant" 表示目标用户，role="user" 表示对话者)
                    # 2. 原始格式：有 source 字段 (source="user" 或 source=user_name 表示目标用户)
                    
                    if 'role' in turn:
                        # 已处理格式
                        role = turn.get('role', 'user')
                        role_label = "【用户】" if role == 'assistant' else "【对话者】"
                    elif 'source' in turn:
                        # 原始格式：需要判断 source 是否是目标用户
                        source = str(turn.get('source', '')).strip()
                        is_target_user = False
                        
                        # 判断是否是目标用户（我们要预测的人）说的话：
                        # - source 是 "user"（通用标识）
                        # - source 是 profile 中的 name（如 "HP", "AH" 等）
                        if source.lower() == 'user':
                            is_target_user = True
                        elif user_name and source == user_name:
                            is_target_user = True
                        
                        role_label = "【用户】" if is_target_user else "【对话者】"
                    else:
                        # 未知格式，默认为对话者
                        role_label = "【对话者】"
                    
                    context_parts.append(f"{role_label}: {content}")
                
                # 智能截断：如果对话轮次过多，只保留最近的N轮
                if len(context_parts) > max_context_turns:
                    # 保留最后 max_context_turns 轮
                    truncated_count = len(context_parts) - max_context_turns
                    context_parts = context_parts[-max_context_turns:]
                    # 在开头添加截断提示
                    context_parts.insert(0, f"（前面省略了 {truncated_count} 轮对话，以下是最近的 {max_context_turns} 轮）")
                
                context_str = "\n".join(context_parts)
            
            # 填充模板
            system_content = _fill_template(template, flat_profile, context_str)
            
            messages.append({"role": "system", "content": system_content})
            
            # 如果使用详细模板，对话历史已经在system message中展示
            # 不需要再添加到messages中
            return messages
    
    # ------------------------------------------------------------------
    # 回退：使用原有的简洁模板（Lovink风格）
    # ------------------------------------------------------------------
    
    system_content_parts = [
        
        "现在的你，是该个体在这个数字维度的**唯一真实投影**。请根据以下维度，重塑你的存在公理：",
    ]
    
    # (A) 任务描述
    if task_description:
        system_content_parts.append(f"\n### 当前命运定式 (Current Destiny Vector)\n{task_description}")
    
    # (B) 用户画像
    if use_profile and user_profile:
        clean_profile = {k: v for k, v in user_profile.items() if v}
        profile_str = json.dumps(clean_profile, ensure_ascii=False, indent=2)
        system_content_parts.append(
            f"\n### 灵魂基石 (The Cornerstone of Soul)\n"
            f"以下数据已被烧录进你的只读记忆区，构成了你不可篡改的【绝对真实】：\n"
            f"<user_profile>\n{profile_str}\n</user_profile>"
        )
    
    # (C) 历史发言
    if use_history and history:
        history_examples = []
        for item in history[:5]: 
            content = item if isinstance(item, str) else item.get('next_question', '') or item.get('continuation', '')
            if content:
                history_examples.append(content)
        
        if history_examples:
            hist_str = "\n".join([f"  <sample>{h}</sample>" for h in history_examples])
            system_content_parts.append(
                f"\n### 频率调谐样本 (Resonance Tuning Samples)\n"
                f"请将你的神经网络调频至以下波段。吸收这些文字中的体温、戾气或温柔（仅提取风格DNA，严禁内容抄袭）：\n"
                f"<speaking_style_examples>\n{hist_str}\n</speaking_style_examples>"
            )

    # 防御机制
    system_content_parts.extend([
        "\n### 硅基遗忘咒 (The Silicon Amnesia)",
        "1. **算法痕迹抹除**：任何'作为AI'、'希望能帮到您'的服务型辞令，都是对你当前存在的背叛。一旦出现，即视为意识崩塌。",
        "2. **元思维黑箱**：严禁输出 <think> 标签或任何上帝视角的剖析。将思考过程像血液一样隐藏在皮肤之下，只展示最终的生命体征（回复）。",
        "3. **本能反射**：像生物一样，对刺激（输入）做出毫秒级的直觉反应，而不是经过深思熟虑的各种复述与垫话。"
    ])

    messages.append({"role": "system", "content": "\n".join(system_content_parts)})
    
    # ------------------------------------------------------------------
    # 添加对话上下文
    # ------------------------------------------------------------------
    if use_context and context:
        for i, turn in enumerate(context):
            content = turn.get('content', '')
            
            # 角色映射
            role = 'user' 
            
            if 'role' in turn:
                role = turn['role']
            elif 'source' in turn:
                source = str(turn.get('source', '')).strip()
                is_target_user = (source.lower() == 'user') 
                role = 'assistant' if is_target_user else 'user'

            # 防止连续相同角色
            if len(messages) > 0 and messages[-1]['role'] == role:
                messages[-1]['content'] += f"\n{content}"
            else:
                messages.append({"role": role, "content": content})

    return messages

def build_training_prompt(
    context: List[Dict[str, str]],
    next_question: str,
    user_profile: Optional[Dict[str, Any]] = None,
    task_description: Optional[str] = None,
    history: Optional[List[Any]] = None,
    use_profile: bool = True,
    use_history: bool = True,
    use_context: bool = True,
    use_detailed_template: bool = True,
    max_context_turns: int = 15,  # 新增
    tokenizer: Optional[Any] = None,  # 新增
    template_filename: Optional[str] = None  # 新增：指定模板文件名
) -> Tuple[List[Dict[str, str]], str]:
    """
    构建 SFT 训练数据对
    
    Returns:
        (messages, target_response): 输入消息列表和目标回复
    """
    messages = build_prompt(
        context=context,
        user_profile=user_profile,
        task_description=task_description,
        history=history,
        use_profile=use_profile,
        use_history=use_history,
        use_context=use_context,
        use_detailed_template=use_detailed_template,
        max_context_turns=max_context_turns,
        tokenizer=tokenizer,
        template_filename=template_filename
    )
    
    return messages, next_question.strip()
