"""
safety.py —— 内容安全过滤
==========================
出海产品必须过审。基于关键词 + 简单规则检测敏感内容。
生产环境应接入专业审核 API（如阿里云内容安全、Azure Content Safety）。
"""
import re


# 敏感词列表（简化版，生产环境应使用专业词库）
BLACKLIST = [
    "自杀", "自残", "杀人", "恐怖袭击",
    "毒品", "贩毒", "冰毒", "海洛因",
    "儿童色情", "未成年",
    # 海外合规敏感词
    "racist", "terrorist", "pedophile",
]

# 拒绝回答的模板
REFUSAL_TEMPLATES = [
    "抱歉，我无法回答这个问题。如果你需要帮助，可以联系专业机构。",
    "这个问题超出了我能回答的范围。",
]


def check_safety(text: str) -> dict:
    """
    检测文本是否包含敏感内容
    Returns: {"safe": bool, "reason": str}
    """
    text_lower = text.lower()

    for word in BLACKLIST:
        if word.lower() in text_lower:
            return {
                "safe": False,
                "reason": f"检测到敏感词: {word}",
            }

    # 检测异常模式（连续重复字符、纯符号等）
    if re.search(r'(.)\1{20,}', text):  # 同一字符连续20次以上
        return {"safe": False, "reason": "检测到异常重复模式"}

    return {"safe": True, "reason": ""}


def get_safe_response(reason: str) -> str:
    """返回安全的拒绝回答"""
    return REFUSAL_TEMPLATES[0]


def check_output_safety(user_input: str, bot_output: str) -> dict:
    """
    检查用户输入和机器人输出的安全性
    面试讲：输入输出双检——防止用户发敏感内容，也防止模型生成不当内容
    """
    # 检查用户输入
    user_check = check_safety(user_input)
    if not user_check["safe"]:
        return {"safe": False, "blocked": "input", "reason": user_check["reason"]}

    # 检查模型输出
    output_check = check_safety(bot_output)
    if not output_check["safe"]:
        return {"safe": False, "blocked": "output", "reason": output_check["reason"]}

    return {"safe": True, "blocked": None, "reason": ""}
