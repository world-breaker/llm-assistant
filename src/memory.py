"""
memory.py —— 多轮对话记忆管理
==============================
情感陪伴场景：不能一问一答就忘。需要记住上下文。
"""
from typing import List, Dict


class ConversationMemory:
    """简单的滑动窗口对话记忆"""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add(self, user_msg: str, bot_msg: str):
        self.history.append({"user": user_msg, "bot": bot_msg})
        # 滑动窗口：只保留最近 N 轮
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_context(self) -> str:
        """把历史拼接成上下文文本"""
        if not self.history:
            return ""
        lines = ["[对话历史]"]
        for i, turn in enumerate(self.history):
            lines.append(f"用户: {turn['user']}")
            lines.append(f"助手: {turn['bot']}")
        return "\n".join(lines)

    def get_messages(self) -> List[Dict[str, str]]:
        """返回 LangChain 格式的消息列表"""
        msgs = []
        for turn in self.history:
            msgs.append({"role": "user", "content": turn["user"]})
            msgs.append({"role": "assistant", "content": turn["bot"]})
        return msgs

    def clear(self):
        self.history = []

    def __len__(self):
        return len(self.history)
