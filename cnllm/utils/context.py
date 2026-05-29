"""
对话上下文构建工具

提供 ``ContextBox`` 用于将 ``resp.*`` 的完整累积结果格式化为
OpenAI 标准消息列表，自动处理 assistant + tool 消息。

用法::

    from cnllm import ContextBox

    messages += ContextBox(resp.still, resp.think)
    # → [{"role": "assistant", "content": "think...\\n\\nstill..."}]

    messages += ContextBox(resp.still, resp.think, resp.tools,
                           executor=execute_tool)
    # → assistant + tool_calls 自动附着，工具执行结果逐条追加
"""
from typing import Dict, Any, List


class ContextBox(list):
    """构建对话上下文消息列表，自动处理 assistant + tool 消息。

    参数:
        still: ``resp.still``，模型回复文本
        think: ``resp.think``，推理过程（可选，自动拼接）
        tools: ``resp.tools``，工具调用列表（可选）
        executor: 执行工具的可选参数，接收原始 ``tc`` dict，
                  返回执行结果字符串
    """

    def __init__(self, still: str = "", think: str = None,
                 tools: List[Dict] = None, executor=None):
        content = think + "\n\n" + still if think else still
        assistant_msg: Dict[str, Any] = {"role": "assistant",
                                          "content": content}
        if tools:
            assistant_msg["tool_calls"] = tools
        msgs: List[Dict[str, Any]] = [assistant_msg]
        if tools and executor:
            for tc in tools:
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": executor(tc),
                })
        if not msgs[0].get("content") and not msgs[0].get("tool_calls"):
            raise ValueError("ContextBox requires at least one of still, think, or tools")
        super().__init__(msgs)
