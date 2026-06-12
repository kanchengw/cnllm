"""
429 类型检测工具
类型: concurrency(并发/QPS) / rpm(含TPM及未知类型)
"""
import re
from typing import Optional, Dict

_CONCURR_RE = re.compile(
    r"(concurrent|concurrency|qps|并发超额|并发请求数|"
    r"Request rate increased too quickly|dimension:\s*concurrency|"
    r"engine is currently overloaded|please control request concurrency|"
    r"max concurrent requests reached|QPS limit exceeded|"
    r"concurrency limit exceeded)", re.I
)

_RPM_KW_RE = re.compile(
    r"rpm|每分钟请求|Requests per minute|EndpointRPMExceeded|"
    r"Rate limit reached for RPM|RPM limit|rate limit for requests",
    re.I
)


def detect_rate_type(message="", retry_after=-1,
                     headers=None):
    """判定 429 类型。优先级: 响应头 > retry_after > 精确关键词 > 默认"""
    if headers:
        if headers.get("X-RateLimit-Remaining-Concurrency") is not None:
            return "concurrency"
        if headers.get("X-RateLimit-Remaining-Requests") is not None:
            return "rpm"
    if retry_after >= 0 and retry_after < 2.0:
        return "concurrency"
    if message:
        if _CONCURR_RE.search(message):
            return "concurrency"
        if _RPM_KW_RE.search(message):
            return "rpm"
    return "rpm"
