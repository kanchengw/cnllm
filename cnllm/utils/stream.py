"""
CNLLM 流式处理模块

提供 SSE 解码和 HTTP 流处理
"""
import json
from typing import Iterator, Dict, Any, Callable, AsyncIterator


class SSEDecoder:
    _seen_data: set = set()

    @staticmethod
    def decode_stream(response_iterator) -> Iterator[Dict[str, Any]]:
        seen_data = set()
        for line in response_iterator:
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data.strip() == '[DONE]':
                        break
                    if data in seen_data:
                        continue
                    seen_data.add(data)
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue


class AsyncSSEDecoder:
    @staticmethod
    async def decode_stream(response_async_iterator) -> AsyncIterator[Dict[str, Any]]:
        async for line in response_async_iterator:
            if line:
                line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                if line_str.startswith('data: '):
                    data = line_str[6:]
                    if data.strip() == '[DONE]':
                        break
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue


class StreamHandler:
    @staticmethod
    def handle_stream(
        client,
        api_path: str,
        payload: Dict[str, Any],
        extra_headers: Dict[str, str] = None
    ) -> Iterator[Dict[str, Any]]:
        for raw_chunk in SSEDecoder.decode_stream(client.post_stream(api_path, payload, extra_headers)):
            yield raw_chunk


class AsyncStreamHandler:
    @staticmethod
    async def ahandle_stream(
        client,
        api_path: str,
        payload: Dict[str, Any],
        extra_headers: Dict[str, str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        async for raw_chunk in AsyncSSEDecoder.decode_stream(
            client.apost_stream(api_path, payload, extra_headers)
        ):
            yield raw_chunk
