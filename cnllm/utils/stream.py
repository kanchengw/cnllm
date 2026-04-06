import json
from typing import Iterator, Dict, Any, Callable


class SSEDecoder:
    @staticmethod
    def decode_stream(response_iterator) -> Iterator[Dict[str, Any]]:
        for line in response_iterator:
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data.strip() == '[DONE]':
                        return
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
        to_openai_stream_format_func: Callable[[Dict[str, Any], str], Dict[str, Any]],
        extra_headers: Dict[str, str] = None
    ) -> Iterator[Dict[str, Any]]:
        model = payload.get("model", "")
        for raw_chunk in SSEDecoder.decode_stream(client.post_stream(api_path, payload, extra_headers)):
            yield to_openai_stream_format_func(raw_chunk, model)


class StreamResultAccumulator:
    def __init__(self, chunks, adapter):
        self._chunks = []
        self._adapter = adapter
        self._iterator = iter(chunks)
        self._thinking = ""
        self._content = ""
        self._tools = None

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self._iterator)

        reasoning = chunk.get("_reasoning_content")
        if reasoning:
            self._thinking += reasoning

        delta = chunk.get("choices", [{}])[0].get("delta", {}) if chunk.get("choices") else {}
        content_delta = delta.get("content", "")
        if content_delta:
            self._content += content_delta

        tools_delta = delta.get("tool_calls")
        if tools_delta:
            self._tools = tools_delta

        chunk["_thinking"] = self._thinking
        chunk["_content"] = self._content
        if self._tools:
            chunk["_tools"] = self._tools

        self._chunks.append(chunk)
        return chunk

    def get_chunks(self):
        return self._chunks
