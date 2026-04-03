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
        to_openai_stream_format_func: Callable[[Dict[str, Any], str], Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        model = payload.get("model", "")
        for raw_chunk in SSEDecoder.decode_stream(client.post_stream(api_path, payload)):
            yield to_openai_stream_format_func(raw_chunk, model)


class StreamResultAccumulator:
    def __init__(self, chunks, adapter):
        self._chunks = []
        self._adapter = adapter
        self._iterator = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self._iterator)
        self._adapter._collect_stream_result(chunk)
        self._chunks.append(chunk)
        return chunk

    def get_chunks(self):
        return self._chunks