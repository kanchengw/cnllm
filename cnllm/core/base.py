import time
import requests
from typing import Dict, Any, Iterator
from ..utils.exceptions import ModelAPIError


class BaseHttpClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.api_key = ''.join(api_key.strip().split())
        self.base_url = base_url.strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8"
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url=f"{self.base_url}{path}",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 429:
                    wait_time = self.retry_delay * (2 ** attempt)
                    raise ModelAPIError(
                        f"请求被限流 (429)。\n"
                        f"等待 {wait_time:.1f} 秒后重试... (尝试 {attempt + 1}/{self.max_retries})"
                    )

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                last_error = f"请求超时 (timeout={self.timeout}s)"
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue

            except requests.exceptions.ConnectionError as e:
                last_error = f"连接失败: {str(e)}"
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if status_code == 429:
                    wait_time = self.retry_delay * (2 ** attempt)
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    raise ModelAPIError(
                        f"请求被限流 (429)。已重试 {self.max_retries} 次。\n"
                        f"请稍后重试或联系 API 提供商。"
                    )
                elif status_code >= 500:
                    last_error = f"服务器错误 ({status_code})"
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    raise ModelAPIError(
                        f"API 服务器错误 ({status_code})。已重试 {self.max_retries} 次。\n"
                        f"请稍后重试。"
                    )
                else:
                    raise ModelAPIError(
                        f"API 请求失败 (HTTP {status_code}): {str(e)}\n"
                        f"请检查 API Key 和请求参数。"
                    )
            except Exception as e:
                raise ModelAPIError(f"API 请求失败: {str(e)}")

        raise ModelAPIError(f"重试 {self.max_retries} 次后仍然失败。最后错误: {last_error}")

    def post_stream(self, path: str, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8"
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url=f"{self.base_url}{path}",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    stream=True
                )

                if response.status_code == 429:
                    wait_time = self.retry_delay * (2 ** attempt)
                    raise ModelAPIError(
                        f"请求被限流 (429)。\n"
                        f"等待 {wait_time:.1f} 秒后重试... (尝试 {attempt + 1}/{self.max_retries})"
                    )

                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data.strip() == '[DONE]':
                                return
                            try:
                                import json
                                yield json.loads(data)
                            except json.JSONDecodeError:
                                continue

                return

            except requests.exceptions.Timeout:
                last_error = f"请求超时 (timeout={self.timeout}s)"
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue

            except requests.exceptions.ConnectionError as e:
                last_error = f"连接失败: {str(e)}"
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                if status_code >= 500:
                    last_error = f"服务器错误 ({status_code})"
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                raise ModelAPIError(
                    f"API 请求失败 (HTTP {status_code}): {str(e)}"
                )
            except Exception as e:
                raise ModelAPIError(f"流式请求失败: {str(e)}")

        raise ModelAPIError(f"重试 {self.max_retries} 次后仍然失败。最后错误: {last_error}")
