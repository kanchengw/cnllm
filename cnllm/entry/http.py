import time
import requests
from typing import Iterator, Dict, Any
from ..utils.exceptions import (
    ModelAPIError,
    RateLimitError,
    TimeoutError,
    NetworkError,
    ServerError,
    InvalidRequestError,
    AuthenticationError
)


class BaseHttpClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
        provider: str = "unknown"
    ):
        self.api_key = ''.join(api_key.strip().split())
        self.base_url = base_url.strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.provider = provider

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8"
        }

    def _raise_for_status(self, response: requests.Response, attempt: int) -> None:
        status_code = response.status_code

        if status_code == 401:
            raise AuthenticationError(provider=self.provider)
        elif status_code == 429:
            raise RateLimitError(provider=self.provider)
        elif status_code == 400:
            raise InvalidRequestError(message=str(response.text), provider=self.provider)
        elif status_code >= 500:
            raise ServerError(
                message=f"API 服务器错误 ({status_code})",
                provider=self.provider
            )
        else:
            raise InvalidRequestError(
                message=f"API 请求失败 (HTTP {status_code})",
                provider=self.provider
            )

    def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = self._build_headers()

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url=f"{self.base_url}{path}",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    raise RateLimitError(provider=self.provider)

                if response.status_code >= 400:
                    self._raise_for_status(response, attempt)

                return response.json()

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise TimeoutError(provider=self.provider)

            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

            except requests.exceptions.HTTPError:
                raise

            except requests.exceptions.RequestException:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

        raise ModelAPIError(f"重试 {self.max_retries} 次后仍然失败")

    def post_stream(self, path: str, payload: Dict[str, Any]) -> Iterator[bytes]:
        headers = self._build_headers()

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
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    raise RateLimitError(provider=self.provider)

                if response.status_code >= 400:
                    self._raise_for_status(response, attempt)

                return response.iter_lines()

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise TimeoutError(provider=self.provider)

            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

            except requests.exceptions.HTTPError:
                raise

            except requests.exceptions.RequestException:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

        raise ModelAPIError(f"重试 {self.max_retries} 次后仍然失败")
