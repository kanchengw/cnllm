import time
import json
import httpx
import asyncio
from typing import Iterator, AsyncIterator, Dict, Any, Optional

from ..utils.exceptions import (
    CNLLMError,
    ModelAPIError,
    RateLimitError,
    TimeoutError as CNLLMTimeoutError,
    NetworkError,
    ServerError,
    InvalidRequestError,
    InvalidURLError,
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
        provider: str = "unknown",
        header_mappings: Dict[str, str] = None
    ):
        self.api_key = ''.join(api_key.strip().split())
        self.base_url = base_url.strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.provider = provider
        self.header_mappings = header_mappings or {}

        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20
                )
            )
        return self._sync_client

    def close(self):
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    async def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20
                )
            )
        return self._async_client

    async def aclose(self):
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    def _build_headers(self, extra_headers: Dict[str, str] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8"
        }
        if extra_headers:
            for key, value in extra_headers.items():
                mapped_key = self.header_mappings.get(key, key)
                headers[mapped_key] = value
        return headers

    def _raise_for_status(self, response: httpx.Response, attempt: int) -> None:
        status_code = response.status_code

        try:
            error_detail = response.json() if response.text else {}
        except (json.JSONDecodeError, Exception):
            error_detail = {}

        if status_code == 401:
            raise AuthenticationError(provider=self.provider)
        elif status_code == 429:
            raise RateLimitError(provider=self.provider)
        elif status_code == 400:
            msg = error_detail.get("error", {}).get("message") or str(response.text)[:200] if response.text else "请求参数错误"
            raise InvalidRequestError(message=msg, provider=self.provider)
        elif status_code >= 500:
            msg = error_detail.get("error", {}).get("message") or f"API 服务器错误 ({status_code})"
            raise ServerError(message=msg, provider=self.provider)
        else:
            raise InvalidRequestError(
                message=f"API 请求失败 (HTTP {status_code})",
                provider=self.provider
            )

    def post(self, path: str, payload: Dict[str, Any], extra_headers: Dict[str, str] = None) -> Dict[str, Any]:
        headers = self._build_headers(extra_headers)
        base = self.base_url.rstrip("/")
        url_path = path.lstrip("/")
        url = f"{base}/{url_path}"

        for attempt in range(self.max_retries):
            try:
                client = self._get_sync_client()
                response = client.post(
                    url=url,
                    headers=headers,
                    json=payload
                )

                if response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    raise RateLimitError(provider=self.provider)

                if response.status_code >= 400:
                    self._raise_for_status(response, attempt)

                try:
                    return response.json()
                except json.JSONDecodeError:
                    raise InvalidRequestError(
                        message=f"API 返回了无效的 JSON 响应 (HTTP {response.status_code})",
                        provider=self.provider
                    )

            except httpx.TimeoutException:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise CNLLMTimeoutError(provider=self.provider)

            except httpx.ConnectError:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

            except (httpx.InvalidURL,) as e:
                raise InvalidURLError(message=str(e), provider=self.provider)

            except httpx.HTTPError:
                raise

            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

        raise ModelAPIError(f"重试 {self.max_retries} 次后仍然失败", provider=self.provider)

    def post_stream(self, path: str, payload: Dict[str, Any], extra_headers: Dict[str, str] = None) -> Iterator[bytes]:
        headers = self._build_headers(extra_headers)
        base = self.base_url.rstrip("/")
        url_path = path.lstrip("/")
        url = f"{base}/{url_path}"

        for attempt in range(self.max_retries):
            try:
                client = self._get_sync_client()
                with client.stream("POST", url=url, headers=headers, json=payload) as response:
                    if response.status_code == 429:
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        raise RateLimitError(provider=self.provider)

                    if response.status_code >= 400:
                        self._raise_for_status(response, attempt)

                    for line in response.iter_lines():
                        if line:
                            yield line.encode('utf-8')
                return

            except httpx.TimeoutException:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise CNLLMTimeoutError(provider=self.provider)

            except httpx.ConnectError:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

            except (httpx.InvalidURL,) as e:
                raise InvalidURLError(message=str(e), provider=self.provider)

            except httpx.HTTPError:
                raise

            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

        raise ModelAPIError(f"重试 {self.max_retries} 次后仍然失败", provider=self.provider)

    async def apost(self, path: str, payload: Dict[str, Any], extra_headers: Dict[str, str] = None) -> Dict[str, Any]:
        headers = self._build_headers(extra_headers)
        base = self.base_url.rstrip("/")
        url_path = path.lstrip("/")
        url = f"{base}/{url_path}"

        for attempt in range(self.max_retries):
            try:
                client = await self._get_async_client()
                response = await client.post(
                    url=url,
                    headers=headers,
                    json=payload
                )

                if response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    raise RateLimitError(provider=self.provider)

                if response.status_code >= 400:
                    self._raise_for_status(response, attempt)

                try:
                    return response.json()
                except json.JSONDecodeError:
                    raise InvalidRequestError(
                        message=f"API 返回了无效的 JSON 响应 (HTTP {response.status_code})",
                        provider=self.provider
                    )

            except httpx.TimeoutException:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise CNLLMTimeoutError(provider=self.provider)

            except httpx.ConnectError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

            except (httpx.InvalidURL,) as e:
                raise InvalidURLError(message=str(e), provider=self.provider)

            except httpx.HTTPError:
                raise

            except CNLLMTimeoutError:
                raise

            except RateLimitError:
                raise

            except AuthenticationError:
                raise

            except ServerError:
                raise

            except InvalidRequestError:
                raise

            except InvalidURLError:
                raise

            except ModelAPIError:
                raise

            except CNLLMError:
                raise

            except json.JSONDecodeError:
                raise

            except Exception as e:
                logger = __import__('logging').getLogger(__name__)
                logger.error(f"[{self.provider}] Unexpected error: {type(e).__name__}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

        raise ModelAPIError(f"重试 {self.max_retries} 次后仍然失败", provider=self.provider)

    async def apost_stream(self, path: str, payload: Dict[str, Any], extra_headers: Dict[str, str] = None) -> AsyncIterator[bytes]:
        headers = self._build_headers(extra_headers)
        base = self.base_url.rstrip("/")
        url_path = path.lstrip("/")
        url = f"{base}/{url_path}"

        for attempt in range(self.max_retries):
            try:
                client = await self._get_async_client()
                async with client.stream("POST", url=url, headers=headers, json=payload) as response:
                    if response.status_code == 429:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        raise RateLimitError(provider=self.provider)

                    if response.status_code >= 400:
                        self._raise_for_status(response, attempt)

                    async for line in response.aiter_lines():
                        if line:
                            yield line.encode('utf-8')
                return

            except httpx.TimeoutException:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise CNLLMTimeoutError(provider=self.provider)

            except httpx.ConnectError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

            except (httpx.InvalidURL,) as e:
                raise InvalidURLError(message=str(e), provider=self.provider)

            except httpx.HTTPError:
                raise

            except Exception:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

        raise ModelAPIError(f"重试 {self.max_retries} 次后仍然失败", provider=self.provider)
