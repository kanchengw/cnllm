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
    AuthenticationError,
    ContentFilteredError,
    TokenLimitError,
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
        header_mappings: Dict[str, str] = None,
        yaml_default: str = None,
    ):
        self.api_key = "".join(api_key.strip().split())
        self.base_url = base_url.strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.provider = provider
        self.header_mappings = header_mappings or {}
        self.yaml_default = yaml_default.strip() if yaml_default else None

        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

    _URL_VERSION_PATTERN = None  # compiled lazily in _build_url

    def _build_url(self, path: str) -> str:
        """根据 5 条 URL 规则构造完整请求 URL。

        规则1: base_url 已包含完整 path → 原样返回
        规则2: base_url 以 /v{digit} 结尾 → 追加 path
        规则5: base_url 是 yaml_default 的前缀 → 补齐到 yaml_default 再拼 path
        规则3/4: 兜底 → base_url / path
        """
        base = self.base_url.rstrip("/")
        clean_path = path.strip().lstrip("/")

        if not clean_path:
            return base

        # 规则1: 完整路径
        if base.endswith(f"/{clean_path}"):
            return base

        # 规则2: 到版本号为止（去掉 path 中重复的版本前缀）
        if self._URL_VERSION_PATTERN is None:
            import re
            self._URL_VERSION_PATTERN = re.compile(r"/v\d+$")
        m = self._URL_VERSION_PATTERN.search(base)
        if m:
            vdir = m.group(0)  # e.g. "/v1"
            resource = clean_path
            # 如果 path 以相同版本号开头（如 "v1/chat/completions"），去掉版本前缀
            prefix = vdir.lstrip("/") + "/"
            if resource.startswith(prefix):
                resource = resource[len(prefix):]
            return f"{base}/{resource}"

        # 规则5: base_url 是 yaml_default 的前缀
        if self.yaml_default and self.yaml_default.startswith(base):
            yaml_base = self.yaml_default.rstrip("/")
            return f"{yaml_base}/{clean_path}"

        # 规则3/4: 兜底
        return f"{base}/{clean_path}"

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
        elif status_code == 403:
            raise ContentFilteredError(provider=self.provider)
        elif status_code == 408:
            raise CNLLMTimeoutError(provider=self.provider)
        elif status_code == 413:
            raise TokenLimitError(provider=self.provider)
        elif status_code == 429:
            raise RateLimitError(provider=self.provider)
        elif status_code == 400:
            msg = error_detail.get("error", {}).get("message") or str(response.text)[:200] if response.text else "\u8bf7\u6c42\u53c2\u6570\u9519\u8bef"
            raise InvalidRequestError(message=msg, provider=self.provider)
        elif status_code >= 500:
            msg = error_detail.get("error", {}).get("message") or f"API \u670d\u52a1\u5668\u9519\u8bef ({status_code})"
            raise ServerError(message=msg, provider=self.provider)
        else:
            raise InvalidRequestError(
                message=f"API \u8bf7\u6c42\u5931\u8d25 (HTTP {status_code})",
                provider=self.provider
            )

    def post(self, path: str, payload: Dict[str, Any], extra_headers: Dict[str, str] = None) -> Dict[str, Any]:
        headers = self._build_headers(extra_headers)
        url = self._build_url(path)

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
                        message=f"API \u8fd4\u56de\u4e86\u65e0\u6548\u7684 JSON \u54cd\u5e94 (HTTP {response.status_code})",
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

            except (RateLimitError, AuthenticationError, InvalidRequestError, ServerError):
                raise

            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

        raise ModelAPIError(f"\u91cd\u8bd5 {self.max_retries} \u6b21\u540e\u4ecd\u7136\u5931\u8d25", provider=self.provider)

    def post_stream(self, path: str, payload: Dict[str, Any], extra_headers: Dict[str, str] = None) -> Iterator[bytes]:
        headers = self._build_headers(extra_headers)
        url = self._build_url(path)

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
                            yield line.encode("utf-8")
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

            except (RateLimitError, AuthenticationError, InvalidRequestError, ServerError):
                raise

            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

        raise ModelAPIError(f"\u91cd\u8bd5 {self.max_retries} \u6b21\u540e\u4ecd\u7136\u5931\u8d25", provider=self.provider)

    async def apost(self, path: str, payload: Dict[str, Any], extra_headers: Dict[str, str] = None) -> Dict[str, Any]:
        headers = self._build_headers(extra_headers)
        url = self._build_url(path)

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
                        message=f"API \u8fd4\u56de\u4e86\u65e0\u6548\u7684 JSON \u54cd\u5e94 (HTTP {response.status_code})",
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
                logger = __import__("logging").getLogger(__name__)
                logger.error(f"[{self.provider}] Unexpected error: {type(e).__name__}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

        raise ModelAPIError(f"\u91cd\u8bd5 {self.max_retries} \u6b21\u540e\u4ecd\u7136\u5931\u8d25", provider=self.provider)

    async def apost_stream(self, path: str, payload: Dict[str, Any], extra_headers: Dict[str, str] = None) -> AsyncIterator[bytes]:
        headers = self._build_headers(extra_headers)
        url = self._build_url(path)

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
                            yield line.encode("utf-8")
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

            except (RateLimitError, AuthenticationError, InvalidRequestError, ServerError):
                raise

            except Exception:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise NetworkError(provider=self.provider)

        raise ModelAPIError(f"\u91cd\u8bd5 {self.max_retries} \u6b21\u540e\u4ecd\u7136\u5931\u8d25", provider=self.provider)
