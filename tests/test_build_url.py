"""
_build_url 规则测试

验证 BaseHttpClient._build_url 的五条规则：
规则1 - 用户已包含完整路径 → 原样
规则2 - 到版本号为止 → 追加资源部分
规则5 - URL 是 default 前缀 → 补全到 default
规则3/4 - 兜底 → 追加完整 path
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cnllm.entry.http import BaseHttpClient


class TestBuildUrlXiaomi:
    """厂商：xiaomi  default=https://api.xiaomimimo.com  path=v1/chat/completions"""

    YAML_DEFAULT = "https://api.xiaomimimo.com"

    def build(self, base_url: str) -> str:
        client = BaseHttpClient(api_key="test", base_url=base_url, yaml_default=self.YAML_DEFAULT)
        return client._build_url("v1/chat/completions")

    def test_rule1_full_path(self):
        """规则1: 用户已包含完整路径 → 原样"""
        url = "https://token-plan.xxx.com/v1/chat/completions"
        result = self.build(url)
        assert result == url, f"期望原样，得到 {result}"

    def test_rule2_v1_only(self):
        """规则2: 到 /v1 为止 → 追加 chat/completions"""
        url = "https://token-plan.xxx.com/v1"
        result = self.build(url)
        assert result == "https://token-plan.xxx.com/v1/chat/completions", f"得到 {result}"

    def test_rule5_prefix_domain(self):
        """规则5: 只传域名 → 补全到 default 再拼 path"""
        url = "https://api.xiaomimimo.com"
        result = self.build(url)
        assert result == "https://api.xiaomimimo.com/v1/chat/completions", f"得到 {result}"

    def test_rule5_prefix_subdomain(self):
        """规则5: 自定义子域名 → 补全到 default 再拼 path"""
        url = "https://custom.xiaomimimo.com"
        result = self.build(url)
        # 不匹配 default 前缀 → 规则3/4 兜底
        assert result == "https://custom.xiaomimimo.com/v1/chat/completions", f"得到 {result}"

    def test_rule3_bare_domain(self):
        """规则3/4: 其他域名兜底"""
        url = "https://my-gateway.com"
        result = self.build(url)
        assert result == "https://my-gateway.com/v1/chat/completions", f"得到 {result}"

    def test_rule1_different_path(self):
        """规则1: 用户传了完整但不同路径 → 仍然是规则1"""
        url = "https://custom.api.com/v2/generate"
        result = self.build(url)
        # 不以 /v1/chat/completions 结尾 → 不命中规则1
        # 不以 /v1 结尾 → 不命中规则2
        # 不匹配 default 前缀 → 规则3/4 兜底
        assert result == "https://custom.api.com/v2/generate/v1/chat/completions", f"得到 {result}"

    def test_default_url(self):
        """默认 base_url 本身"""
        url = "https://api.xiaomimimo.com"
        result = self.build(url)
        assert result == "https://api.xiaomimimo.com/v1/chat/completions", f"得到 {result}"


class TestBuildUrlGLM:
    """厂商：GLM  default=https://open.bigmodel.cn/api/paas  path=v4/chat/completions"""

    YAML_DEFAULT = "https://open.bigmodel.cn/api/paas"

    def build(self, base_url: str) -> str:
        client = BaseHttpClient(api_key="test", base_url=base_url, yaml_default=self.YAML_DEFAULT)
        return client._build_url("v4/chat/completions")

    def test_rule1_full_path(self):
        """规则1: 用户已包含完整路径 → 原样"""
        url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        result = self.build(url)
        assert result == url, f"期望原样，得到 {result}"

    def test_rule2_v4_only(self):
        """规则2: 到 /v4 为止 → 追加 chat/completions"""
        url = "https://open.bigmodel.cn/api/paas/v4"
        result = self.build(url)
        assert result == "https://open.bigmodel.cn/api/paas/v4/chat/completions", f"得到 {result}"

    def test_rule5_to_api_paas(self):
        """规则5: 到 /api/paas（default 级别）→ 补全到 default + 拼完整 path"""
        url = "https://open.bigmodel.cn/api/paas"
        result = self.build(url)
        assert result == "https://open.bigmodel.cn/api/paas/v4/chat/completions", f"得到 {result}"

    def test_rule5_bare_domain(self):
        """规则5: 只传域名 → 补全到 default 再拼 path"""
        url = "https://open.bigmodel.cn"
        result = self.build(url)
        # open.bigmodel.cn 是 open.bigmodel.cn/api/paas 的前缀 → 补全
        assert result == "https://open.bigmodel.cn/api/paas/v4/chat/completions", f"得到 {result}"

    def test_rule3_custom_domain(self):
        """规则3/4: 自定义域名兜底"""
        url = "https://my-glm-gateway.com"
        result = self.build(url)
        assert result == "https://my-glm-gateway.com/v4/chat/completions", f"得到 {result}"

    def test_rule5_partial_prefix(self):
        """规则5: 传了部分路径但不是完整 default"""
        url = "https://open.bigmodel.cn/api"
        result = self.build(url)
        # open.bigmodel.cn/api 是 open.bigmodel.cn/api/paas 的前缀 → 补全
        assert result == "https://open.bigmodel.cn/api/paas/v4/chat/completions", f"得到 {result}"


class TestBuildUrlMiniMax:
    """厂商：MiniMax  default=https://api.minimaxi.com  path=v1/text/chatcompletion_v2"""

    YAML_DEFAULT = "https://api.minimaxi.com"

    def build(self, base_url: str) -> str:
        client = BaseHttpClient(api_key="test", base_url=base_url, yaml_default=self.YAML_DEFAULT)
        return client._build_url("v1/text/chatcompletion_v2")

    def test_rule1_full_path(self):
        """规则1: 完整路径"""
        url = "https://api.minimaxi.com/v1/text/chatcompletion_v2"
        result = self.build(url)
        assert result == url, f"期望原样，得到 {result}"

    def test_rule2_v1_only(self):
        """规则2: 到 /v1 → 追加 text/chatcompletion_v2"""
        url = "https://custom.minimaxi.com/v1"
        result = self.build(url)
        assert result == "https://custom.minimaxi.com/v1/text/chatcompletion_v2", f"得到 {result}"

    def test_rule3_bare_domain(self):
        """规则3/4: 兜底拼接完整 path"""
        url = "https://my-gateway.com"
        result = self.build(url)
        assert result == "https://my-gateway.com/v1/text/chatcompletion_v2", f"得到 {result}"


class TestBuildUrlDeepSeek:
    """厂商：DeepSeek  default=https://api.deepseek.com  path=v1/chat/completions"""

    YAML_DEFAULT = "https://api.deepseek.com"

    def build(self, base_url: str) -> str:
        client = BaseHttpClient(api_key="test", base_url=base_url, yaml_default=self.YAML_DEFAULT)
        return client._build_url("v1/chat/completions")

    def test_rule1_full_path(self):
        url = "https://api.deepseek.com/v1/chat/completions"
        result = self.build(url)
        assert result == url

    def test_rule2_v1_only(self):
        url = "https://custom.deepseek.com/v1"
        result = self.build(url)
        assert result == "https://custom.deepseek.com/v1/chat/completions"

    def test_rule3_bare_domain(self):
        url = "https://my-gateway.com"
        result = self.build(url)
        assert result == "https://my-gateway.com/v1/chat/completions"


class TestBuildUrlNoYamlDefault:
    """没有 yaml_default（兜底场景）"""

    def build(self, base_url: str, path: str = "v1/chat/completions") -> str:
        client = BaseHttpClient(api_key="test", base_url=base_url)
        return client._build_url(path)

    def test_no_default_bare_domain(self):
        """无 yaml_default → 规则3/4 直接拼"""
        result = self.build("https://my-gateway.com")
        assert result == "https://my-gateway.com/v1/chat/completions"

    def test_no_default_v1(self):
        """无 yaml_default → 规则2 仍生效"""
        result = self.build("https://my-gateway.com/v1")
        assert result == "https://my-gateway.com/v1/chat/completions"

    def test_no_default_full(self):
        """无 yaml_default → 规则1 仍生效"""
        result = self.build("https://my-gateway.com/v1/chat/completions")
        assert result == "https://my-gateway.com/v1/chat/completions"


class TestBuildUrlEdgeCases:
    """边界场景"""

    def build(self, base_url: str, path: str = "v1/chat/completions", yaml_default: str = None) -> str:
        client = BaseHttpClient(api_key="test", base_url=base_url, yaml_default=yaml_default)
        return client._build_url(path)

    def test_trailing_slash(self):
        """带尾斜杠的 URL"""
        result = self.build("https://my-gateway.com/")
        assert result == "https://my-gateway.com/v1/chat/completions"

    def test_empty_path(self):
        """path 为空字符串"""
        result = self.build("https://my-gateway.com", path="")
        assert result == "https://my-gateway.com"

    def test_path_with_leading_slash(self):
        """path 以 / 开头"""
        result = self.build("https://my-gateway.com", path="/v1/chat/completions")
        assert result == "https://my-gateway.com/v1/chat/completions"

    def test_rule5_exact_match_default(self):
        """规则5: 用户 URL 完全等于 yaml_default"""
        result = self.build("https://api.deepseek.com", yaml_default="https://api.deepseek.com")
        assert result == "https://api.deepseek.com/v1/chat/completions"

    def test_rule1_endswith_same_path(self):
        """规则1: endswith 匹配"""
        url = "https://api.deepseek.com/v1/chat/completions"
        result = self.build(url, path="v1/chat/completions", yaml_default="https://api.deepseek.com")
        assert result == url

    def test_url_with_port(self):
        """带端口号的 URL"""
        result = self.build("https://localhost:8080")
        assert result == "https://localhost:8080/v1/chat/completions"

    def test_url_with_port_v1(self):
        """带端口号 + /v1"""
        result = self.build("https://localhost:8080/v1")
        assert result == "https://localhost:8080/v1/chat/completions"
