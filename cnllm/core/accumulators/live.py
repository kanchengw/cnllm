"""Rich Live 实时刷新视图，用于流式响应字典的终端原地展示。"""


class LiveDict:
    """原地刷新的累积字典视图。

    用法:
        with resp.live as ld:
            for chunk in resp:
                ld.refresh()
    """

    def __init__(self, accumulator):
        self._acc = accumulator
        self._live = None
        self._saved_warn_filters = None

    def __enter__(self):
        # 屏蔽 Live 生命周期内的 ResourceWarning（SSL socket gc 噪音）
        import warnings
        self._saved_warn_filters = warnings.filters.copy()
        warnings.simplefilter("ignore", ResourceWarning)
        from rich.live import Live
        from rich.text import Text
        self._live = Live(Text(""), refresh_per_second=10)
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        if self._live:
            self._live.__exit__(*args)
        import warnings
        if self._saved_warn_filters is not None:
            warnings.filters = self._saved_warn_filters

    def refresh(self):
        from rich.text import Text
        text = repr(self._acc._accumulate())
        self._live.update(Text(text))
