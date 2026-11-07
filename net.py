"""受限网络下的取数：短超时、限页数、失败说人话。

打包版被拷到另一台机器后，最常见的故障是某个接口在慢网络里翻几百页，
界面只有一行静态的「首次较慢」，看起来就是永远卡死。这里统一收口两件事：

1. 超时。akshare 默认 (10, 60)，读取能给到 60 秒，慢网络下一次卡壳就一分钟，
   几百页滚起来就是几小时。这里压到连接 6 秒、读取 20 秒。
2. 报错。取数失败直接抛出带原因的错误，不再静默返回空表让界面显示 0 条。
"""
from __future__ import annotations

import time

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
TIMEOUT = (6, 20)

# 东财 datacenter 的通用入口，增减持、解禁、融资融券都走这里
DC_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


class NetError(RuntimeError):
    """网络取数失败，message 是给人看的原因，不是堆栈。"""


def reason(exc: BaseException) -> str:
    """把 requests 的异常翻成能看懂的一句话。"""
    if isinstance(exc, requests.exceptions.ProxyError):
        return "代理拒绝了请求，检查这台机器的代理或 VPN"
    if isinstance(exc, requests.exceptions.SSLError):
        return "HTTPS 证书校验失败，可能被公司网关或安全软件拦截"
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return "连不上东方财富，检查网络、DNS 与防火墙"
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return "东方财富响应太慢，网络质量差或本机被限流"
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "网络连接中断，检查网络"
    return f"{type(exc).__name__}: {exc}"


def get_json(url: str, params: dict | None = None, *, tries: int = 2,
             timeout=TIMEOUT) -> dict:
    """取 JSON。失败重试一次，两次都失败抛 NetError。"""
    last: BaseException | None = None
    for i in range(max(1, tries)):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=UA)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i < tries - 1:
                time.sleep(1.0)
    raise NetError(reason(last)) from last


def dc_pages(report: str, filter_: str = "", *, page_size: int = 500,
             max_pages: int = 12, sort_columns: str = "",
             sort_types: str = "", progress=None, label: str = "") -> list[dict]:
    """拉东财 datacenter 报表，带页数上限。

    单页最多 500 条，全量报表动辄几百页。调用方必须给出 filter 把范围缩小，
    max_pages 只是兜底，防止某个 filter 失效时退化成翻几百页。
    """
    params = {
        "reportName": report,
        "columns": "ALL",
        "quoteColumns": "",
        "filter": filter_,
        "pageNumber": "1",
        "pageSize": str(page_size),
        "sortTypes": sort_types,
        "sortColumns": sort_columns,
        "source": "WEB",
        "client": "WEB",
        "p": "1",
        "pageNo": "1",
        "pageNum": "1",
    }
    first = get_json(DC_URL, params)
    res = first.get("result") or {}
    pages = max(1, min(int(res.get("pages") or 1), max_pages))
    rows = list(res.get("data") or [])
    if progress and pages > 1:
        progress(f"{label} 1/{pages} 页")
    for p in range(2, pages + 1):
        if progress:
            progress(f"{label} {p}/{pages} 页")
        one = dict(params, pageNumber=str(p), p=str(p), pageNo=str(p),
                   pageNum=str(p))
        try:
            j = get_json(DC_URL, one)
        except NetError:
            break  # 翻到一半断了就用手上的，不整批作废
        lst = (j.get("result") or {}).get("data") or []
        if not lst:
            break
        rows.extend(lst)
    return rows


def tune_akshare(connect: int = 6, read: int = 20) -> None:
    """把 akshare 的全局超时收紧。它自己走 akshare.utils.http，默认 (10, 60)。"""
    try:
        from akshare.utils.context import config
        config.timeout = (connect, read)
    except Exception:  # noqa: BLE001
        pass


def preflight() -> None:
    """开跑前摸一下东财通不通。

    不通就直接抛错，不必让四个步骤各超时两轮、白等几分钟才报同一句话。
    用最小的报表请求探路，命中的是后面真的要用的那个域名。
    """
    get_json(
        DC_URL,
        {"reportName": "RPT_EXECUTIVE_HOLD_DETAILS", "columns": "SECURITY_CODE",
         "pageNumber": "1", "pageSize": "1", "source": "WEB", "client": "WEB"},
        tries=1, timeout=(5, 8))
