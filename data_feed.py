"""数据层：封装 akshare 拉取 A 股多周期 K 线、行情快照与全市场代码表。

对外函数返回归一化列名的 pandas.DataFrame：
    date, open, close, high, low, volume, amount, pct_change
date 为 datetime64（分钟线含时分），其余为 float。

周期：
    分时(1分钟) / 5分钟 / 15分钟 / 30分钟 / 60分钟 / 日K / 周K / 月K
"""
from __future__ import annotations

import os
import json
import time
import logging
import datetime

import pandas as pd

import akshare as ak

from config import BARS_DAILY_YEARS, BARS_WEEKLY_YEARS, BARS_MONTHLY_YEARS

logger = logging.getLogger("quant_desk.data_feed")

# 归一化列名
_COL_MAP = {
    "日期": "date",
    "时间": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
    "均价": "avg_price",
}

# 界面周期名 -> akshare 分钟周期
MIN_PERIODS = {"分时": "1", "5分钟": "5", "15分钟": "15", "30分钟": "30", "60分钟": "60"}
# 界面周期名 -> akshare k 线周期
KLINE_PERIODS = {"日K": "daily", "周K": "weekly", "月K": "monthly"}

ALL_PERIODS = ["分时", "5分钟", "15分钟", "30分钟", "60分钟", "日K", "周K", "月K"]

_SPOT_CACHE = {"ts": 0.0, "df": None}
_SPOT_TTL = 30.0  # 秒

_SPOT_FULL_CACHE = {"ts": 0.0, "df": None}
_SPOT_FULL_TTL = 1800.0  # 全市场代码表缓存 30 分钟


def _to_num(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("open", "close", "high", "low", "volume", "amount",
              "pct_change", "turnover", "amplitude", "change", "avg_price"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """统一收尾：排好序、补齐缺失列、补 pct_change。"""
    if df is None or df.empty:
        return pd.DataFrame()
    df = _to_num(df)
    for c in ("open", "close", "high", "low", "volume"):
        if c not in df.columns:
            return pd.DataFrame()
    df = df.dropna(subset=["date", "open", "close", "high", "low"]).sort_values("date")
    df = df.reset_index(drop=True)
    if "amount" not in df.columns:
        df["amount"] = float("nan")
    if "pct_change" not in df.columns or df["pct_change"].isna().all():
        df["pct_change"] = df["close"].pct_change() * 100.0
    # 分钟线第一根常见 开盘=0（东财不返回首笔开盘），用收盘兜住，否则蜡烛画不出来
    bad = df["open"] <= 0
    if bad.any():
        df.loc[bad, "open"] = df.loc[bad, "close"]
    for c in ("high", "low"):
        bad = df[c] <= 0
        if bad.any():
            df.loc[bad, c] = df.loc[bad, "close"]
    return df


def get_history(symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """拉取单只 A 股历史日线（回测用，保持原接口不变）。

    symbol: 6 位代码；start/end: 'YYYYMMDD'；adjust: 'qfq'前复权/'hfq'后复权/''不复权。
    失败或空数据返回空 DataFrame。
    """
    try:
        raw = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start, end_date=end, adjust=adjust,
        )
    except Exception as exc:  # akshare 网络/解析异常种类多，统一兜住
        logger.warning("get_history %s 失败: %s", symbol, exc)
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.rename(columns=_COL_MAP)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return _finalize(df)


def get_bars(symbol: str, period: str, adjust: str = "qfq") -> pd.DataFrame:
    """按界面周期名拉 K 线，供看盘绘图使用。

    period: ALL_PERIODS 之一（分时/5分钟/.../日K/周K/月K）。
    """
    try:
        if period in MIN_PERIODS:
            return _get_minute(symbol, MIN_PERIODS[period])
        return _get_kline(symbol, KLINE_PERIODS.get(period, "daily"), adjust)
    except Exception as exc:
        logger.warning("get_bars %s %s 失败: %s", symbol, period, exc)
        return pd.DataFrame()


def _get_minute(symbol: str, period: str) -> pd.DataFrame:
    """分钟线。1 分钟不支持复权参数，其余可传 qfq。"""
    if period == "1":
        raw = ak.stock_zh_a_hist_min_em(symbol=symbol, period="1", adjust="")
    else:
        raw = ak.stock_zh_a_hist_min_em(symbol=symbol, period=period, adjust="qfq")
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns=_COL_MAP)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = _finalize(df)
    if df.empty:
        return df
    if period == "1":
        # 分时只保留最近一个交易日，避免 5 天连在一起看不出当日节奏
        last_day = df["date"].dt.date.max()
        df = df[df["date"].dt.date == last_day].reset_index(drop=True)
        if len(df) > 1:
            df["pct_change"] = df["close"].pct_change() * 100.0
    return df


def _get_kline(symbol: str, period: str, adjust: str) -> pd.DataFrame:
    years = {"daily": BARS_DAILY_YEARS, "weekly": BARS_WEEKLY_YEARS,
             "monthly": BARS_MONTHLY_YEARS}.get(period, 2)
    end = datetime.date.today()
    start = end - datetime.timedelta(days=int(365.25 * years))
    raw = ak.stock_zh_a_hist(
        symbol=symbol, period=period,
        start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"),
        adjust=adjust,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns=_COL_MAP)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return _finalize(df)


def get_spot(force: bool = False) -> pd.DataFrame:
    """A 股全市场实时快照（东方财富）。带 30s 短缓存。"""
    now = time.time()
    if not force and _SPOT_CACHE["df"] is not None and now - _SPOT_CACHE["ts"] < _SPOT_TTL:
        return _SPOT_CACHE["df"]
    try:
        df = ak.stock_zh_a_spot_em()
        _SPOT_CACHE["df"] = df
        _SPOT_CACHE["ts"] = now
        return df
    except Exception as exc:
        logger.warning("get_spot 失败: %s", exc)
        return _SPOT_CACHE["df"] if _SPOT_CACHE["df"] is not None else pd.DataFrame()


def _secid(code: str) -> str:
    """东财 secid 前缀：港股(HK 前缀)=116，北交所(4/8/920 开头)=0，沪市(6/9)=1，其余深市=0。"""
    c = str(code).strip().upper()
    if c.startswith("HK"):
        return "116." + c[2:].zfill(5)
    if c[:1] in ("4", "8") or c[:3] == "920":
        return "0." + c
    return ("1." if c[:1] in ("6", "9") else "0.") + c


def _norm_code(s) -> str:
    """统一代码口径：港股补 HK 前缀保留 5 位，其余补足 6 位。"""
    c = str(s).strip().upper()
    if c.startswith("HK"):
        return "HK" + c[2:].zfill(5)
    return c.zfill(6)


def get_quotes(symbols: list[str]) -> dict[str, dict]:
    """批量取多只股票的最新行情（东财 ulist 接口，一次请求约 0.2s）。

    比 akshare 的全市场快照（~65s）快两个数量级，适合盯盘表格刷新。
    支持港股：传 "HK00290" 这类带前缀的代码。
    """
    out: dict[str, dict] = {}
    codes = [_norm_code(s) for s in symbols if str(s).strip()]
    if not codes:
        return out
    if len(codes) > 100:                    # 东财单次上限，分批发
        for i in range(0, len(codes), 80):
            out.update(get_quotes(codes[i:i + 80]))
        return out
    try:
        import requests
        url = "https://push2.eastmoney.com/api/qt/ulist.np/get"
        params = {
            "fltt": "2",
            "secids": ",".join(_secid(c) for c in codes),
            "fields": "f2,f3,f4,f5,f6,f8,f12,f13,f14,f15,f16,f17,f18",
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        }
        r = requests.get(url, params=params, timeout=8,
                         headers={"User-Agent": "Mozilla/5.0"})
        diff = (r.json().get("data") or {}).get("diff") or []
        if isinstance(diff, dict):          # 单只时东财返回 dict 而非 list
            diff = [diff]
        for d in diff:
            mk = d.get("f13")
            raw = str(d.get("f12", ""))
            code = ("HK" + raw.zfill(5)) if mk == 116 else raw.zfill(6)
            if code not in codes:
                continue
            def g(key):
                v = d.get(key)
                try:
                    f = float(v)
                    return f if f == f else None
                except (TypeError, ValueError):
                    return None
            out[code] = {
                "name": str(d.get("f14", "") or ""),
                "price": g("f2"),
                "pct": g("f3"),
                "change": g("f4"),
                "volume": g("f5"),
                "amount": g("f6"),
                "turnover": g("f8"),
                "high": g("f15"),
                "low": g("f16"),
                "open": g("f17"),
                "pre_close": g("f18"),
            }
    except Exception as exc:
        logger.warning("get_quotes 失败: %s", exc)

    # 北交所、配债等在 ulist 批量接口里常缺，逐个补查（一般只剩个位数）
    missing = [c for c in codes if c not in out][:24]
    for c in missing:
        q = _quote_one(c)
        if q:
            out[c] = q
    return out


def _quote_one(code: str) -> dict | None:
    """单只行情兜底（东财 stock/get）。北交所价格需按小数位还原。"""
    try:
        import requests
        url = "https://push2.eastmoney.com/api/qt/stock/get"
        params = {"secid": _secid(code), "ut": "fa5fd1943c7b386f172d6893dbfba10b",
                  "fields": "f43,f57,f58,f59,f60,f169,f170"}
        r = requests.get(url, params=params, timeout=6,
                         headers={"User-Agent": "Mozilla/5.0"})
        d = (r.json() or {}).get("data")
        if not d or d.get("f43") in (None, "-"):
            return None
        dec = d.get("f59")
        dec = int(dec) if isinstance(dec, (int, float)) else 2
        scale = 10 ** dec
        px = float(d["f43"]) / scale
        pre = float(d["f60"]) / scale if d.get("f60") not in (None, "-") else None
        return {
            "name": str(d.get("f58", "") or ""),
            "price": px,
            "pct": (px / pre - 1) * 100 if pre else None,
            "change": (px - pre) if pre else None,
            "volume": None, "amount": None, "turnover": None,
            "high": None, "low": None, "open": None,
            "pre_close": pre,
        }
    except Exception:
        return None


_FX_CACHE = {"ts": 0.0, "hkd": 0.0}


def get_hkd_cny() -> float:
    """港币兑人民币汇率（港股成本是人民币、行情是港币，折算要用）。取不到退回 0.85。"""
    now = time.time()
    if _FX_CACHE["hkd"] and now - _FX_CACHE["ts"] < 1800:
        return _FX_CACHE["hkd"]
    try:
        import requests
        url = "https://push2.eastmoney.com/api/qt/stock/get"
        params = {"secid": "133.HKDCNH", "ut": "fa5fd1943c7b386f172d6893dbfba10b",
                  "fields": "f43,f59"}
        r = requests.get(url, params=params, timeout=6,
                         headers={"User-Agent": "Mozilla/5.0"})
        d = (r.json() or {}).get("data") or {}
        px = float(d.get("f43")) / (10 ** int(d.get("f59") or 4))
        if 0.5 < px < 1.5:
            _FX_CACHE.update(ts=now, hkd=px)
            return px
    except Exception:
        pass
    return _FX_CACHE["hkd"] or 0.85


_NAMES_CACHE = {"ts": 0.0, "map": {}}
_NAMES_TTL = 86400.0  # 代码名称表一天更新一次足够
_NAMES_FILE = os.path.join(os.path.expanduser("~"), ".quantdesk", "names.json")


def get_stock_names(force: bool = False) -> dict[str, str]:
    """全市场 代码->名称 映射（搜索添加标的用）。内存缓存 + 本地文件缓存 24h。"""
    now = time.time()
    if not force and _NAMES_CACHE["map"] and now - _NAMES_CACHE["ts"] < _NAMES_TTL:
        return _NAMES_CACHE["map"]

    if not force:
        try:
            with open(_NAMES_FILE, encoding="utf-8") as f:
                obj = json.load(f)
            if obj.get("map") and now - float(obj.get("ts", 0)) < _NAMES_TTL:
                _NAMES_CACHE.update(ts=float(obj["ts"]), map=obj["map"])
                return obj["map"]
        except Exception:
            pass

    m: dict[str, str] = {}
    try:
        df = ak.stock_info_a_code_name()
        if df is not None and not df.empty:
            m = {str(c).zfill(6): str(n) for c, n in zip(df["code"], df["name"])}
    except Exception as exc:
        logger.warning("stock_info_a_code_name 失败，回退快照: %s", exc)

    if not m:
        try:
            spot = get_spot()
            if spot is not None and not spot.empty and "代码" in spot.columns:
                m = {str(c).zfill(6): str(n) for c, n in
                     zip(spot["代码"].astype(str), spot["名称"].astype(str))}
        except Exception:
            pass

    if m:
        _NAMES_CACHE.update(ts=now, map=m)
        try:
            os.makedirs(os.path.dirname(_NAMES_FILE), exist_ok=True)
            with open(_NAMES_FILE, "w", encoding="utf-8") as f:
                json.dump({"ts": now, "map": m}, f, ensure_ascii=False)
        except Exception:
            pass
    return m


def _squash(s: str) -> str:
    """全角转半角 + 去空格，便于"A股/B股/Ａ"这类名称互通匹配。"""
    import unicodedata
    return "".join(unicodedata.normalize("NFKC", str(s)).split()).lower()


def search_stocks(keyword: str, limit: int = 12) -> list[dict]:
    """按代码或名称模糊搜索标的，返回 [{code, name}, ...]。"""
    kw = _squash(keyword)
    if not kw:
        return []
    names = get_stock_names()
    if not names:
        return []
    hits = []
    for code, name in names.items():
        if kw in code or kw in _squash(name):
            hits.append({"code": code, "name": name,
                         "exact": _squash(name) == kw or code == kw})
    # 精确名/完整代码优先，其次按代码排序
    hits.sort(key=lambda x: (not x["exact"], x["code"]))
    return [{"code": h["code"], "name": h["name"]} for h in hits[:limit]]
