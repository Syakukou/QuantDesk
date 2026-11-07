"""数据源适配层：默认走东方财富，tushare 留一个开关。

为什么要有这一层：现在没有 tushare token，回测只能拿东财的日线，涨跌停价、
停牌标记、历史成分权重这些拿不到，只能用近似值。等 token 到位，改一处就能
换成时点正确的数据，其余代码不用动。

各数据源的能力差异在 capabilities() 里讲清楚，界面上会显示当前用的是哪一档，
避免拿近似数据冒充精确数据。
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

import data_feed

_CFG_DIR = os.path.join(os.path.expanduser("~"), ".quantdesk")
_CFG_FILE = os.path.join(_CFG_DIR, "datasource.json")
_IND_CACHE: dict[str, tuple[float, str]] = {}
_IND_TTL = 86400.0


class Provider:
    name = "base"
    label = "基础"

    def bars(self, code: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        raise NotImplementedError

    def industry_map(self, codes: list[str]) -> dict[str, str]:
        return {}

    def capabilities(self) -> dict:
        return {}


class EastMoneyProvider(Provider):
    """东方财富。免费、无需 token、覆盖全 A 股，但没有 PIT 与涨跌停价。"""
    name = "eastmoney"
    label = "东方财富（免费，无需 token）"

    def bars(self, code, start, end, adjust="qfq"):
        df = data_feed.get_history(code, start, end, adjust)
        if df is None or df.empty:
            return pd.DataFrame()
        return df

    def industry_map(self, codes):
        out = {}
        now = time.time()
        need = []
        for c in codes:
            c = str(c).zfill(6)
            hit = _IND_CACHE.get(c)
            if hit and now - hit[0] < _IND_TTL:
                out[c] = hit[1]
            else:
                need.append(c)
        if need:
            out.update(self._fetch_industry(need))
        return out

    def _fetch_industry(self, codes: list[str]) -> dict[str, str]:
        res: dict[str, str] = {}
        try:
            import requests
            url = "https://push2.eastmoney.com/api/qt/ulist.np/get"
            for i in range(0, len(codes), 60):
                batch = codes[i:i + 60]
                params = {
                    "fltt": "2",
                    "secids": ",".join(data_feed._secid(c) for c in batch),
                    "fields": "f12,f14,f100",
                    "ut": "fa5fd1943c7b386f172d6893dbfba10b",
                }
                r = requests.get(url, params=params, timeout=8,
                                 headers={"User-Agent": "Mozilla/5.0"})
                diff = (r.json().get("data") or {}).get("diff") or []
                if isinstance(diff, dict):
                    diff = [diff]
                for d in diff:
                    code = str(d.get("f12", "")).zfill(6)
                    ind = str(d.get("f100", "") or "").strip()
                    if code and ind and ind != "-":
                        res[code] = ind
                        _IND_CACHE[code] = (time.time(), ind)
        except Exception:
            pass
        return res

    def capabilities(self):
        return {
            "level": "近似",
            "has": ["日/周/月/分钟 K 线", "前复权价", "行业归属", "实时快照"],
            "missing": ["历史涨跌停价", "停牌标记", "财报公告日（PIT）",
                        "历史指数成分权重", "解禁日程"],
            "note": ("涨跌停按昨收×板块幅度推算，停牌用成交量为 0 判断，"
                     "财务数据是当前快照而非时点值"),
        }


class TushareProvider(Provider):
    """tushare pro。需要 token 与积分，但能拿到回测真正需要的那几张表。"""
    name = "tushare"
    label = "tushare pro"

    def __init__(self, token: str = ""):
        self.token = token or _load_cfg().get("tushare_token", "")

    def _pro(self):
        import tushare as ts
        if not self.token:
            raise RuntimeError("未配置 tushare token")
        ts.set_token(self.token)
        return ts.pro_api()

    def bars(self, code, start, end, adjust="qfq"):
        pro = self._pro()
        tc = _ts_code(code)
        d = pro.daily(ts_code=tc, start_date=start, end_date=end)
        if d is None or d.empty:
            return pd.DataFrame()
        a = pro.adj_factor(ts_code=tc, start_date=start, end_date=end)
        d = d.merge(a[["trade_date", "adj_factor"]], on="trade_date", how="left")
        d = d.sort_values("trade_date").reset_index(drop=True)
        base = d["adj_factor"].iloc[-1] if adjust == "qfq" else 1.0
        f = d["adj_factor"] / base if adjust == "qfq" else d["adj_factor"]
        out = pd.DataFrame({
            "date": pd.to_datetime(d["trade_date"]),
            "open": d["open"] * f, "high": d["high"] * f,
            "low": d["low"] * f, "close": d["close"] * f,
            "volume": d["vol"] * 100, "amount": d["amount"] * 1000,
            "pct_change": d["pct_chg"],
        })
        return out.sort_values("date").reset_index(drop=True)

    def industry_map(self, codes):
        try:
            pro = self._pro()
            basic = pro.stock_basic(fields="ts_code,name,industry")
            m = {str(r["ts_code"]).split(".")[0]: r["industry"]
                 for _, r in basic.iterrows()}
            return {str(c).zfill(6): m.get(str(c).zfill(6), "") for c in codes}
        except Exception:
            return {}

    def capabilities(self):
        return {
            "level": "时点正确",
            "has": ["日线+复权因子", "历史涨跌停价 stk_limit", "停牌 suspend_d",
                    "财报披露日 disclosure_date", "历史成分权重 index_weight",
                    "解禁日程 share_float", "技术因子", "筹码分布"],
            "missing": ["实时 tick（需更高权限）"],
            "note": "2000 积分可用基础 PIT；5000 积分起可用涨跌停价与技术因子",
        }


def _ts_code(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith(("6", "9")):
        return c + ".SH"
    if c[:1] in ("4", "8") or c[:3] == "920":
        return c + ".BJ"
    return c + ".SZ"


def _load_cfg() -> dict:
    try:
        with open(_CFG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cfg(**kw) -> None:
    cfg = _load_cfg()
    cfg.update(kw)
    try:
        os.makedirs(_CFG_DIR, exist_ok=True)
        with open(_CFG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


_ACTIVE: Provider | None = None


def active() -> Provider:
    global _ACTIVE
    if _ACTIVE is None:
        cfg = _load_cfg()
        name = cfg.get("provider", "eastmoney")
        token = cfg.get("tushare_token", "")
        if name == "tushare" and token:
            _ACTIVE = TushareProvider(token)
        else:
            _ACTIVE = EastMoneyProvider()
    return _ACTIVE


def switch(name: str, token: str = "") -> Provider:
    global _ACTIVE
    if name == "tushare":
        if not token:
            token = _load_cfg().get("tushare_token", "")
        if not token:
            raise RuntimeError("切到 tushare 需要先填 token")
        _ACTIVE = TushareProvider(token)
        save_cfg(provider="tushare", tushare_token=token)
    else:
        _ACTIVE = EastMoneyProvider()
        save_cfg(provider="eastmoney")
    return _ACTIVE


def capabilities() -> dict:
    return active().capabilities()


def status_line() -> str:
    p = active()
    cap = p.capabilities()
    return f"{p.label}｜数据档位：{cap.get('level', '')}"
