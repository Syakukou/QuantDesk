"""多市场统一净值：把 A 股、港股、美股、ETF、可转债、基金折算成人民币，拼成一条曲线。

为什么券商做不到这件事：券商按账户分市场展示，港股通账户、A 股账户、场外基金
各自一条曲线，没人替你合成。而真实的组合风险来自「合起来之后」。

这里做了三件券商不给的事：
  1. 统一币种：港股按 HKD/CNY、美股按 USD/CNY 逐日折算，不是用当前汇率一刀切
  2. 汇率分解：把海外资产的收益拆成「标的涨了多少」和「汇率帮了多少」
  3. 再平衡对照：同一组权重，日频再平衡 vs 买入持有，差多少

两种用法：
  · 配置模拟：给定各市场权重，看这个配置历史上会走成什么样
  · 当前持仓：把你现在的持仓按当前市值权重回溯

汇率来自 forex_hist_em 的 USDCNH 与 USDHKD 两条序列，HKD/CNY 由两者相除得到。
不用 HKDCNH 是因为它只有 2025 年以后的 400 多条，做长周期回测不够。

每个标的都带一个 kind（取数类型），不能只看代码猜：000300 是沪深300指数，
000001 既是平安银行也是华夏成长，靠六位数字分辨不出来。
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import data_feed
import metrics

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".quantdesk", "cache", "fx")
TTL = 12 * 3600

# kind -> (显示市场, 币种)
KIND_INFO = {
    "cn_index": ("A股", "CNY"),
    "cn_stock": ("A股", "CNY"),
    "etf": ("ETF", "CNY"),
    "hk_index": ("港股", "HKD"),
    "hk_stock": ("港股", "HKD"),
    "us_index": ("美股", "USD"),
    "us_stock": ("美股", "USD"),
    "cov_index": ("可转债", "CNY"),
    "fund": ("基金", "CNY"),
}

# 可选标的：显示市场 / 名称 / kind / 代码
REPRESENTATIVES = [
    ("A股", "沪深300", "cn_index", "000300"),
    ("A股", "中证500", "cn_index", "000905"),
    ("A股", "中证1000", "cn_index", "000852"),
    ("A股", "上证50", "cn_index", "000016"),
    ("港股", "恒生指数", "hk_index", "HSI"),
    ("港股", "恒生科技", "hk_index", "HSTECH"),
    ("美股", "标普500", "us_index", "SPX"),
    ("美股", "纳斯达克", "us_index", "IXIC"),
    ("可转债", "中证转债", "cov_index", "000832"),
    ("ETF", "黄金ETF", "etf", "518880"),
    ("ETF", "沪深300ETF", "etf", "510300"),
    ("基金", "华夏成长", "fund", "000001"),
]

_DEFAULT_MIX = [("cn_index", "000300", 40.0, "沪深300"),
                ("hk_index", "HSI", 20.0, "恒生指数"),
                ("us_index", "SPX", 20.0, "标普500"),
                ("cov_index", "000832", 20.0, "中证转债")]

_DATE_ALIAS = ["日期", "date", "净值日期", "时间"]
_CLOSE_ALIAS = ["收盘", "close", "收盘价", "最新价", "latest", "单位净值", "今收"]


def _cached(name: str, fn, force: bool = False, ttl: int = TTL):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name)
    if not force and os.path.exists(path):
        if time.time() - os.path.getmtime(path) < ttl:
            try:
                return pd.read_pickle(path)
            except Exception:  # noqa: BLE001
                pass
    df = fn()
    try:
        df.to_pickle(path)
    except Exception:  # noqa: BLE001
        pass
    return df


def _pick(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def _norm(df) -> pd.DataFrame:
    """把各家接口五花八门的列名压成 date / close 两列。"""
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return pd.DataFrame(columns=["date", "close"])
    d = _pick(df, _DATE_ALIAS)
    c = _pick(df, _CLOSE_ALIAS)
    if d is None or c is None:
        return pd.DataFrame(columns=["date", "close"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df[d], errors="coerce"),
        "close": pd.to_numeric(df[c], errors="coerce"),
    })
    out = out.dropna(subset=["date", "close"])
    out = out[out["close"] > 0]
    return out.sort_values("date").drop_duplicates("date").reset_index(drop=True)


# ------------------------------------------------------------------ 汇率

def fx_series(start: str, end: str, force: bool = False) -> pd.DataFrame:
    """人民币汇率。返回 date / usdcny / usdhkd / hkdcny 四列。"""
    import akshare as ak

    usd = _cached("fx_usdcnh.pkl",
                  lambda: ak.forex_hist_em(symbol="USDCNH"), force=force)
    hkd = _cached("fx_usdhkd.pkl",
                  lambda: ak.forex_hist_em(symbol="USDHKD"), force=force)
    a = _norm(usd).rename(columns={"close": "usdcny"})
    b = _norm(hkd).rename(columns={"close": "usdhkd"})
    if a.empty:
        raise RuntimeError("美元兑人民币汇率取不到，多市场折算无法进行")
    fx = a.merge(b, on="date", how="left")
    # 港币盯住美元，USDHKD 缺失时用联系汇率中枢补，误差在 0.5% 以内
    fx["usdhkd"] = fx["usdhkd"].ffill().bfill().fillna(7.80)
    fx["hkdcny"] = fx["usdcny"] / fx["usdhkd"]
    lo, hi = pd.to_datetime(start), pd.to_datetime(end)
    fx = fx[(fx["date"] >= lo) & (fx["date"] <= hi)]
    if len(fx) < 2:
        raise RuntimeError(f"汇率序列在 {start}–{end} 区间内只有 {len(fx)} 条，太短")
    return fx.reset_index(drop=True)


# ------------------------------------------------------------------ 行情

def _fetch(kind: str, code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    s, e = start.replace("-", ""), end.replace("-", "")
    if kind == "cn_index":
        df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=s, end_date=e)
    elif kind == "cov_index":
        df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=s, end_date=e)
    elif kind == "cn_stock":
        df = data_feed.get_history(code, s, e)
    elif kind == "etf":
        df = ak.fund_etf_hist_em(symbol=code, period="daily",
                                 start_date=s, end_date=e, adjust="qfq")
    elif kind == "hk_index":
        df = ak.stock_hk_index_daily_em(symbol=code.upper())
    elif kind == "hk_stock":
        df = ak.stock_hk_hist(symbol=code, period="daily",
                              start_date=s, end_date=e, adjust="qfq")
    elif kind == "us_index":
        sym = {"SPX": ".INX", "INX": ".INX", "DJI": ".DJI",
               "IXIC": ".IXIC"}.get(code.upper(), code)
        df = ak.index_us_stock_sina(symbol=sym)
    elif kind == "us_stock":
        df = ak.stock_us_hist(symbol=code, period="daily",
                              start_date=s, end_date=e, adjust="qfq")
    elif kind == "fund":
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
    else:
        raise ValueError(f"未知取数类型 {kind}")

    out = _norm(df)
    if out.empty:
        return out
    lo, hi = pd.to_datetime(start), pd.to_datetime(end)
    return out[(out["date"] >= lo) & (out["date"] <= hi)].reset_index(drop=True)


def history(kind: str, code: str, start: str, end: str,
            force: bool = False) -> pd.DataFrame:
    key = f"px_{kind}_{code}_{start}_{end}.pkl".replace(":", "").replace("/", "")
    return _cached(key, lambda: _fetch(kind, code, start, end), force=force)


# ------------------------------------------------------------------ 合成

def _align(series: dict[str, pd.DataFrame], fx: pd.DataFrame) -> pd.DataFrame:
    """铺到同一日期轴上，缺失日向前填充，只保留所有序列都有价的区间。

    用并集而不是交集：港股在 A 股春节休市时照常交易，取交集会直接丢掉这些
    天的涨跌。代价是某市场休市那天它的贡献是 0，波动率会略被低估，结论里写明。
    """
    dates = sorted(set().union(*[set(s["date"]) for s in series.values()])
                   | set(fx["date"]))
    grid = pd.DataFrame({"date": pd.Series(dates, dtype="datetime64[ns]")})
    for k, s in series.items():
        grid = grid.merge(s.rename(columns={"close": k}), on="date", how="left")
        grid[k] = grid[k].ffill()
    grid = grid.merge(fx, on="date", how="left")
    for c in ("usdcny", "usdhkd", "hkdcny"):
        grid[c] = grid[c].ffill().bfill()
    # 任一序列还没开始的早期日期直接切掉，避免半截净值
    grid = grid.dropna(subset=list(series.keys())).reset_index(drop=True)
    return grid


def blend(mix: list[tuple[str, str, float, str]], start: str, end: str,
          bench: tuple[str, str] | None = None, force: bool = False) -> dict:
    """mix: [(kind, code, weight, name)]，返回统一净值与分解。"""
    mix = [m for m in mix if float(m[2]) > 0]
    if not mix:
        return {"ok": False, "msg": "权重全为 0，没有可算的东西"}

    fx = fx_series(start, end, force=force)
    series, names, kinds, missing = {}, {}, {}, []
    for kind, code, _w, name in mix:
        key = f"{kind}:{code}"
        try:
            s = history(kind, code, start, end, force=force)
        except Exception:  # noqa: BLE001
            s = pd.DataFrame()
        if s.empty or len(s) < 20:
            missing.append(f"{name or code}")
            continue
        series[key] = s
        names[key] = name or code
        kinds[key] = kind
    if not series:
        return {"ok": False, "msg": "所选标的都没取到行情，检查代码或网络"}

    grid = _align(series, fx)
    if len(grid) < 20:
        return {"ok": False, "msg": f"对齐后只有 {len(grid)} 个交易日，太短"}

    wsum = sum(float(m[2]) for m in mix if f"{m[0]}:{m[1]}" in series) or 1.0

    rows = []
    navs = {}
    for kind, code, w, name in mix:
        key = f"{kind}:{code}"
        if key not in series:
            continue
        market, ccy = KIND_INFO.get(kind, ("其他", "CNY"))
        px = grid[key]
        if ccy == "HKD":
            rate = grid["hkdcny"]
        elif ccy == "USD":
            rate = grid["usdcny"]
        else:
            rate = pd.Series(1.0, index=grid.index)
        p0, r0 = float(px.iloc[0]), float(rate.iloc[0])
        nav = (px / p0) * (rate / r0)
        navs[key] = nav.to_numpy(dtype=float)

        ret_local = float(px.iloc[-1] / p0 - 1.0)
        ret_cny = float(nav.iloc[-1] - 1.0)
        ret_fx = ((1.0 + ret_cny) / (1.0 + ret_local) - 1.0
                  if ret_local > -0.999 else 0.0)
        rows.append({
            "key": key, "market": market, "kind": kind, "code": code,
            "name": names[key], "ccy": ccy, "w": float(w) / wsum,
            "ret_local": ret_local, "ret_cny": ret_cny, "fx": ret_fx,
            "contrib": float(w) / wsum * ret_cny,
            "first": pd.Timestamp(series[key]["date"].iloc[0]).strftime("%Y-%m-%d"),
            "nav": nav.to_numpy(dtype=float),
        })

    keys = list(navs.keys())
    # 买入持有：权重随价格漂移，不再动
    bh = np.zeros(len(grid))
    for r in rows:
        bh = bh + r["w"] * r["nav"]
    # 日频再平衡：每天把权重拉回目标。这样每天的组合收益就是
    # Σ 目标权重 × 当日涨跌，是「再平衡收益」的理想上界。
    reb = np.ones(len(grid))
    for i in range(1, len(grid)):
        gross = 0.0
        for r in rows:
            prev_px, now_px = r["nav"][i - 1], r["nav"][i]
            if prev_px > 1e-12:
                gross += r["w"] * (now_px / prev_px)
        reb[i] = reb[i - 1] * gross

    idx = pd.DatetimeIndex(grid["date"])
    nav_reb = pd.Series(reb, index=idx)
    nav_bh = pd.Series(bh, index=idx)

    bench_nav, bench_name = None, ""
    if bench:
        try:
            bk, bc = bench
            bs = history(bk, bc, start, end, force=force)
            if not bs.empty:
                bser = pd.Series(bs["close"].to_numpy(dtype=float),
                                 index=pd.DatetimeIndex(bs["date"]))
                bser = bser.reindex(idx).ffill().bfill()
                bench_nav = (bser / bser.iloc[0]).to_numpy(dtype=float)
                bench_name = _rep_name(bk, bc)
        except Exception:  # noqa: BLE001
            pass

    m = metrics.nav_metrics(nav_reb)
    m_bh = metrics.nav_metrics(nav_bh)

    corr = {}
    if len(keys) >= 2:
        mat = pd.DataFrame({k: pd.Series(v, index=idx) for k, v in navs.items()})
        cm = mat.pct_change().corr()
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                v = float(cm.loc[a, b])
                if np.isfinite(v):
                    corr[f"{names[a]}|{names[b]}"] = round(v, 2)

    fx_total = sum(r["w"] * r["fx"] for r in rows)
    return {
        "ok": True,
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "nav": reb.tolist(),
        "nav_bh": bh.tolist(),
        "bench": bench_nav.tolist() if bench_nav is not None else None,
        "bench_name": bench_name,
        "series": rows,
        "metrics": m,
        "metrics_bh": m_bh,
        "corr": corr,
        "fx_total": fx_total,
        "missing": missing,
        "verdict": _verdict(rows, m, m_bh, corr, fx_total, missing),
    }


def _rep_name(kind: str, code: str) -> str:
    for _mkt, n, k, c in REPRESENTATIVES:
        if k == kind and c == code:
            return n
    return code


def _verdict(rows, m, m_bh, corr, fx_total, missing) -> list[dict]:
    out = []
    # 权重太小的标的贡献会四舍五入成 0.00，选出来没有意义，先滤掉
    solid = [r for r in rows if r["w"] >= 0.02] or rows
    top = max(solid, key=lambda r: r["contrib"])
    bot = min(solid, key=lambda r: r["contrib"])
    out.append({
        "title": "谁在赚钱，谁在拖后腿",
        "text": (f"{top['name']} 贡献 {top['contrib'] * 100:+.2f} 个百分点"
                 f"（权重 {top['w'] * 100:.1f}%，本币涨跌 {top['ret_local'] * 100:+.1f}%）；"
                 f"{bot['name']} 贡献 {bot['contrib'] * 100:+.2f} 个百分点"
                 f"（权重 {bot['w'] * 100:.1f}%，本币涨跌 {bot['ret_local'] * 100:+.1f}%）"),
    })

    fx_rows = [r for r in rows if r["ccy"] != "CNY"]
    if fx_rows:
        fb = max(fx_rows, key=lambda r: r["fx"])
        out.append({
            "title": "汇率帮了多少",
            "text": (f"外币资产折算后，汇率整体影响 {fx_total * 100:+.2f} 个百分点。"
                     f"其中 {fb['name']} 的汇率效应 {fb['fx'] * 100:+.1f}%"
                     f"（本币收益 {fb['ret_local'] * 100:+.1f}%，"
                     f"折人民币后 {fb['ret_cny'] * 100:+.1f}%）"),
        })
    else:
        out.append({"title": "汇率帮了多少",
                    "text": "全部资产都是人民币计价，没有汇率影响"})

    if corr:
        hi = max(corr.items(), key=lambda kv: kv[1])
        lo = min(corr.items(), key=lambda kv: kv[1])
        out.append({
            "title": "分散是不是真的",
            "text": (f"最低相关是 {lo[0].replace('|', ' 与 ')} {lo[1]}，"
                     f"最高是 {hi[0].replace('|', ' 与 ')} {hi[1]}。"
                     + ("全部低于 0.6，跨市场确实分散了风险"
                        if hi[1] < 0.6 else
                        "有市场对高度同涨同跌，名义上分散、实际还是同一个风险")),
        })

    d = m["annual_return"] - m_bh["annual_return"]
    out.append({
        "title": "再平衡值不值",
        "text": (f"日频再平衡年化 {m['annual_return'] * 100:.2f}%，"
                 f"买入不动 {m_bh['annual_return'] * 100:.2f}%，差 {d * 100:+.2f} 个百分点。"
                 + ("再平衡在这组资产里是加分的，涨跌不同步才有再平衡收益"
                    if d > 0.002 else
                    "再平衡没加分，说明这几块资产步调太一致，或波动不够大")
                 + "。日频是上界，实际按季度做会少一截"),
    })

    rec = m.get("dd_recover_days", 0)
    rec_txt = ("到数据末尾仍未回本" if rec == 0 else f"扛 {rec} 个交易日才回本")
    out.append({
        "title": "最难受的时候",
        "text": (f"最大回撤 {m['max_drawdown'] * 100:.2f}%，"
                 f"从 {m.get('dd_start') or '-'} 跌到 {m.get('dd_end') or '-'}，"
                 f"{rec_txt}；最长连续 {m.get('longest_underwater', 0)} 天在水下"),
    })

    if m.get("days", 0) < 180:
        out.insert(0, {
            "title": "先说样本",
            "text": (f"只有 {m.get('days', 0)} 天、{m.get('n', 0)} 个交易日。"
                     f"年化数字是把这几个月的涨跌放大成一年，参考价值有限，"
                     f"看总收益和回撤更实在"),
        })

    if missing:
        out.append({
            "title": "没算进去的",
            "text": "、".join(missing) + " 没取到行情，权重已剔除并重新归一",
        })
    return out


# ------------------------------------------------------------------ 持仓模式

def classify(code: str) -> tuple[str, str]:
    """按代码判断 (kind, 显示市场)。只用于交割单里的实际持仓。

    券商的代码不统一：港股有 HK03996，可转债有 754129 / 371132 这种配售代码，
    都不是能直接取行情的交易代码，得分清楚再取数。
    """
    c = str(code).strip().upper()
    if c.startswith("HK"):
        return "hk_stock", "港股"
    if c.startswith("US"):
        return "us_stock", "美股"
    if c.startswith(("SH", "SZ", "BJ")):
        c = c[2:]
    if any(ch.isalpha() for ch in c):
        return "us_stock", "美股"
    if len(c) == 5 and c.isdigit():
        return "hk_stock", "港股"
    if len(c) == 6 and c.isdigit():
        # 11/12 是转债交易代码，75/37/07 是配售申购代码，都归到可转债
        if c[:2] in ("11", "12", "75", "37", "07"):
            return "cov_index", "可转债"
        if c[0] == "5" or c[:2] in ("15", "16", "18"):
            return "etf", "ETF"
        return "cn_stock", "A股"
    return "fund", "基金"


def norm_code(code: str) -> str:
    """把港股/美股代码收拾成 akshare 认识的形状。"""
    c = str(code).strip().upper()
    if c.startswith("HK"):
        return c[2:].zfill(5)
    return c


def from_positions(positions: list[dict], start: str, end: str,
                   force: bool = False, grace_days: int = 30) -> dict:
    """用当前持仓按市值权重回溯。positions 需含 code / name / qty / cost_price。

    这里有个坑：只要持仓里有一只刚上市的票，整条曲线的起点就会被它拖到最后。
    所以先探一遍每只票的行情覆盖，把「起点晚于区间开头太多」的剔出去单独列明，
    保住完整的回溯区间。剔掉的是少数，换来的是两年曲线而不是两个月。
    """
    if not positions:
        return {"ok": False, "msg": "没有持仓"}
    raw, skipped, dropped = [], [], []
    lo = pd.to_datetime(start)
    for p in positions:
        code = str(p.get("code", "")).strip()
        qty = float(p.get("qty") or 0)
        px = float(p.get("cost_price") or 0)
        name = p.get("name", code)
        if qty <= 0 or px <= 0:
            continue
        kind, market = classify(code)
        if kind == "cov_index":
            skipped.append(f"{name}（可转债请用配置模拟，或单独看轮动页）")
            continue
        code = norm_code(code)
        try:
            s = history(kind, code, start, end, force=force)
        except Exception:  # noqa: BLE001
            s = pd.DataFrame()
        if s.empty or len(s) < 30:
            dropped.append(f"{name}（无行情）")
            continue
        first = pd.Timestamp(s["date"].iloc[0])
        if first > lo + pd.Timedelta(days=grace_days):
            dropped.append(f"{name}（{first.strftime('%Y-%m-%d')} 才有行情）")
            continue
        raw.append((kind, code, qty * px, name, market))
    if not raw:
        return {"ok": False,
                "msg": "持仓里没有一只能覆盖所选区间的标的。"
                       "把开始日期往后调，或看配置模拟"}
    total = sum(r[2] for r in raw)
    mix = [(k, c, w / total * 100, n) for k, c, w, n, _m in raw]
    res = blend(mix, start, end, force=force)
    if not res.get("ok"):
        return res
    res["mode"] = "positions"
    res["skipped"] = skipped
    res["dropped"] = dropped
    res["n_positions"] = len(raw)
    res["coverage"] = sum(r[2] for r in raw)
    note = f"{len(raw)} 只持仓参与回溯，权重按成本市值归一。"
    if dropped:
        note += f" 剔除 {len(dropped)} 只：{'、'.join(dropped)}。"
    res["window_note"] = note
    return res


def default_mix() -> list[tuple[str, str, float, str]]:
    return list(_DEFAULT_MIX)


def presets() -> list[tuple[str, list[tuple[str, str, float, str]]]]:
    """几个常用配置，省得每次手填。"""
    return [
        ("股债四六（A股+转债）", [
            ("cn_index", "000300", 60.0, "沪深300"),
            ("cov_index", "000832", 40.0, "中证转债")]),
        ("全球四等分", [
            ("cn_index", "000300", 25.0, "沪深300"),
            ("hk_index", "HSI", 25.0, "恒生指数"),
            ("us_index", "SPX", 25.0, "标普500"),
            ("cov_index", "000832", 25.0, "中证转债")]),
        ("中美港三地", [
            ("cn_index", "000300", 40.0, "沪深300"),
            ("hk_index", "HSTECH", 30.0, "恒生科技"),
            ("us_index", "IXIC", 30.0, "纳斯达克")]),
        ("进攻偏科技", [
            ("cn_index", "000852", 30.0, "中证1000"),
            ("hk_index", "HSTECH", 30.0, "恒生科技"),
            ("us_index", "IXIC", 40.0, "纳斯达克")]),
    ]
