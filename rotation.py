"""可转债轮动回测：双低 / 低价 / 低溢价 / 低纯债溢价，定期调仓，含真实成本。

为什么值得单独做：券商客户端能看可转债行情和溢价率，但没有一个会把
「按双低排序、每周换仓、扣掉手续费之后还剩多少」这条曲线画出来。
而可转债是 A 股里少数有明确规则、又天然带期权价值的品种，规则化程度高。

数据用的是 bond_zh_cov_value_analysis，每只债一条历史序列，直接带转股溢价率。
这样不用自己算转股价修正，也不会有下修导致的计算错误。

回测口径：
  · 调仓日收盘价成交，扣双边成本（可转债无印花税，佣金约万 0.2）
  · 用当日可见的数据打分，不偷看未来
  · 强赎/退市：数据断在哪天就以那天的收盘价冻结，下一期调仓剔除
  · 生存偏差：池子里包含已退市的债，不是只拿今天还活着的

基准是中证转债指数 000832。
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import metrics

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".quantdesk", "cache", "cov")
TTL = 12 * 3600
BENCH_CODE = "000832"
BENCH_NAME = "中证转债"

STRATEGIES = {
    "double_low": {
        "name": "双低",
        "desc": "价格 + 转股溢价率×100，经典双低，兼顾债性与股性",
        "cols": ["price", "premium"],
    },
    "low_price": {
        "name": "低价",
        "desc": "只看价格，越低越买，吃债底保护",
        "cols": ["price"],
    },
    "low_premium": {
        "name": "低溢价",
        "desc": "只看转股溢价率，越低越买，跟涨能力强但下跌也快",
        "cols": ["premium"],
    },
    "bond_floor": {
        "name": "低纯债溢价",
        "desc": "离债底越近越安全，纯防守",
        "cols": ["bond_premium"],
    },
}

FREQS = {"W": "每周", "2W": "每两周", "M": "每月"}

# 信用评级由高到低。低价债里混着要违约的，所以给一个可选的评级门槛。
_RATING_ORDER = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
                 "BBB+", "BBB", "BBB-", "BB", "B", "CCC", "CC", "C"]


def _rating_rank(s: str) -> int:
    s = str(s or "").upper().replace("STI", "").strip()
    return _RATING_ORDER.index(s) if s in _RATING_ORDER else len(_RATING_ORDER)


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


def universe(force: bool = False) -> pd.DataFrame:
    """全部可转债清单（含已退市，避免生存偏差）。"""
    import akshare as ak

    def fetch():
        df = ak.bond_zh_cov()
        out = pd.DataFrame({
            "code": df["债券代码"].astype(str).str.zfill(6),
            "name": df["债券简称"].astype(str),
            "listed": pd.to_datetime(df["上市时间"], errors="coerce"),
            "rating": df.get("信用评级", pd.Series("", index=df.index)).astype(str),
            "size": pd.to_numeric(df.get("发行规模"), errors="coerce"),
        })
        return out.dropna(subset=["listed"]).reset_index(drop=True)

    return _cached("universe.pkl", fetch, force=force)


def _one(code: str, force: bool = False) -> pd.DataFrame:
    """单只转债的历史收盘与转股溢价率。落盘缓存，否则每次都要重拉 700 只。"""
    def fetch():
        import akshare as ak

        d = ak.bond_zh_cov_value_analysis(symbol=code)
        if d is None or len(d) == 0:
            return pd.DataFrame()
        out = pd.DataFrame({
            "date": pd.to_datetime(d["日期"], errors="coerce"),
            "price": pd.to_numeric(d["收盘价"], errors="coerce"),
            "premium": pd.to_numeric(d.get("转股溢价率"), errors="coerce"),
            "bond_premium": pd.to_numeric(d.get("纯债溢价率"), errors="coerce"),
            "conv_value": pd.to_numeric(d.get("转股价值"), errors="coerce"),
        })
        # 溢价率缺失时用转股价值回算，比丢掉这一行强
        need = out["premium"].isna() & (out["conv_value"] > 0) & (out["price"] > 0)
        out.loc[need, "premium"] = (out.loc[need, "price"]
                                    / out.loc[need, "conv_value"] * 100.0 - 100.0)
        out = out.dropna(subset=["date", "price"])
        out = out[out["price"] > 0]
        # 极端的溢价率多半是转股价异常，留在池子里会把双低因子带偏
        out.loc[out["premium"].abs() > 500, "premium"] = np.nan
        return out.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    d = _cached(f"b_{code}.pkl", fetch, force=force)
    return d if isinstance(d, pd.DataFrame) else pd.DataFrame()


def build_panel(start: str, end: str, progress=None, workers: int = 8,
                force: bool = False, max_bonds: int | None = None) -> dict:
    """拉全部转债历史，构造成 {code: DataFrame}。首次约 25 秒，之后走缓存。"""
    uni = universe(force=force)
    lo, hi = pd.to_datetime(start), pd.to_datetime(end)
    # 区间内存在过的债：上市日早于区间结束（上市前没有行情，但也不必排除退市的）
    cand = uni[uni["listed"] <= hi].reset_index(drop=True)
    if max_bonds:
        cand = cand.head(max_bonds)

    panel, failed = {}, []
    total = len(cand)
    done = 0
    t0 = time.time()

    def job(row):
        try:
            return row.code, _one(row.code, force=force)
        except Exception:  # noqa: BLE001
            return row.code, None

    rows = list(cand.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(job, r): r for r in rows}
        for f in as_completed(futs):
            code, d = f.result()
            done += 1
            if d is not None and len(d) >= 5:
                d = d[(d["date"] >= lo) & (d["date"] <= hi)]
                if len(d) >= 5:
                    panel[code] = d.reset_index(drop=True)
                else:
                    failed.append(code)
            else:
                failed.append(code)
            if progress and done % 25 == 0:
                progress(done, total, f"{done}/{total} 只，已取得 {len(panel)} 只")

    meta = cand.set_index("code")
    return {
        "panel": panel, "meta": meta, "failed": failed,
        "n": len(panel), "cost": round(time.time() - t0, 1),
        "start": start, "end": end,
    }


def _wide(panel: dict, field: str) -> pd.DataFrame:
    cols = {}
    for code, d in panel.items():
        s = d.set_index("date")[field]
        cols[code] = s[~s.index.duplicated(keep="last")]
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def _rebalance_dates(idx: pd.DatetimeIndex, freq: str) -> list:
    """按周末 / 隔周 / 月末取调仓日，用区间内最后一个交易日。"""
    s = pd.Series(idx, index=idx)
    if freq == "M":
        keys = idx.to_period("M")
    elif freq == "2W":
        # ISO 周号取偶数周
        iso = idx.isocalendar()
        keys = (iso["year"].astype(int) * 100
                + (iso["week"].astype(int) // 2))
        keys = pd.Index(keys)
    else:
        iso = idx.isocalendar()
        keys = pd.Index(iso["year"].astype(int) * 100 + iso["week"].astype(int))
    g = pd.Series(idx, index=keys)
    return list(g.groupby(level=0).max().sort_values())


def backtest(panel: dict, meta: pd.DataFrame, strategy: str = "double_low",
             freq: str = "W", top_n: int = 15, cost_rate: float = 0.0007,
             max_price: float = 150.0, min_price: float = 80.0,
             min_rating: str = "", bench: pd.DataFrame | None = None) -> dict:
    """按因子排序定期换仓。返回净值、逐期持仓、超额与判读。

    min_rating 传 "AA-" 这类门槛时，评级更低的债不入选。注意评级取的是当前值，
    回测早期用它带一点前视偏差，所以默认关闭，开了要在界面上说明。
    """
    if not panel:
        return {"ok": False, "msg": "没有可用的转债数据"}
    px = _wide(panel, "price")
    fields = {f: _wide(panel, f) for f in ("premium", "bond_premium")}
    if px.empty or px.shape[1] < top_n:
        return {"ok": False, "msg": f"可用转债只有 {px.shape[1]} 只，少于持仓数 {top_n}"}

    allow = None
    if min_rating:
        thr = _rating_rank(min_rating)
        allow = {c for c in px.columns
                 if c in meta.index and _rating_rank(meta.loc[c, "rating"]) <= thr}
        if len(allow) < top_n:
            return {"ok": False,
                    "msg": f"评级 {min_rating} 以上的只有 {len(allow)} 只，不够 {top_n} 只"}

    dates = px.index
    reb_dates = set(_rebalance_dates(dates, freq))
    spec = STRATEGIES.get(strategy, STRATEGIES["double_low"])

    # 打分矩阵。注意 premium 本身已经是百分数（15.33 表示 15.33%），
    # 双低的正确算法是「价格（元）+ 溢价率（百分点）」，不能再乘 100。
    if strategy == "double_low":
        score = px + fields["premium"]
    elif strategy == "low_price":
        score = px.copy()
    elif strategy == "low_premium":
        score = fields["premium"].copy()
    else:
        score = fields["bond_premium"].copy()


    shares: dict[str, float] = {}
    lastp: dict[str, float] = {}
    cash = 1.0
    navs, hold_log, periods = [], [], []
    turnover_sum = 0.0
    fee_sum = 0.0
    first = True

    for d in dates:
        # 当日估值：退市/停牌用最后一次有效价冻结，收益记 0
        val = cash
        for code, sh in shares.items():
            p = px.at[d, code] if code in px.columns else np.nan
            if pd.notna(p):
                lastp[code] = float(p)
            val += sh * lastp.get(code, 0.0)
        if val <= 1e-9:
            break
        navs.append(val)

        if d not in reb_dates and not (first and not shares):
            continue

        row = score.loc[d]
        price_row = px.loc[d]
        ok = (row.notna() & price_row.notna()
              & (price_row >= min_price) & (price_row <= max_price))
        if allow is not None:
            ok &= pd.Series(row.index.isin(allow), index=row.index)
        pick = list(row[ok].sort_values().index[:top_n])

        # 现有持仓的市值权重，用于算换手与成本
        old_val = {}
        for code, sh in shares.items():
            old_val[code] = sh * lastp.get(code, 0.0)
        old_w = {c: v / val for c, v in old_val.items() if v > 0}
        new_w = {c: 1.0 / len(pick) for c in pick} if pick else {}

        if shares:
            allc = set(old_w) | set(new_w)
            delta = sum(abs(new_w.get(c, 0.0) - old_w.get(c, 0.0)) for c in allc)
            turn = delta / 2.0
            fee = cost_rate * delta
            val *= (1.0 - fee)
        else:
            turn, fee = 1.0, cost_rate
            val *= (1.0 - cost_rate)

        turnover_sum += turn
        fee_sum += fee
        # 每一期都留一份持仓明细：界面上点调仓记录要能看到「那一期到底买了什么」，
        # 只存最新一期的话，历史换仓永远只能看汇总数。
        prem_row = fields["premium"].loc[d]
        holds = []
        for rank, code in enumerate(pick, start=1):
            pm = prem_row.get(code)
            holds.append({
                "rank": rank, "code": code,
                "name": str(meta.loc[code, "name"]) if code in meta.index else code,
                "price": round(float(price_row[code]), 2),
                "premium": round(float(pm), 2) if pm is not None and pd.notna(pm) else None,
                "score": round(float(row[code]), 2),
                "rating": str(meta.loc[code, "rating"]) if code in meta.index else "",
            })
        periods.append({
            "date": d.strftime("%Y-%m-%d"),
            "n": len(pick),
            "avg_price": round(float(price_row[pick].mean()), 2) if pick else 0.0,
            "avg_premium": (round(float(fields["premium"].loc[d, pick].mean()), 2)
                            if pick else 0.0),
            "turnover": round(turn * 100, 1),
            "nav": round(val, 4),
            "holdings": holds,
        })
        hold_log.append({"date": d.strftime("%Y-%m-%d"), "codes": list(pick)})

        # 按目标权重重建持仓
        shares, cash = {}, 0.0
        if pick:
            per = val / len(pick)
            for c in pick:
                p = float(price_row[c])
                if p > 0:
                    shares[c] = per / p
        else:
            cash = val
        first = False
        navs[-1] = val

    if not navs:
        return {"ok": False, "msg": "回测期间没有可成交的标的"}
    # 末段补齐（最后几个交易日不再调仓，净值继续按市值走）
    for d in dates[len(navs):]:
        val = cash
        for code, sh in shares.items():
            p = px.at[d, code] if code in px.columns else np.nan
            if pd.notna(p):
                lastp[code] = float(p)
            val += sh * lastp.get(code, 0.0)
        navs.append(val)

    ser = pd.Series(navs, index=dates)
    m = metrics.nav_metrics(ser)

    bench_nav, bench_m = None, None
    if bench is not None and len(bench):
        bs = bench.set_index("date")["close"].reindex(dates).ffill().bfill()
        if bs.notna().sum() > 2:
            bench_nav = (bs / bs.iloc[0]).tolist()
            bench_m = metrics.nav_metrics(bs / bs.iloc[0])

    # 最新一期持仓明细
    last = periods[-1]["date"] if periods else None
    holdings = []
    if last:
        d = pd.Timestamp(last)
        row = score.loc[d]
        price_row = px.loc[d]
        ok = (row.notna() & price_row.notna()
              & (price_row >= min_price) & (price_row <= max_price))
        if allow is not None:
            ok &= pd.Series(row.index.isin(allow), index=row.index)
        ranked = row[ok].sort_values()
        for r, code in enumerate(ranked.index[:top_n], start=1):
            nm = meta.loc[code, "name"] if code in meta.index else code
            holdings.append({
                "rank": r, "code": code, "name": str(nm),
                "price": round(float(price_row[code]), 2),
                "premium": round(float(fields["premium"].loc[d, code]), 2)
                if pd.notna(fields["premium"].loc[d, code]) else None,
                "score": round(float(row[code]), 2),
                "rating": str(meta.loc[code, "rating"]) if code in meta.index else "",
            })

    stat = {
        "rebalances": len(periods),
        "avg_turnover": round(turnover_sum / max(len(periods), 1) * 100, 1),
        "avg_price": round(float(np.mean([p["avg_price"] for p in periods])), 2)
        if periods else 0.0,
        "avg_premium": round(float(np.mean([p["avg_premium"] for p in periods])), 2)
        if periods else 0.0,
        "avg_n": round(float(np.mean([p["n"] for p in periods])), 1) if periods else 0,
        "fee_sum": round(fee_sum * 100, 2),
    }
    # 每年被手续费吃掉多少：总费用按年摊
    yrs = max(m.get("days", 1), 1) / 365.25
    stat["fee_annual"] = round(fee_sum / yrs * 100, 2) if yrs > 0 else 0.0

    excess = None
    if bench_m:
        excess = {
            "annual": m["annual_return"] - bench_m["annual_return"],
            "total": m["total_return"] - bench_m["total_return"],
            "dd_diff": m["max_drawdown"] - bench_m["max_drawdown"],
        }

    return {
        "ok": True,
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "nav": navs,
        "bench": bench_nav,
        "bench_name": BENCH_NAME if bench_nav else "",
        "metrics": m,
        "bench_metrics": bench_m,
        "excess": excess,
        "periods": periods,
        "holdings": holdings,
        "stats": stat,
        "strategy": spec["name"],
        "freq": FREQS.get(freq, freq),
        "top_n": top_n,
        "verdict": _verdict(m, bench_m, excess, stat, spec, freq),
    }


def _verdict(m, bm, ex, stat, spec, freq) -> list[dict]:
    fname = FREQS.get(freq, freq)
    out = [{
        "title": "结论",
        "text": (f"{spec['name']}策略{fname}调仓，{m.get('n', 0)} 个交易日"
                 f"年化 {m['annual_return'] * 100:.2f}%，"
                 f"波动 {m['vol'] * 100:.2f}%，夏普 {m['sharpe']:.2f}，"
                 f"最大回撤 {m['max_drawdown'] * 100:.2f}%"),
    }]
    if bm and ex:
        out.append({
            "title": "跟中证转债比",
            "text": (f"基准年化 {bm['annual_return'] * 100:.2f}%，"
                     f"策略 {m['annual_return'] * 100:.2f}%，"
                     f"超额 {ex['annual'] * 100:+.2f} 个百分点；"
                     f"回撤比基准{'小' if ex['dd_diff'] < 0 else '大'} "
                     f"{abs(ex['dd_diff']) * 100:.2f} 个百分点。"
                     + ("跑赢基准，且回撤更小，说明因子有效"
                        if ex["annual"] > 0 and ex["dd_diff"] < 0 else
                        "有超额但回撤更大，赚的是风险溢价，不是免费的午餐"
                        if ex["annual"] > 0 else
                        "没跑赢基准，这个因子的收益盖不住交易成本")),
        })
    out.append({
        "title": "换手与成本",
        "text": (f"共调仓 {stat['rebalances']} 次，平均每次换手 {stat['avg_turnover']}%，"
                 f"平均持仓 {stat['avg_n']} 只，均价 {stat['avg_price']} 元、"
                 f"平均溢价率 {stat['avg_premium']:.1f}%。"
                 f"手续费累计吃掉 {stat.get('fee_sum', 0):.2f} 个百分点，"
                 f"折合每年 {stat.get('fee_annual', 0):.2f}%。"
                 + ("换手偏高，频率再降一档可能更划算"
                    if stat["avg_turnover"] > 60 else
                    "换手在合理范围，成本没有明显侵蚀收益")),
    })
    out.append({
        "title": "这个策略怕什么",
        "text": ("可转债轮动最怕两件事：一是集体杀溢价，所有债一起跌，"
                 "分散没用；二是信用风险，低价债里混着要违约的，"
                 "所以别看价格低就买，还得看正股有没有退市风险。"
                 "另外强赎公告出来后正股和转债都会跳水，规则化策略反应不过来"),
    })
    return out


def load_bench(start: str, end: str, force: bool = False) -> pd.DataFrame:
    import akshare as ak

    def fetch():
        d = ak.index_zh_a_hist(symbol=BENCH_CODE, period="daily",
                               start_date=start.replace("-", ""),
                               end_date=end.replace("-", ""))
        return pd.DataFrame({
            "date": pd.to_datetime(d["日期"]),
            "close": pd.to_numeric(d["收盘"], errors="coerce"),
        }).dropna()

    return _cached(f"bench_{BENCH_CODE}_{start}_{end}.pkl", fetch, force=force)
