"""组合风险 X 光：回答「我这堆票到底是不是一个票」。

券商会告诉你每只票的盈亏、总市值、仓位比例，但它不会告诉你：
  · 你买了 20 只票，实际有效分散度可能只有 6 只
  · 你的组合里两只票相关性 0.92，一起涨一起跌，属于重复下注
  · 95% 概率下，明天最差会亏多少（VaR）
  · 你赚的钱是选股赚的，还是行业押对了

这些都是持仓层面的问题，跟「今天涨没涨」无关，所以券商客户端不做。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

TRADING_DAYS = 242.0


def _returns(bars: pd.DataFrame) -> pd.Series:
    if bars is None or bars.empty or "close" not in bars:
        return pd.Series(dtype=float)
    s = bars.copy()
    s["d"] = pd.to_datetime(s["date"]).dt.normalize()
    s = s.drop_duplicates("d").set_index("d")["close"].astype(float)
    return s.pct_change()


def analyze(holdings: list[dict], bars_lookup, industry_of=None,
            lookback: int = 250) -> dict:
    """holdings: [{code, name, qty, cost_price, price?}]

    返回 weights/hhi/corr/clusters/var/contrib/industry/warns。
    """
    rows = []
    rets = {}
    for h in holdings:
        code = h["code"]
        qty = float(h.get("qty") or 0)
        if qty <= 0:
            continue
        bars = None
        try:
            bars = bars_lookup(code)
        except Exception:
            bars = None
        px = h.get("price")
        if not px and bars is not None and not bars.empty:
            px = float(bars["close"].iloc[-1])
        px = float(px or h.get("cost_price") or 0)
        mv = qty * px
        cost = qty * float(h.get("cost_price") or px)
        r = _returns(bars).tail(lookback) if bars is not None else pd.Series(dtype=float)
        if len(r) > 30:
            rets[code] = r
        rows.append({
            "code": code, "name": h.get("name") or code,
            "qty": qty, "price": px, "mv": mv, "cost": cost,
            "pnl": mv - cost, "pnl_pct": (mv / cost - 1) * 100 if cost else 0.0,
            "industry": (industry_of(code) if industry_of else "") or "未分类",
            "has_bars": code in rets,
        })

    total = sum(r["mv"] for r in rows)
    if total <= 0:
        return {"ok": False, "msg": "没有有效持仓"}
    for r in rows:
        r["weight"] = r["mv"] / total * 100
    rows.sort(key=lambda x: -x["weight"])

    weights = [r["mv"] / total for r in rows]
    hhi = float(sum(w * w for w in weights))
    eff_n = 1.0 / hhi if hhi > 0 else 0.0

    # 行业暴露
    ind = defaultdict(float)
    for r in rows:
        ind[r["industry"]] += r["mv"]
    industry = sorted(
        [{"name": k, "weight": round(v / total * 100, 2), "mv": round(v, 0)}
         for k, v in ind.items()], key=lambda x: -x["weight"])

    # 相关性
    corr_rows = []
    clusters = []
    var_block = {}
    port_ret = None
    if len(rets) >= 2:
        mat = pd.DataFrame(rets).sort_index()
        mat = mat.dropna(how="all")
        mat = mat.fillna(0.0)
        cm = mat.corr()
        codes = [r["code"] for r in rows if r["code"] in cm.columns]
        # 没行情的票不能参与组合收益计算，权重要在可用集合内重新归一
        _mv = {r["code"]: r["mv"] for r in rows if r["code"] in cm.columns}
        _sum = sum(_mv.values()) or 1.0
        wmap = {k: v / _sum for k, v in _mv.items()}
        nm_map = {x["code"]: x["name"] for x in rows}
        for i, a in enumerate(codes):
            for b in codes[i + 1:]:
                c = float(cm.loc[a, b])
                if c >= 0.6:
                    corr_rows.append({"a": a, "a_name": nm_map.get(a, a),
                                      "b": b, "b_name": nm_map.get(b, b),
                                      "corr": round(c, 2)})
        corr_rows.sort(key=lambda x: -x["corr"])

        # 并查集做相关性聚类
        parent = {c: c for c in codes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for row in corr_rows:
            if row["corr"] >= 0.6:
                ra, rb = find(row["a"]), find(row["b"])
                if ra != rb:
                    parent[ra] = rb
        buckets = defaultdict(list)
        for c in codes:
            buckets[find(c)].append(c)
        for root, members in buckets.items():
            if len(members) >= 2:
                wsum = sum(wmap[m] for m in members)
                nm = {x["code"]: x["name"] for x in rows}
                clusters.append({
                    "members": [{"code": m, "name": nm.get(m, m)} for m in members],
                    "weight": round(wsum * 100, 2),
                    "avg_corr": round(float(np.mean(
                        [abs(cm.loc[a, b]) for a in members for b in members if a < b])), 2)
                    if len(members) > 1 else 0.0,
                })
        clusters.sort(key=lambda x: -x["weight"])

        # 组合日收益与 VaR
        w = np.array([wmap[c] for c in codes])
        port = (mat[codes] * w).sum(axis=1)
        port_ret = port
        if len(port) > 30:
            var_block = _var_block(port, total)

        # 风险贡献
        cov = mat[codes].cov()
        sig_p = float(np.sqrt(w @ cov.values @ w))
        if sig_p > 1e-12:
            mrc = (cov.values @ w) / sig_p
            contrib = w * mrc / sig_p
            contrib_map = {c: round(float(x) * 100, 1) for c, x in zip(codes, contrib)}
            for r in rows:
                r["risk_contrib"] = contrib_map.get(r["code"], 0.0)

    # 集中度告警
    warns = []
    if rows:
        top = rows[0]
        if top["weight"] >= 30:
            warns.append(f"{top['name']} 单票 {top['weight']:.0f}%，一个跌停就是组合 -{top['weight'] / 10:.1f}%")
        if industry and industry[0]["weight"] >= 50:
            warns.append(f"{industry[0]['name']} 一个行业占 {industry[0]['weight']:.0f}%，这是行业押注不是组合")
        if eff_n < len(rows) * 0.4:
            warns.append(f"{len(rows)} 只票的有效分散度只有 {eff_n:.1f} 只，大部分仓位是重复的")
        for c in clusters:
            if c["weight"] >= 40:
                names = "、".join(m["name"] for m in c["members"][:4])
                warns.append(f"{names} 高度同涨同跌（相关性 {c['avg_corr']}），合计 {c['weight']:.0f}%")
                break
        rc_rows = [x for x in rows if x.get("risk_contrib")]
        if rc_rows:
            top_rc = max(rc_rows, key=lambda x: x["risk_contrib"])
            if top_rc["risk_contrib"] >= 40 and top_rc["weight"] < 40:
                warns.append(
                    f"{top_rc['name']} 仓位只占 {top_rc['weight']:.0f}%，"
                    f"却贡献了 {top_rc['risk_contrib']:.0f}% 的组合波动")
        if var_block.get("var95", 0) > 3:
            warns.append(f"95% 概率单日亏损不超过 {var_block['var95']:.1f}%，这个波动不适合满仓")
    if not warns:
        warns.append("集中度、行业暴露、相关性三项都没有明显异常")

    return {
        "ok": True,
        "rows": rows,
        "total_mv": round(total, 0),
        "hhi": round(hhi, 3),
        "eff_n": round(eff_n, 1),
        "n": len(rows),
        "industry": industry,
        "corr": corr_rows[:15],
        "clusters": clusters,
        "var": var_block,
        "warns": warns,
        "n_with_bars": len(rets),
    }


def _var_block(port: pd.Series, total_mv: float) -> dict:
    r = port.dropna()
    if len(r) < 30:
        return {}
    var95 = -float(np.percentile(r, 5)) * 100
    var99 = -float(np.percentile(r, 1)) * 100
    tail = r[r <= np.percentile(r, 5)]
    es95 = -float(tail.mean()) * 100 if len(tail) else var95
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) * 100
    worst = r.nsmallest(5)
    return {
        "var95": round(var95, 2),
        "var99": round(var99, 2),
        "es95": round(es95, 2),
        "ann_vol": round(ann_vol, 2),
        "var95_amt": round(var95 / 100 * total_mv, 0),
        "worst_days": [{"date": d.strftime("%Y-%m-%d"), "ret": round(v * 100, 2)}
                       for d, v in worst.items()],
        "days": len(r),
    }


def stress(port_ret: pd.Series | None, shocks=(-3, -5, -8)) -> list[dict]:
    """情景压力：假设组合整体跟随大盘下跌，亏多少钱。"""
    if port_ret is None or len(port_ret) < 30:
        return []
    beta = 1.0          # 无基准时按 1 倍处理，界面上会标明是线性外推
    out = []
    for s in shocks:
        out.append({"shock": s, "ret": round(s * beta, 2)})
    return out
