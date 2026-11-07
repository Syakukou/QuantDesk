"""截面选股：全市场因子打分排行 + 因子有效性检验。

口径纪律（A 股）：
- T 日收盘后计算因子，只使用当日收盘即可见的字段，不含未来信息
- T 日信号只能 T+1 开盘成交，因此检验收益一律「开盘到开盘」
- 不用收盘到收盘，那是美股口径，在 A 股会凭空多出半天的信息优势

设计取舍：
- 因子主料来自东财全市场快照（一次请求拿 5900+ 只票 23 个字段），
  不逐只拉日线，否则一轮要半小时以上
- 需要历史序列才能算的因子（波动率、距高点、量能斜率）只对初筛后的
  候选池补查，控制请求量
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".quantdesk", "cache")
SNAP_TTL = 6 * 3600  # 快照缓存 6 小时，盘后一天内复用


# ---------------------------------------------------------------- 工具

def _rankdata(a: np.ndarray) -> np.ndarray:
    """等价于 scipy.stats.rankdata(method='average')，不引 scipy。"""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # 处理并列
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """秩相关，缺失自动剔除。"""
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 8:
        return float("nan")
    rx, ry = _rankdata(x[m]), _rankdata(y[m])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else float("nan")


def mad_winsor(s: pd.Series, n: float = 5.0) -> pd.Series:
    """中位数绝对偏差去极值，比 3σ 更稳（收益分布厚尾）。"""
    v = pd.to_numeric(s, errors="coerce")
    med = v.median()
    mad = (v - med).abs().median()
    if not np.isfinite(mad) or mad <= 0:
        return v
    lo, hi = med - n * 1.4826 * mad, med + n * 1.4826 * mad
    return v.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    """截面百分位标准化，不是 (x-均值)/标准差。

    选股因子里「60 日涨幅」这类右偏严重的字段，标准差会被少数妖股撑大，
    其余四千多只票全挤在 0 附近，打分就失去区分度。转成百分位后，
    极端值最多只影响自己那一格，不会污染整体分布。
    """
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().sum() < 8:
        return pd.Series(0.0, index=s.index)
    r = v.rank(pct=True, na_option="keep")
    # 映射到约 ±1.73，与标准正态同量级，方便和原来的权重对比
    return ((r - 0.5) * 3.4641).fillna(0.0)


# ---------------------------------------------------------------- 快照

def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def snapshot(force: bool = False, progress=None) -> pd.DataFrame:
    """全市场快照。一次请求约 60 秒（东财分 59 页），因此落盘缓存。"""
    path = _cache_path("spot_a.csv")
    if not force and os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < SNAP_TTL:
            df = pd.read_csv(path, dtype={"代码": str})
            if len(df) > 3000:
                if progress:
                    progress(f"用本地快照缓存（{age / 60:.0f} 分钟前）")
                return df

    if progress:
        progress("正在拉全市场快照，约 60 秒…")
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------- 因子

FACTORS = {
    # 名称: (方向, 默认权重, 人话解释)
    # 权重不是拍脑袋来的：方向全部来自 2024-01 起全市场无偏样本的 IC 检验，
    # 结论是这批数据里动量整体反向、低波动占优、放量跑输。
    # 换样本区间后必须用 calibrate() 重算，别把旧符号当真理。
    "reversal": (+1, 0.8, "短期反转，当日跌得多的排前面"),
    "mom60": (-1, 0.6, "60 日涨幅取反，这批数据里追涨是亏的"),
    "ytd": (-1, 0.5, "年初至今涨幅取反"),
    "ep": (+1, 0.7, "市盈率倒数，越便宜分越高"),
    "bp": (+1, 0.5, "市净率倒数"),
    "size": (-1, 0.4, "对数总市值取反，小盘偏多"),
    "turnover": (-1, 0.5, "换手率取反，避开过热"),
    "amp": (-1, 0.5, "振幅取反，避开日内剧烈波动"),
    "volratio": (-1, 0.3, "量比取反，放量后往往跑输"),
    "liq": (+1, 0.4, "成交额对数，保证能进出"),
}


def build_panel(df: pd.DataFrame,
                exclude_st: bool = True,
                exclude_new: bool = True,
                min_amount: float = 3e7) -> pd.DataFrame:
    """把快照整理成因子面板，一票一行。"""
    d = df.copy()
    num = ["最新价", "涨跌幅", "成交额", "振幅", "换手率", "量比",
           "市盈率-动态", "市净率", "总市值", "流通市值", "60日涨跌幅",
           "年初至今涨跌幅", "昨收"]
    for c in num:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if exclude_st and "名称" in d.columns:
        st = d["名称"].astype(str).str.contains("ST|退", na=False)
        d = d[~st]

    # 次新股必须剔掉：它们的「60 日涨幅」实际是上市以来涨幅，会把动量因子带偏
    if "名称" in d.columns:
        nm = d["名称"].astype(str).str.replace(" ", "", regex=False)
        d = d[~nm.str.match(r"^[NC]")]

    # 停牌 / 一字板 / 无成交的票没法买，先剔掉
    if "最新价" in d.columns:
        d = d[d["最新价"].notna()]
    if "成交额" in d.columns:
        d = d[d["成交额"].fillna(0) >= min_amount]

    d = d[d["代码"].str.match(r"^(00|30|60|68|92)")]

    if exclude_new and "市盈率-动态" in d.columns:
        # 亏损票保留但 EP 记为缺失，不参与该项打分
        pass

    p = pd.DataFrame(index=d.index)
    p["code"] = d["代码"]
    p["name"] = d.get("名称", pd.Series("", index=d.index))
    p["price"] = d.get("最新价")
    p["chg"] = d.get("涨跌幅")
    p["amount"] = d.get("成交额")
    p["mv"] = d.get("总市值")

    p["mom60"] = d.get("60日涨跌幅")
    p["ytd"] = d.get("年初至今涨跌幅")
    p["reversal"] = -d.get("涨跌幅", pd.Series(np.nan, index=d.index))
    pe = d.get("市盈率-动态", pd.Series(np.nan, index=d.index))
    p["ep"] = np.where(pe > 0, 1.0 / pe, np.nan)
    pb = d.get("市净率", pd.Series(np.nan, index=d.index))
    p["bp"] = np.where(pb > 0, 1.0 / pb, np.nan)
    p["size"] = np.log(d.get("总市值", pd.Series(np.nan, index=d.index)).clip(lower=1e7))
    p["turnover"] = d.get("换手率")
    p["amp"] = d.get("振幅")
    p["liq"] = np.log(d.get("成交额", pd.Series(np.nan, index=d.index)).clip(lower=1e4))
    p["volratio"] = d.get("量比")

    return p.reset_index(drop=True)


def score(p: pd.DataFrame, weights: dict | None = None,
          top_n: int = 50) -> pd.DataFrame:
    """因子加权合成总分并排名。缺失因子按 0 处理（等于不表态）。"""
    w = dict({k: v[1] for k, v in FACTORS.items()})
    if weights:
        w.update(weights)

    total = pd.Series(0.0, index=p.index)
    wsum = 0.0
    for k, wt in w.items():
        if k not in p.columns or wt == 0:
            continue
        z = zscore(p[k])
        z = z.fillna(0.0)
        total += z * wt
        wsum += abs(wt)
        p[f"z_{k}"] = z

    p["factor_score"] = total / wsum if wsum else 0.0
    p = p.sort_values("factor_score", ascending=False).reset_index(drop=True)
    p["rank"] = np.arange(1, len(p) + 1)
    p["pct"] = p["rank"] / max(len(p), 1) * 100  # 前百分之几，越小越好

    # 入选项：排名前 top_n，给出一个整手可买的参考股数（按 10 万单票预算）
    p["lot_100k"] = (100000 / p["price"] / 100).fillna(0).astype(int) * 100
    return p


def explain(row: pd.Series, weights: dict | None = None) -> list[tuple[str, str, float]]:
    """拆解某一票的得分来源，回答「它凭什么排这么前」。"""
    w = dict({k: v[1] for k, v in FACTORS.items()})
    if weights:
        w.update(weights)
    out = []
    for k, wt in sorted(w.items(), key=lambda x: -abs(x[1])):
        col = f"z_{k}"
        if col not in row or wt == 0:
            continue
        z = float(row[col])
        contrib = z * wt
        if abs(z) < 0.15:
            continue
        out.append((FACTORS[k][2] if k in FACTORS else k,
                    "加分" if contrib > 0 else "减分", round(contrib, 2)))
    return out


# ---------------------------------------------------------------- IC 检验

def _open_returns(bars: dict[str, pd.DataFrame], hold: int) -> pd.DataFrame:
    """开盘到开盘的持有期收益宽表：T+1 开盘买入，T+1+hold 开盘卖出。"""
    cols = {}
    for code, df in bars.items():
        if df is None or df.empty or "open" not in df.columns:
            continue
        s = df.set_index("date")["open"].astype(float)
        s = s[~s.index.duplicated()]
        cols[code] = s
    if not cols:
        return pd.DataFrame()
    op = pd.DataFrame(cols).sort_index()
    return op.shift(-(1 + hold)) / op.shift(-1) - 1


def _factor_series(bars: dict[str, pd.DataFrame], window: int) -> pd.DataFrame:
    """在 T 日收盘可算的因子：过去 window 日涨幅。"""
    cols = {}
    for code, df in bars.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        s = df.set_index("date")["close"].astype(float)
        s = s[~s.index.duplicated()]
        cols[code] = s
    if not cols:
        return pd.DataFrame()
    cl = pd.DataFrame(cols).sort_index()
    return cl / cl.shift(window) - 1


def ic_report(bars: dict[str, pd.DataFrame], hold: int = 5,
              window: int = 20) -> dict:
    """因子有效性：逐期截面秩相关，输出 IC 均值、ICIR、胜率、分组收益。"""
    fac = _factor_series(bars, window)
    fwd = _open_returns(bars, hold)
    if fac.empty or fwd.empty:
        return {"ok": False, "msg": "样本不足，无法计算 IC"}

    idx = fac.index.intersection(fwd.index)
    if len(idx) < hold + 40:
        return {"ok": False, "msg": f"有效日期只有 {len(idx)} 天，太短"}

    ics, dates = [], []
    for dt in idx:
        x = fac.loc[dt].to_numpy(dtype=float)
        y = fwd.loc[dt].to_numpy(dtype=float)
        if np.isnan(x).sum() > len(x) * 0.6:
            continue
        ic = spearman(x, y)
        if np.isfinite(ic):
            ics.append(ic)
            dates.append(dt)

    if len(ics) < 20:
        return {"ok": False, "msg": f"有效截面只有 {len(ics)} 期"}

    arr = np.array(ics)
    mean_ic = float(arr.mean())
    std_ic = float(arr.std(ddof=1))
    icir = mean_ic / std_ic if std_ic else 0.0
    t_stat = icir * np.sqrt(len(arr))

    # 五分组：看多空是否单调
    q_rets = {i: [] for i in range(5)}
    for dt in dates:
        x = fac.loc[dt]
        y = fwd.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() < 20:
            continue
        xm, ym = x[m], y[m]
        try:
            q = pd.qcut(xm.rank(method="first"), 5, labels=False)
        except ValueError:
            continue
        for qi in range(5):
            sel = ym[q == qi]
            if len(sel):
                q_rets[qi].append(float(sel.mean()))

    groups = []
    for qi in range(5):
        v = q_rets[qi]
        groups.append({
            "group": f"Q{qi + 1}",
            "name": ["最差", "偏弱", "中性", "偏强", "最强"][qi],
            "avg_ret": round(float(np.mean(v)) * 100, 3) if v else 0.0,
            "n": len(v),
        })

    spread = (groups[4]["avg_ret"] - groups[0]["avg_ret"]) if groups else 0.0
    inv = sum(1 for i in range(4) if groups[i]["avg_ret"] > groups[i + 1]["avg_ret"] + 1e-9)
    mono = inv == 0
    quasi = inv <= 1

    return {
        "ok": True,
        "window": window,
        "hold": hold,
        "n_period": len(arr),
        "ic_mean": round(mean_ic, 4),
        "ic_std": round(std_ic, 4),
        "icir": round(float(icir), 3),
        "t_stat": round(float(t_stat), 2),
        "win_rate": round(float((arr > 0).mean()) * 100, 1),
        "groups": groups,
        "spread": round(spread, 3),
        "monotonic": mono,
        "quasi_monotonic": quasi,
        "verdict": _ic_verdict(mean_ic, icir, t_stat, quasi, spread),
    }


def _ic_verdict(mean_ic, icir, t_stat, mono, spread) -> str:
    if abs(t_stat) < 2:
        return (f"t 值只有 {t_stat:.1f}，统计上分不清它和噪声，"
                "不能拿这个因子单独做决策")
    side = "正" if mean_ic > 0 else "负"
    tone = "强" if abs(icir) > 0.5 else "中等" if abs(icir) > 0.3 else "弱"
    base = f"IC 均值 {mean_ic:.3f}，ICIR {icir:.2f}，{side}向预测{tone}"
    if mono:
        return base + f"；五分组收益单调，多空价差 {spread:.2f}%，可以用"
    return base + f"；但五分组不单调，多空价差 {spread:.2f}%，中间有噪音"


# ---------------------------------------------------------------- 多因子扫描

def _wide(bars: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    cols = {}
    for code, df in bars.items():
        if df is None or df.empty or field not in df.columns:
            continue
        s = df.set_index("date")[field].astype(float)
        s = s[~s.index.duplicated()]
        cols[code] = s
    return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()


def _build(cl: pd.DataFrame, vo: pd.DataFrame, kind: str, n: int):
    if kind == "momentum":
        return cl / cl.shift(n) - 1
    if kind == "reversal":
        return -(cl / cl.shift(n) - 1)
    if kind == "volatility":
        return cl.pct_change().rolling(n).std()
    if kind == "volume_ratio":
        return vo.rolling(max(n // 4, 3)).mean() / vo.rolling(n).mean().replace(0, np.nan)
    if kind == "dist_high":
        return cl / cl.rolling(n).max() - 1
    if kind == "ma_bias":
        return cl / cl.rolling(n).mean() - 1
    if kind == "amihud":
        return (cl.pct_change().abs() / vo.replace(0, np.nan)) * 1e9
    return None


SCAN_SPECS = [
    ("5日反转", "reversal", 5),
    ("20日动量", "momentum", 20),
    ("20日反转", "reversal", 20),
    ("60日动量", "momentum", 60),
    ("120日动量", "momentum", 120),
    ("20日波动率", "volatility", 20),
    ("60日波动率", "volatility", 60),
    ("量能放大", "volume_ratio", 20),
    ("距60日高点", "dist_high", 60),
    ("20日乖离", "ma_bias", 20),
]


def _ic_stats(fac: pd.DataFrame, fwd: pd.DataFrame) -> dict:
    idx = fac.index.intersection(fwd.index)
    if len(idx) < 30:
        return {"ok": False}
    ics = []
    for dt in idx:
        x = fac.loc[dt].to_numpy(dtype=float)
        y = fwd.loc[dt].to_numpy(dtype=float)
        if np.isnan(x).sum() > len(x) * 0.6:
            continue
        v = spearman(x, y)
        if np.isfinite(v):
            ics.append(v)
    if len(ics) < 20:
        return {"ok": False}
    a = np.array(ics)
    m, sd = float(a.mean()), float(a.std(ddof=1))
    icir = m / sd if sd else 0.0
    return {
        "ok": True,
        "n": len(a),
        "ic": m,
        "icir": icir,
        "t": icir * np.sqrt(len(a)),
        "win": float((a > 0).mean()),
    }


def factor_scan(bars: dict[str, pd.DataFrame], hold: int = 5) -> pd.DataFrame:
    """同一批股票、同一口径下横向比较各因子，用来决定权重要不要信它。"""
    cl = _wide(bars, "close")
    vo = _wide(bars, "volume")
    fwd = _open_returns(bars, hold)
    if cl.empty or fwd.empty:
        return pd.DataFrame()

    rows = []
    for name, kind, n in SCAN_SPECS:
        fac = _build(cl, vo, kind, n)
        if fac is None or fac.empty:
            continue
        st = _ic_stats(fac, fwd)
        if not st.get("ok"):
            rows.append({"因子": name, "IC均值": np.nan, "ICIR": np.nan,
                         "t值": np.nan, "胜率%": np.nan, "结论": "样本不足"})
            continue
        if abs(st["t"]) < 2:
            verdict = "不能单独用（与噪声分不开）"
        elif abs(st["icir"]) > 0.3:
            verdict = ("正向有效" if st["ic"] > 0 else "反向有效") + "，值得进模型"
        else:
            verdict = ("弱正向" if st["ic"] > 0 else "弱反向") + "，只做辅助"
        rows.append({
            "因子": name,
            "IC均值": round(st["ic"], 4),
            "ICIR": round(st["icir"], 3),
            "t值": round(st["t"], 2),
            "胜率%": round(st["win"] * 100, 1),
            "结论": verdict,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["abs_t"] = df["t值"].abs()
        df = df.sort_values("abs_t", ascending=False).drop(columns=["abs_t"])
    return df.reset_index(drop=True)


# 扫描因子（日线口径）到快照因子（当天口径）的映射。
# 严格对应的只有 60 日动量；其余是近似，标注出来，别假装精确。
SCAN_TO_FACTOR = {
    "60日动量": ("mom60", "严格"),
    "120日动量": ("ytd", "近似（年初至今 vs 120 日）"),
    "5日反转": ("reversal", "近似（5 日 vs 当日）"),
    "20日波动率": ("amp", "近似（收盘波动 vs 日内振幅）"),
    "量能放大": ("volratio", "近似（5/20 均量 vs 量比）"),
}


def calibrate(scan: pd.DataFrame) -> tuple[dict, list[dict]]:
    """按 IC 检验结果给因子定权重：方向随 IC 符号，大小随 |ICIR|。

    t 值不到 2 的直接给 0 权重，宁可不用，也不拿噪声当信号。
    """
    weights, detail = {}, []
    if scan is None or scan.empty:
        return weights, detail

    for _, r in scan.iterrows():
        m = SCAN_TO_FACTOR.get(r["因子"])
        if not m:
            continue
        key, match = m
        if pd.isna(r["ICIR"]) or pd.isna(r["t值"]):
            continue
        ic, icir, t = float(r["IC均值"]), float(r["ICIR"]), float(r["t值"])
        if abs(t) < 2:
            w = 0.0
            why = "t<2，不给权重"
        else:
            # 0.35 的 ICIR 视为"够用"，映射到满权重 1.0，其余按比例
            w = float(np.sign(ic) * min(abs(icir) / 0.35, 1.0))
            why = ("正向" if ic > 0 else "反向") + f"，ICIR {icir:.2f}"
        weights[key] = round(w, 2)
        detail.append({
            "因子": r["因子"], "对应字段": key, "口径": match,
            "IC": round(ic, 4), "t值": round(t, 2),
            "权重": round(w, 2), "理由": why,
        })
    return weights, detail
