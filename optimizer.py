"""组合优化：等权 / 最小方差 / 风险平价 / 最大夏普 / Black-Litterman。

不引 scipy，全部用 numpy 实现：
- 风险平价用循环坐标下降（CCD），每步解一个一元二次方程，收敛稳定
- 最小方差用投影梯度，可行域是单纯形（权重和为 1、非负）
- 最大夏普先解析求切点组合，再投影回单纯形
- Black-Litterman 用贝叶斯公式把观点并进先验收益

为什么要自己写而不装 cvxpy：打包体积会翻倍，而这里的问题规模小（几十只票），
手写求解足够，还能看清每一步在做什么。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------- 统计量

def returns_matrix(bars: dict[str, pd.DataFrame],
                   min_obs: int = 60) -> pd.DataFrame:
    """把各票日线收盘价对齐成收益矩阵，行是日期、列是代码。"""
    cols = {}
    for code, df in bars.items():
        if df is None or len(df) < min_obs or "close" not in df.columns:
            continue
        s = df.set_index("date")["close"].astype(float)
        s = s[~s.index.duplicated()].sort_index()
        cols[code] = s
    if not cols:
        return pd.DataFrame()
    px = pd.DataFrame(cols).sort_index()
    px = px.ffill()
    ret = px.pct_change()
    # 数据起点不齐，砍掉前段
    ret = ret.iloc[int(len(ret) * 0.1):] if len(ret) > 120 else ret
    ret = ret.dropna(how="all")
    keep = [c for c in ret.columns if ret[c].notna().sum() >= min_obs]
    return ret[keep].dropna(axis=0, how="all")


def annual_stats(ret: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """年化收益用几何口径，不用「日均×252」。

    线性年化遇到北交所这种 ±30% 涨跌停的票会算出 -200% 这种不存在的数字，
    几何年化最差也就 -100%，不会突破下界。
    """
    n = max(len(ret), 1)
    growth = (1 + ret).prod()
    mu = growth ** (TRADING_DAYS / n) - 1
    mu = mu.replace([np.inf, -np.inf], np.nan).fillna(-1.0).to_numpy()
    cov = (ret.cov() * TRADING_DAYS).to_numpy()
    cov = (cov + cov.T) / 2
    return mu, cov


def _cov_reg(cov: np.ndarray, shrink: float = 0.05) -> np.ndarray:
    """协方差向对角收缩。样本量不够时，原始协方差矩阵会把噪声放大成'分散化'。"""
    n = cov.shape[0]
    diag = np.diag(np.diag(cov))
    return (1 - shrink) * cov + shrink * diag + np.eye(n) * 1e-10


# ---------------------------------------------------------------- 求解器

def _project_simplex(v: np.ndarray) -> np.ndarray:
    """把向量投影到单纯形上（权重非负且和为 1）。"""
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0)


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def min_variance(cov: np.ndarray, steps: int = 4000, lr: float | None = None) -> np.ndarray:
    """最小方差：梯度 w'Σw 关于 w 是 2Σw，投影梯度下降。"""
    n = cov.shape[0]
    w = equal_weight(n)
    scale = np.trace(cov) / n
    lr = lr or min(0.5 / (scale * n + 1e-12), 1.0)
    for _ in range(steps):
        g = 2 * cov @ w
        step = lr * g
        w = _project_simplex(w - step)
    return w


def risk_parity(cov: np.ndarray, iters: int = 800) -> np.ndarray:
    """风险平价：循环坐标下降，令每个资产的边际风险贡献相等。

    等价于解 min 0.5 w'Σw - (1/n)Σln(w_i)，对第 i 个坐标是一元二次方程，
    取正根，逐坐标扫到收敛。
    """
    n = cov.shape[0]
    w = np.ones(n) / n
    target = 1.0 / n
    for _ in range(iters):
        for i in range(n):
            a = cov[i, i]
            if a <= 0:
                continue
            b = w @ cov[i, :] - cov[i, i] * w[i]
            c = -target
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            w[i] = (-b + np.sqrt(disc)) / (2 * a)
        s = w.sum()
        if s > 0:
            w = w / s
    return w


def max_sharpe(cov: np.ndarray, mu: np.ndarray, rf: float = 0.02) -> np.ndarray:
    """最大夏普：无约束切点组合 w ∝ Σ⁻¹(μ-rf)，再投影回单纯形。

    超额收益全为负时（没有谁跑得过大盘无风险利率），退化成最小方差，
    因为此时"最大化夏普"在数学上无解，硬解会给出全押一只的荒谬结果。
    """
    n = cov.shape[0]
    excess = mu - rf
    if np.all(excess <= 1e-9):
        return min_variance(cov)
    try:
        raw = np.linalg.solve(cov, excess)
    except np.linalg.LinAlgError:
        return min_variance(cov)
    if np.all(raw <= 0):
        return min_variance(cov)
    raw = np.maximum(raw, 0)
    if raw.sum() <= 0:
        return equal_weight(n)
    return _project_simplex(raw / raw.sum())


def black_litterman(cov: np.ndarray, mkt_w: np.ndarray,
                    views: list[dict] | None = None,
                    tau: float = 0.05, delta: float = 2.5,
                    rf: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Black-Litterman：市场均衡收益作为先验，叠加你的观点。

    views 每条形如 {"long": [0, 1], "long_w": [0.6, 0.4],
                    "short": [], "ret": 0.12, "conf": 0.5}
    含义：认为「代码0 六成 + 代码1 四成」这个组合未来一年涨 12%，把握 0.5。
    """
    n = cov.shape[0]
    w_mkt = np.asarray(mkt_w, dtype=float)
    w_mkt = w_mkt / w_mkt.sum() if w_mkt.sum() > 0 else equal_weight(n)

    pi = delta * cov @ w_mkt  # 反向优化得到均衡超额收益

    if not views:
        return pi, w_mkt

    k = len(views)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    conf = np.zeros(k)
    for i, v in enumerate(views):
        for j, wt in zip(v.get("long", []), v.get("long_w", [1.0] * len(v.get("long", [])))):
            if j < n:
                P[i, j] += wt
        for j in v.get("short", []):
            if j < n:
                P[i, j] -= 1.0
        Q[i] = float(v.get("ret", 0.0))
        conf[i] = max(min(float(v.get("conf", 0.5)), 0.95), 0.05)

    tau_cov = tau * cov
    # 观点越有把握，Ω 越小，后验越靠近观点
    omega = np.diag(np.diag(P @ tau_cov @ P.T)) / np.maximum(conf, 0.05)

    try:
        inv_tau = np.linalg.inv(tau_cov)
        inv_om = np.linalg.inv(omega)
        A = inv_tau + P.T @ inv_om @ P
        b = inv_tau @ pi + P.T @ inv_om @ Q
        mu_bl = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return pi, w_mkt
    return mu_bl, w_mkt


# ---------------------------------------------------------------- 组合评估

def evaluate(w: np.ndarray, cov: np.ndarray, mu: np.ndarray,
             rf: float = 0.02, names: list[str] | None = None) -> dict:
    var = float(w @ cov @ w)
    vol = float(np.sqrt(max(var, 0)))
    ret = float(w @ mu)
    return {
        "ret": round(ret * 100, 2),
        "vol": round(vol * 100, 2),
        "sharpe": round((ret - rf) / vol, 2) if vol > 1e-9 else 0.0,
        "hhi": round(float((w ** 2).sum()), 3),
        "eff_n": round(1.0 / float((w ** 2).sum()), 1) if (w ** 2).sum() > 0 else 0,
        "top_w": round(float(w.max()) * 100, 1),
    }


def risk_contrib(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """各资产对组合波动的贡献占比，加总为 100%。"""
    var = float(w @ cov @ w)
    if var <= 0:
        return np.zeros_like(w)
    mrc = cov @ w
    return w * mrc / var


# ---------------------------------------------------------------- 主入口

METHODS = {
    "等权": "equal",
    "最小方差": "minvar",
    "风险平价": "rp",
    "最大夏普": "ms",
    "BL（含观点）": "bl",
}


def optimize(holdings: list[dict], bars_lookup,
             method: str = "rp", start: str = "20240101", end: str = "",
             views: list[dict] | None = None,
             rf: float = 0.02, max_weight: float = 0.25) -> dict:
    """holdings: [{code, name, mv}]；返回各方法权重对比与调仓建议。"""
    end = end or pd.Timestamp.now().strftime("%Y%m%d")
    codes = [h["code"] for h in holdings]
    names = {h["code"]: h.get("name", h["code"]) for h in holdings}

    bars = {}
    for c in codes:
        try:
            df = bars_lookup(c, start, end)
            if df is not None and len(df) >= 60:
                bars[c] = df
        except Exception:  # noqa: BLE001
            continue

    ret = returns_matrix(bars)
    if ret.empty or ret.shape[1] < 3:
        return {"ok": False, "msg": f"可用行情只有 {ret.shape[1]} 只，至少需要 3 只"}

    # 北交所 ±30% 涨跌停，年化波动常超 100%，放进来会主导整个风险预算。
    # 不是不看它，而是它需要单独一套风控，不能和主板混在一个优化器里。
    vol_ann = ret.std() * np.sqrt(TRADING_DAYS)
    wild = [c for c in ret.columns if vol_ann.get(c, 0) > 1.0]
    if wild:
        ret = ret.drop(columns=wild)

    if ret.shape[1] < 3:
        return {"ok": False,
                "msg": f"剔除极端波动票后只剩 {ret.shape[1]} 只，无法优化"}

    usable = list(ret.columns)
    mu, cov = annual_stats(ret)
    cov = _cov_reg(cov)

    mv = {h["code"]: float(h.get("mv", 0) or 0) for h in holdings}
    cur = np.array([mv.get(c, 0) for c in usable], dtype=float)
    if cur.sum() <= 0:
        cur = equal_weight(len(usable))
    else:
        cur = cur / cur.sum()

    w_eq = equal_weight(len(usable))
    w_mv = min_variance(cov)
    w_rp = risk_parity(cov)
    w_ms = max_sharpe(cov, mu, rf)
    mu_bl, w_mkt = black_litterman(
        cov, np.array([mv.get(c, 1) for c in usable]), views, rf=rf)
    w_bl = max_sharpe(cov, mu_bl, rf)

    # 单票上限约束：超限部分按比例分给未超限的，避免一刀切后权重和不为 1
    def cap(wts: np.ndarray) -> np.ndarray:
        w = wts.copy()
        for _ in range(60):
            over = w > max_weight
            if not over.any():
                break
            excess = (w[over] - max_weight).sum()
            w[over] = max_weight
            room = ~over
            if w[room].sum() <= 0:
                break
            w[room] += excess * (w[room] / w[room].sum())
        return w / w.sum()

    cand = {
        "当前持仓": cur,
        "等权": w_eq,
        "最小方差": cap(w_mv),
        "风险平价": cap(w_rp),
        "最大夏普": cap(w_ms),
        "BL（含观点）": cap(w_bl),
    }

    rows = []
    for label, w in cand.items():
        st = evaluate(w, cov, mu, rf)
        st["方案"] = label
        rows.append(st)

    chosen = cand.get(METHODS.get(method, "风险平价"), cand["风险平价"])
    rc = risk_contrib(chosen, cov)

    detail = pd.DataFrame({
        "code": usable,
        "name": [names.get(c, c) for c in usable],
        "当前%": np.round(cur * 100, 2),
        "建议%": np.round(chosen * 100, 2),
        "调整pp": np.round((chosen - cur) * 100, 2),
        "年化收益%": np.round(mu * 100, 2),
        "年化波动%": np.round(np.sqrt(np.diag(cov)) * 100, 2),
        "风险贡献%": np.round(rc * 100, 1),
        "夏普": np.round((mu - rf) / np.maximum(np.sqrt(np.diag(cov)), 1e-9), 2),
    }).sort_values("建议%", ascending=False)

    cur_stat = evaluate(cur, cov, mu, rf)
    new_stat = evaluate(chosen, cov, mu, rf)
    turnover = float(np.abs(chosen - cur).sum() / 2)

    return {
        "ok": True,
        "method": method,
        "rows": rows,
        "detail": detail,
        "n": len(usable),
        "missing": [c for c in codes if c not in usable],
        "wild": [names.get(c, c) for c in wild],
        "turnover": round(turnover * 100, 1),
        "verdict": _verdict(cur_stat, new_stat, turnover * 100, method),
        "current": cur_stat,
        "target": new_stat,
    }


def _verdict(cur: dict, new: dict, turnover: float, method: str) -> list[dict]:
    """turnover 是百分比（55.1 表示 55.1%），内部所有阈值都按百分比比较。"""
    out = []
    dvol = new["vol"] - cur["vol"]
    dret = new["ret"] - cur["ret"]
    out.append({
        "title": "按这套权重，波动会怎么变",
        "text": (f"当前组合年化波动 {cur['vol']}%，按{method}调到目标权重后 {new['vol']}%，"
                 f"{'降' if dvol < 0 else '升'} {abs(dvol):.2f} 个百分点"),
    })
    out.append({
        "title": "收益预期（别当真）",
        "text": (f"年化收益 {cur['ret']}% → {new['ret']}%，夏普 {cur['sharpe']} → {new['sharpe']}。"
                 "这个收益是用历史均值外推的，马科维茨的老毛病就是拿过去当未来，"
                 "它的调仓建议要信风险结构，别信收益数字"),
    })
    # 单边换手 → 卖出付千一印花税 + 双边万三佣金
    cost = turnover * 0.0016
    out.append({
        "title": "要付多少调仓成本",
        "text": (f"单边换手 {turnover:.1f}%，印花税加佣金加滑点约 {cost:.2f}% 的净资产"),
    })
    if dvol < -1.5 and turnover > 40:
        out.append({
            "title": "该不该动",
            "text": (f"波动能降 {abs(dvol):.1f} 个点，但要换掉 {turnover:.0f}% 的仓位。"
                     "如果不是马上要降风险敞口，分两到三批走更划算"),
        })
    elif turnover < 15:
        out.append({"title": "该不该动",
                    "text": f"换手只有 {turnover:.1f}%，现有结构已经接近最优，不用为调而调"})
    else:
        out.append({
            "title": "该不该动",
            "text": f"换手 {turnover:.0f}%，属于中等偏大，建议先看调整幅度最大的三只再决定",
        })
    return out
