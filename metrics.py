"""绩效指标：交易级、组合级、稳健性三块。

旧版只有 8 个数字（总收益、年化、回撤、胜率、笔数、夏普）。问题是这 8 个
数字回答不了「策略靠什么赚钱」——是靠一笔运气，还是靠 60% 的小胜堆出来的；
是年化 30% 但中途水下两年，还是每年稳稳 12%。

所以这一层分三块：
  · 交易级：每笔的持股天数、成本、MFE/MAE、回吐、退出原因
  · 组合级：风险调整后收益、回撤形态、盈亏结构、资金效率
  · 稳健性：滚动样本外、参数敏感性、蒙特卡洛、成本压力
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 242.0
RISK_FREE = 0.02


def nav_metrics(nav: pd.Series, risk_free: float = RISK_FREE) -> dict:
    """从一条归一化净值序列算组合级指标。

    与 summarize() 的区别：那个要吃回测引擎的 trades，这个只吃净值。
    多市场净值、可转债轮动这类「先得到净值再说交易」的场景用它。
    """
    s = pd.Series(nav).astype(float).dropna()
    if len(s) < 3 or s.iloc[0] <= 0:
        return {"total_return": 0.0, "annual_return": 0.0, "vol": 0.0,
                "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
                "calmar": 0.0, "days": 0, "n": len(s)}
    base = float(s.iloc[0])
    final = float(s.iloc[-1])
    total = final / base - 1.0
    days = max((s.index[-1] - s.index[0]).days, 1)
    years = days / 365.25
    cagr = (final / base) ** (1.0 / years) - 1.0 if years > 0 and final > 0 else 0.0
    rets = s.pct_change().dropna()
    vol = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) > 2 else 0.0
    dn = rets[rets < 0]
    dvol = float(dn.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(dn) > 2 else 0.0
    sharpe = (cagr - risk_free) / vol if vol > 1e-9 else 0.0
    sortino = (cagr - risk_free) / dvol if dvol > 1e-9 else 0.0
    dd = max_drawdown_detail(s)
    mdd = abs(dd["max_drawdown"])
    return {
        "total_return": total,
        "annual_return": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": cagr / mdd if mdd > 1e-9 else 0.0,
        "dd_start": dd.get("dd_start", ""),
        "dd_end": dd.get("dd_end", ""),
        "dd_recover_days": dd.get("dd_recover_days", 0),
        "longest_underwater": dd.get("longest_underwater", 0),
        "days": days,
        "n": len(s),
    }


def _daily(res: dict) -> pd.Series:
    df = res["df"]
    s = pd.Series(df["equity"].astype(float).values,
                  index=pd.to_datetime(df["date"]))
    return s


def _drawdown(series: pd.Series):
    peak = series.cummax()
    dd = series / peak - 1.0
    return dd


def max_drawdown_detail(series: pd.Series) -> dict:
    """最大回撤及其起止、修复、水下时长。"""
    if len(series) < 2:
        return {"max_drawdown": 0.0, "dd_start": "", "dd_end": "",
                "dd_recover_days": 0, "longest_underwater": 0}
    peak = series.cummax()
    dd = series / peak - 1.0
    idx = dd.idxmin()
    mdd = float(dd.min())
    # 回撤起点：idx 之前最后一个创新高的位置
    start_pos = 0
    arr_peak = peak.values
    arr_eq = series.values
    target = arr_peak[series.index.get_loc(idx)]
    for j in range(series.index.get_loc(idx), -1, -1):
        if abs(arr_eq[j] - target) < 1e-9:
            start_pos = j
            break
    # 修复日：回撤后首次回到起点净值
    recover = 0
    pos_idx = series.index.get_loc(idx)
    for j in range(pos_idx, len(series)):
        if arr_eq[j] >= target:
            recover = j - pos_idx
            break
    # 最长水下：连续处于回撤状态的最长跨度
    under = (dd.values < -1e-9).astype(int)
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return {
        "max_drawdown": mdd,
        "dd_start": series.index[start_pos].strftime("%Y-%m-%d"),
        "dd_end": pd.Timestamp(idx).strftime("%Y-%m-%d"),
        "dd_recover_days": int(recover),
        "longest_underwater": int(longest),
    }


def summarize(res: dict, init_cash: float | None = None,
              risk_free: float = RISK_FREE) -> dict:
    """一次性算完全部组合级指标。"""
    eq = _daily(res)
    trades = res["trades"]
    init = init_cash if init_cash is not None else float(eq.iloc[0])
    final = float(eq.iloc[-1])
    n = len(eq)
    days = max((eq.index[-1] - eq.index[0]).days, 1)

    total_ret = final / init - 1.0
    years = days / 365.25
    cagr = (final / init) ** (1.0 / years) - 1.0 if years > 0 and final > 0 else 0.0

    rets = eq.pct_change().dropna()
    vol = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) > 2 else 0.0
    downside = rets[rets < 0]
    dvol = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 2 else 0.0
    sharpe = (cagr - risk_free) / vol if vol > 1e-9 else 0.0
    sortino = (cagr - risk_free) / dvol if dvol > 1e-9 else 0.0

    ddd = max_drawdown_detail(eq)
    mdd = abs(ddd["max_drawdown"])
    calmar = cagr / mdd if mdd > 1e-9 else 0.0

    closed = [t for t in trades if not t.get("is_open")]
    wins = [t for t in closed if t["net_pnl"] > 0]
    losses = [t for t in closed if t["net_pnl"] <= 0]
    win_rate = len(wins) / len(closed) if closed else 0.0
    gross_win = sum(t["net_pnl"] for t in wins)
    gross_loss = abs(sum(t["net_pnl"] for t in losses))
    avg_win = gross_win / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    pl_ratio = avg_win / avg_loss if avg_loss > 1e-9 else (999.0 if avg_win > 0 else 0.0)
    profit_factor = gross_win / gross_loss if gross_loss > 1e-9 else (999.0 if gross_win > 0 else 0.0)
    expectancy = sum(t["net_pnl"] for t in closed) / len(closed) if closed else 0.0

    win_hold = [t["hold_days"] for t in wins]
    loss_hold = [t["hold_days"] for t in losses]
    avg_win_hold = float(np.mean(win_hold)) if win_hold else 0.0
    avg_loss_hold = float(np.mean(loss_hold)) if loss_hold else 0.0

    # 连续盈亏
    streak = 0
    max_win_streak = max_loss_streak = 0
    for t in closed:
        if t["net_pnl"] > 0:
            streak = streak + 1 if streak > 0 else 1
            max_win_streak = max(max_win_streak, streak)
        else:
            streak = streak - 1 if streak < 0 else -1
            max_loss_streak = max(max_loss_streak, -streak)

    # 暴露度与换手
    pos = res["df"]["position"].values if "position" in res["df"].columns else np.zeros(n)
    exposure = float(np.mean(pos)) if len(pos) else 0.0
    turnover_amt = sum(t["entry_amount"] + t["exit_amount"] for t in trades)
    avg_eq = float(eq.mean())
    turnover = turnover_amt / avg_eq if avg_eq > 0 else 0.0

    total_fees = sum(t["fees"] for t in trades)
    net_total = sum(t["net_pnl"] for t in trades)

    out = {
        "init_cash": round(init, 2),
        "final_equity": round(final, 2),
        "total_return": round(total_ret * 100, 2),
        "annual_return": round(cagr * 100, 2),
        "volatility": round(vol * 100, 2),
        "max_drawdown": round(-mdd * 100, 2),
        "dd_start": ddd["dd_start"], "dd_end": ddd["dd_end"],
        "dd_recover_days": ddd["dd_recover_days"],
        "longest_underwater": ddd["longest_underwater"],
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "win_rate": round(win_rate * 100, 2),
        "profit_loss_ratio": round(pl_ratio, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 2),
        "trade_count": len(closed),
        "open_trades": len(trades) - len(closed),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_win_hold": round(avg_win_hold, 1),
        "avg_loss_hold": round(avg_loss_hold, 1),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "exposure": round(exposure * 100, 2),
        "turnover": round(turnover, 2),
        "trades_per_year": round(len(closed) / years, 1) if years > 0 else 0.0,
        "total_fees": round(total_fees, 2),
        "fee_ratio": round(total_fees / net_total * 100, 2) if net_total > 1e-9 else 0.0,
        "trade_days": n,
        "years": round(years, 2),
        "risk_free": risk_free,
    }
    return out


def trade_table(trades: list[dict]) -> list[list[str]]:
    """交易明细表的行（字符串，直接塞进表格控件）。"""
    rows = []
    for i, t in enumerate(trades, 1):
        rows.append([
            str(i),
            t["entry_date"] + ((" " + t["entry_time"]) if t["entry_time"] else ""),
            f"{t['entry_price']:.3f}",
            f"{t['shares']:,}",
            f"{t['entry_amount']:,.0f}",
            (t["exit_date"] + ((" " + t["exit_time"]) if t["exit_time"] else ""))
            if not t.get("is_open") else "持有中",
            f"{t['exit_price']:.3f}",
            f"{t['exit_amount']:,.0f}",
            str(t["hold_days"]),
            str(t["hold_bars"]),
            f"{t['net_pnl']:,.0f}",
            f"{t['ret_pct']:.2f}%",
            f"{t['mfe_pct']:.2f}%",
            f"{t['mae_pct']:.2f}%",
            f"{t['giveback_pct']:.2f}%",
            t.get("exit_reason", ""),
        ])
    return rows


TRADE_COLUMNS = ["序号", "入场", "入场价", "股数", "买入金额", "退出", "退出价",
                 "卖出金额", "自然日", "交易日", "净收益", "收益率",
                 "期间最高", "最大浮亏", "回吐", "退出原因"]


def monthly_returns(res: dict) -> list[dict]:
    """月度收益表。"""
    eq = _daily(res)
    if len(eq) < 2:
        return []
    m = eq.resample("ME").last()
    m0 = eq.iloc[0]
    prev = m0
    out = []
    for ts, v in m.items():
        r = v / prev - 1.0
        out.append({"month": ts.strftime("%Y-%m"), "ret": round(r * 100, 2)})
        prev = v
    return out


def rolling_sharpe(res: dict, window: int = 60) -> list[tuple[str, float]]:
    """滚动夏普，看策略是不是只有某段时间有效。"""
    eq = _daily(res)
    if len(eq) < window + 5:
        return []
    r = eq.pct_change().dropna()
    roll = (r.rolling(window).mean() * TRADING_DAYS - RISK_FREE) / (
        r.rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS))
    roll = roll.dropna()
    return [(ts.strftime("%Y-%m-%d"), round(float(v), 2)) for ts, v in roll.items()]


def monte_carlo(trades: list[dict], init_cash: float = 100000.0,
                paths: int = 1000, seed: int = 7) -> dict:
    """按交易的收益率重排，看结果的分布——衡量「这笔收益有多少是顺序运气」。"""
    rets = [t["ret_pct"] / 100.0 for t in trades if not t.get("is_open")]
    if len(rets) < 3:
        return {"p5": 0, "p50": 0, "p95": 0, "loss_prob": 0, "paths": []}
    rng = np.random.default_rng(seed)
    finals = []
    for _ in range(paths):
        seq = rng.choice(rets, size=len(rets), replace=True)
        eqv = init_cash * np.prod(1 + seq * 0.6)   # 0.6 是仓位折算，避免满仓复利失真
        finals.append(eqv)
    finals = np.array(finals)
    return {
        "p5": round(float(np.percentile(finals, 5)), 0),
        "p50": round(float(np.percentile(finals, 50)), 0),
        "p95": round(float(np.percentile(finals, 95)), 0),
        "loss_prob": round(float((finals < init_cash).mean() * 100), 1),
        "paths": [round(float(x), 0) for x in finals[:200]],
    }


def cost_sensitivity(df, signal, init_cash: float, code: str = "",
                     name: str = "") -> list[dict]:
    """把佣金和滑点加倍，看策略还剩多少。能扛住成本翻倍的才值得上。"""
    from engine import CostModel, run

    out = []
    for mult, label in [(1.0, "基准"), (2.0, "成本×2"), (3.0, "成本×3"), (5.0, "成本×5")]:
        cm = CostModel(commission=0.00025 * mult, slippage=0.0005 * mult,
                       min_commission=5.0 * mult)
        res = run(df, signal, init_cash, cm, code, name)
        m = summarize(res, init_cash)
        out.append({"label": label, "annual": m["annual_return"],
                    "sharpe": m["sharpe"], "maxdd": m["max_drawdown"],
                    "fees": m["total_fees"]})
    return out


def fmt_money(v: float) -> str:
    if abs(v) >= 1e8:
        return f"{v / 1e8:.2f} 亿"
    if abs(v) >= 1e4:
        return f"{v / 1e4:.2f} 万"
    return f"{v:.0f}"


def verdict(m: dict) -> list[tuple[str, str]]:
    """把指标翻译成人话结论，界面直接显示。每条都带数字，正反都说。"""
    out = []
    n = m["trade_count"]
    wr, pl = m["win_rate"], m["profit_loss_ratio"]

    # 1. 盈亏结构
    if n >= 10:
        if wr >= 60 and pl < 1.0:
            cover = abs(m["avg_loss"]) / m["avg_win"] if m["avg_win"] else 0
            out.append(("小赢大亏", f"胜率 {wr}% 但盈亏比只有 {pl}，平均赚 "
                                   f"{m['avg_win']:,.0f} 亏 {m['avg_loss']:,.0f}，"
                                   f"一笔亏损要 {cover:.1f} 笔盈利才填平"))
        elif wr < 45 and pl >= 1.5:
            out.append(("低胜率高赔率", f"胜率只有 {wr}%，但赚的时候是亏的 "
                                       f"{pl} 倍，靠少数大赢撑住整体"))
        else:
            out.append(("盈亏结构均衡", f"胜率 {wr}%、盈亏比 {pl}、盈利因子 "
                                       f"{m['profit_factor']}，两者没有互相拖累"))

    # 2. 收益质量
    if m["annual_return"] > 0:
        if m["calmar"] >= 1:
            out.append(("收益质量过关", f"年化 {m['annual_return']}%，最大回撤 "
                                       f"{abs(m['max_drawdown'])}%，卡玛 "
                                       f"{m['calmar']}，回撤吃得下"))
        else:
            out.append(("收益被回撤吃掉", f"年化 {m['annual_return']}% 看着不错，但最大回撤 "
                                         f"{abs(m['max_drawdown'])}%，卡玛只有 "
                                         f"{m['calmar']}，实际持有体验差"))
    else:
        out.append(("区间内不赚钱", f"年化 {m['annual_return']}%，夏普 "
                                   f"{m['sharpe']}，这套参数在这段行情上无效"))

    # 3. 时间结构
    if m["longest_underwater"] > 120:
        out.append(("水下时间长", f"最长连续 {m['longest_underwater']} 个交易日没创新高，"
                                 f"接近一年，多数人拿不到回本那天"))
    elif m["dd_recover_days"] > 0:
        out.append(("回撤修复速度", f"最大回撤用了 {m['dd_recover_days']} 个交易日修复，"
                                   f"最长水下 {m['longest_underwater']} 日"))

    # 4. 持仓与频率
    if m["exposure"] < 30:
        out.append(("大部分时间空仓", f"持仓占比 {m['exposure']}%，资金利用率低，"
                                     f"年化收益是被摊薄后的数字"))
    if n and m["trades_per_year"] > 40:
        out.append(("交易偏频繁", f"年交易 {m['trades_per_year']:.0f} 笔，"
                                 f"换手 {m['turnover']:.1f} 倍，费用 {m['total_fees']:,.0f} 元"
                                 + (f"，占净收益 {m['fee_ratio']:.0f}%" if m["fee_ratio"] else "")))

    # 5. 行为特征
    if n >= 10 and m["avg_loss_hold"] > m["avg_win_hold"] * 1.5:
        out.append(("处置效应", f"亏的拿 {m['avg_loss_hold']} 天，赢的只拿 "
                               f"{m['avg_win_hold']} 天，卖出信号本身在惩罚盈利"))

    # 6. 样本
    if n < 10:
        out.append(("样本不足", f"只有 {n} 笔，任何结论都不可靠，至少要 30 笔"))
    elif n < 30:
        out.append(("样本偏少", f"{n} 笔，够看趋势但不够定论，参数换一格结论可能就变"))

    return out[:6]
