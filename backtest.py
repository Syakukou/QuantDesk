"""回测引擎：均线交叉 / 动量，日频事件驱动，多头-only（满仓/空仓）。

设计要点：
- 信号在当日收盘算定，次日才产生收益，严格避免未来函数。
- 仓位只有 0 / 1 两态：信号为 1 时满仓持有，为 0 时空仓持币。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def ma_cross_signal(close: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    """快线上穿慢线持有，下穿空仓。返回 1/0 信号序列。"""
    f = sma(close, fast)
    s = sma(close, slow)
    pos = (f > s).astype(int)
    pos = pos.where(f.notna() & s.notna(), 0)
    return pos


def momentum_signal(close: pd.Series, n: int = 20) -> pd.Series:
    """N 日收益率为正持有，否则空仓。"""
    mom = close.pct_change(n)
    pos = (mom > 0).astype(int)
    pos = pos.where(mom.notna(), 0)
    return pos


def _detect_trades(dates: pd.Series, close: pd.Series, signal: pd.Series) -> list[dict]:
    trades = []
    prev = 0
    entry_date = entry_price = None
    for i in range(len(signal)):
        cur = int(signal.iloc[i])
        if cur == 1 and prev == 0:
            entry_date = dates.iloc[i]
            entry_price = float(close.iloc[i])
        elif cur == 0 and prev == 1 and entry_price is not None:
            exit_date = dates.iloc[i]
            exit_price = float(close.iloc[i])
            ret = (exit_price / entry_price) - 1 if entry_price else 0.0
            trades.append({
                "entry_date": pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
                "entry_price": round(entry_price, 3),
                "exit_date": pd.Timestamp(exit_date).strftime("%Y-%m-%d"),
                "exit_price": round(exit_price, 3),
                "return": round(ret * 100, 2),
            })
            entry_price = None
        prev = cur
    # 回测结束时仍持仓，按末日收盘价平仓标记（不计入已实现收益，仅提示）
    if prev == 1 and entry_price is not None:
        trades.append({
            "entry_date": pd.Timestamp(dates.iloc[-1]).strftime("%Y-%m-%d") if False else pd.Timestamp(entry_date).strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 3),
            "exit_date": "持有中",
            "exit_price": round(float(close.iloc[-1]), 3),
            "return": None,
        })
    return trades


def run_backtest(prices: pd.DataFrame, signal: pd.Series,
                 init_cash: float = 100000.0) -> dict:
    """执行回测。

    prices: 含 date, close 的 DataFrame（按日期升序）。
    signal: 与 prices 同索引的 1/0 序列。
    返回 dict: equity(日期+净值列表), trades, metrics, df(含信号与每日收益)。
    """
    df = prices[["date", "close"]].copy()
    df["signal"] = signal.reindex(df.index).fillna(0).astype(int)
    df["ret"] = df["close"].pct_change().fillna(0.0)
    # 今日收益 = 昨日信号 × 今日收益率（无未来函数）
    df["pos_ret"] = df["signal"].shift(1).fillna(0.0) * df["ret"]
    df["equity"] = init_cash * (1.0 + df["pos_ret"]).cumprod()

    equity = [
        [pd.Timestamp(d).strftime("%Y-%m-%d"), round(float(v), 2)]
        for d, v in zip(df["date"], df["equity"])
    ]
    trades = _detect_trades(df["date"], df["close"], df["signal"])

    # 指标
    final_eq = float(df["equity"].iloc[-1])
    total_return = final_eq / init_cash - 1.0
    n_days = max((pd.Timestamp(df["date"].iloc[-1]) - pd.Timestamp(df["date"].iloc[0])).days, 1)
    ann_return = (final_eq / init_cash) ** (365.0 / n_days) - 1.0

    run_max = df["equity"].cummax()
    drawdown = df["equity"] / run_max - 1.0
    max_drawdown = float(drawdown.min())

    closed = [t for t in trades if t["return"] is not None]
    wins = [t for t in closed if t["return"] > 0]
    win_rate = (len(wins) / len(closed)) if closed else 0.0

    daily = df["pos_ret"].values
    std = float(np.std(daily)) if len(daily) > 1 else 0.0
    sharpe = float(np.mean(daily) / std * np.sqrt(TRADING_DAYS)) if std > 1e-12 else 0.0

    metrics = {
        "init_cash": init_cash,
        "final_equity": round(final_eq, 2),
        "total_return": round(total_return * 100, 2),
        "annual_return": round(ann_return * 100, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "win_rate": round(win_rate * 100, 2),
        "trade_count": len(closed),
        "sharpe": round(sharpe, 2),
        "open_trades": len(trades) - len(closed),
    }
    return {"equity": equity, "trades": trades, "metrics": metrics, "df": df}
