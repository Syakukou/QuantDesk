"""事件驱动回测引擎，按 A 股真实规则撮合。

和旧版最大的区别：旧版是「收盘价对收盘价」，相当于假设你能在信号当天收盘
按收盘价成交，且永远能成交。真实情况是——

  · 信号 T 日收盘才成立，最早 T+1 开盘才可能成交
  · 一字涨停买不进，一字跌停卖不出
  · 停牌不能成交，订单要顺延
  · 佣金有 5 元最低，卖出有印花税，还有过户费和滑点
  · 只能按 100 股整手买

这些规则不补，回测出来的收益率是虚的，尤其在小市值和连板票上会虚得离谱。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

TRADING_DAYS = 242.0


# ---------------------------------------------------------------- 成本模型

@dataclass
class CostModel:
    """交易成本。默认值按 2026 年主流券商与税率设置。"""
    commission: float = 0.00025      # 佣金，万 2.5，双向
    min_commission: float = 5.0      # 单笔最低 5 元
    stamp_tax: float = 0.0005        # 印花税，仅卖出，千分之 0.5
    transfer_fee: float = 0.00001    # 过户费，双向，十万分之一
    slippage: float = 0.0005         # 滑点，单边万分之 5

    def buy_cost(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        return max(amount * self.commission, self.min_commission) + amount * self.transfer_fee

    def sell_cost(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        return (max(amount * self.commission, self.min_commission)
                + amount * self.stamp_tax + amount * self.transfer_fee)


def cost_label(cm: CostModel) -> str:
    return (f"佣金 {cm.commission * 10000:.1f}‱（低 {cm.min_commission:.0f} 元）"
            f"｜印花税 {cm.stamp_tax * 1000:.2f}‰｜滑点 {cm.slippage * 10000:.1f}‱")


# ---------------------------------------------------------------- 涨跌停

def limit_pct(code: str = "", name: str = "") -> float:
    """单日涨跌幅限制。ST 5%，创业板/科创板 20%，北交所 30%，其余 10%。"""
    c = str(code).strip()
    nm = str(name).upper().replace(" ", "")
    if "ST" in nm:
        return 0.05
    if c.startswith(("300", "301", "302", "688", "689")):
        return 0.20
    if c[:1] in ("4", "8") or c[:3] == "920":
        return 0.30
    return 0.10


def _pct_series(df: pd.DataFrame) -> pd.Series:
    """日内涨跌幅（小数）。优先用数据里的 pct_change，缺失则用昨收推算。"""
    if "pct_change" in df.columns and df["pct_change"].notna().sum() > len(df) * 0.5:
        return pd.to_numeric(df["pct_change"], errors="coerce") / 100.0
    if "pre_close" in df.columns and df["pre_close"].notna().sum() > len(df) * 0.5:
        return df["close"] / pd.to_numeric(df["pre_close"], errors="coerce") - 1.0
    return df["close"].pct_change()


# ---------------------------------------------------------------- 信号库

def _c(x) -> pd.Series:
    """策略函数既接受 DataFrame 也接受 close 序列。"""
    if isinstance(x, pd.DataFrame):
        return x["close"].astype(float)
    return x.astype(float)


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=int(n), adjust=False).mean()


def ma_cross_signal(close, fast: int = 5, slow: int = 20, **_kw) -> pd.Series:
    close = _c(close)
    f, s = sma(close, fast), sma(close, slow)
    pos = (f > s).astype(int)
    return pos.where(f.notna() & s.notna(), 0)


def momentum_signal(close, n: int = 20, **_kw) -> pd.Series:
    close = _c(close)
    mom = close.pct_change(int(n))
    return (mom > 0).astype(int).where(mom.notna(), 0)


def macd_signal(close, fast: int = 12, slow: int = 26,
                signal: int = 9, **_kw) -> pd.Series:
    close = _c(close)
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    pos = (dif > dea).astype(int)
    return pos.where(dif.notna() & dea.notna(), 0)


def rsi(close, n: int = 14) -> pd.Series:
    close = _c(close)
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def rsi_signal(close, n: int = 14, low: int = 30, high: int = 70,
               **_kw) -> pd.Series:
    """RSI 上穿 low 建仓，跌破 high 离场（持有到超买）。"""
    close = _c(close)
    r = rsi(close, n)
    pos = np.zeros(len(r), dtype=int)
    holding = 0
    for i, v in enumerate(r.values):
        if not holding and v > low:
            holding = 1
        elif holding and v > high:
            holding = 0
        pos[i] = holding
    return pd.Series(pos, index=close.index)


def boll_signal(close, n: int = 20, k: float = 2.0, **_kw) -> pd.Series:
    close = _c(close)
    mid = sma(close, n)
    std = close.rolling(int(n)).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    pos = (close > upper).astype(int)
    pos = pos.where(close < mid, pos)      # 回到中轨之前保持
    # 用状态机实现「突破上轨买入，跌破中轨卖出」
    out = np.zeros(len(close), dtype=int)
    holding = 0
    c = close.values
    u, m = upper.values, mid.values
    for i in range(len(c)):
        if np.isnan(u[i]) or np.isnan(m[i]):
            out[i] = 0
            continue
        if not holding and c[i] > u[i]:
            holding = 1
        elif holding and c[i] < m[i]:
            holding = 0
        out[i] = holding
    return pd.Series(out, index=close.index)


def donchian_signal(df: pd.DataFrame, n: int = 20, exit_n: int = 10,
                    **_kw) -> pd.Series:
    """唐奇安通道：突破 N 日高点买入，跌破 exit_n 日低点卖出。"""
    high = df["high"] if "high" in df.columns else df["close"]
    low = df["low"] if "low" in df.columns else df["close"]
    upper = high.rolling(int(n)).max().shift(1)
    lower = low.rolling(int(exit_n)).min().shift(1)
    c = df["close"].values
    out = np.zeros(len(c), dtype=int)
    holding = 0
    for i in range(len(c)):
        if np.isnan(upper.iloc[i]) or np.isnan(lower.iloc[i]):
            out[i] = 0
            continue
        if not holding and c[i] > upper.iloc[i]:
            holding = 1
        elif holding and c[i] < lower.iloc[i]:
            holding = 0
        out[i] = holding
    return pd.Series(out, index=df.index)


def atr(df, n: int = 14) -> pd.Series:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame({"close": df})
    h = df["high"] if "high" in df.columns else df["close"]
    l = df["low"] if "low" in df.columns else df["close"]
    c = df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def ma_atr_signal(df: pd.DataFrame, fast: int = 10, slow: int = 30,
                  atr_n: int = 14, atr_k: float = 2.5, **_kw) -> pd.Series:
    """双均线 + ATR 移动止损：跌破「持仓期最高价 - k×ATR」离场。"""
    close = df["close"]
    f, s = sma(close, fast), sma(close, slow)
    a = atr(df, atr_n).values
    raw = (f > s).astype(int).values
    c = close.values
    h = (df["high"] if "high" in df.columns else close).values
    out = np.zeros(len(c), dtype=int)
    holding = 0
    peak = 0.0
    for i in range(len(c)):
        if raw[i] == 1 and not holding:
            if not np.isnan(a[i]) and not np.isnan(s.iloc[i]):
                holding = 1
                peak = c[i]
        if holding:
            peak = max(peak, h[i])
            stop = peak - atr_k * a[i] if not np.isnan(a[i]) else -np.inf
            if c[i] < stop or raw[i] == 0:
                holding = 0
        out[i] = holding
    return pd.Series(out, index=df.index)


STRATEGIES = {
    "双均线": {
        "fn": ma_cross_signal,
        "params": {"fast": (5, 60, 5), "slow": (10, 200, 5)},
        "need": ["close"],
        "desc": "快线上穿慢线持有，下穿空仓",
    },
    "动量": {
        "fn": momentum_signal,
        "params": {"n": (5, 120, 5)},
        "need": ["close"],
        "desc": "过去 N 日收益为正则持有",
    },
    "MACD": {
        "fn": macd_signal,
        "params": {"fast": (5, 20, 1), "slow": (20, 60, 2), "signal": (5, 15, 1)},
        "need": ["close"],
        "desc": "DIF 上穿 DEA 持有",
    },
    "RSI": {
        "fn": rsi_signal,
        "params": {"n": (6, 24, 2), "low": (20, 45, 5), "high": (60, 85, 5)},
        "need": ["close"],
        "desc": "RSI 上穿低位建仓，冲上高位离场",
    },
    "布林突破": {
        "fn": boll_signal,
        "params": {"n": (10, 40, 5), "k": (1.5, 3.0, 0.5)},
        "need": ["close"],
        "desc": "突破上轨买入，跌破中轨卖出",
    },
    "唐奇安通道": {
        "fn": donchian_signal,
        "params": {"n": (10, 60, 5), "exit_n": (5, 30, 5)},
        "need": ["high", "low"],
        "desc": "突破 N 日高点买入，跌破 M 日低点卖出",
    },
    "均线+ATR止损": {
        "fn": ma_atr_signal,
        "params": {"fast": (5, 30, 5), "slow": (20, 80, 5), "atr_k": (1.5, 4.0, 0.5)},
        "need": ["high", "low"],
        "desc": "双均线过滤，ATR 移动止损离场",
    },
}


def make_signal(df: pd.DataFrame, strat: str, **params) -> pd.Series:
    spec = STRATEGIES.get(strat)
    if not spec:
        raise ValueError(f"未知策略 {strat}")
    return spec["fn"](df, **params).fillna(0).astype(int)


# ---------------------------------------------------------------- 撮合引擎

@dataclass
class RoundTrip:
    """一笔完整的买入到卖出。字段就是界面上那张交易明细表。"""
    idx_in: int = 0
    idx_out: int = -1
    entry_date: str = ""
    entry_time: str = ""
    entry_price: float = 0.0
    shares: int = 0
    entry_amount: float = 0.0        # 成交金额（不含费）
    exit_date: str = ""
    exit_time: str = ""
    exit_price: float = 0.0
    exit_amount: float = 0.0
    hold_days: int = 0               # 自然日
    hold_bars: int = 0               # 交易日
    gross_pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0
    ret_pct: float = 0.0
    mfe_pct: float = 0.0             # 期间最高浮盈
    mae_pct: float = 0.0             # 期间最大浮亏
    giveback_pct: float = 0.0        # 从最高点回吐
    exit_reason: str = "信号离场"
    signal_name: str = ""
    is_open: bool = False
    delay_days: int = 0              # 信号日到成交日顺延了几天
    equity_before: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _row_time(df: pd.DataFrame, i: int) -> str:
    d = pd.Timestamp(df["date"].iloc[i])
    if d.hour or d.minute:
        return d.strftime("%H:%M")
    return ""


def _fmt_date(df: pd.DataFrame, i: int) -> str:
    return pd.Timestamp(df["date"].iloc[i]).strftime("%Y-%m-%d")


def run(df: pd.DataFrame, signal: pd.Series, init_cash: float = 100000.0,
        cost: CostModel | None = None, code: str = "", name: str = "",
        position_pct: float = 1.0, signal_name: str = "",
        max_delay: int = 5) -> dict:
    """执行回测。

    df: 含 date/open/high/low/close/volume，可含 pct_change。
    signal: 与 df 等长的 1/0 序列，代表「当日收盘后的目标仓位」。
    position_pct: 每次建仓投入可用资金的比例。

    返回 {equity, dates, trades, metrics, df, cost, logs}
    """
    cm = cost or CostModel()
    df = df.reset_index(drop=True).copy()
    n = len(df)
    if n < 3:
        raise ValueError("数据不足，至少需要 3 根 K 线")

    sig = signal.reset_index(drop=True).fillna(0).astype(int)
    lim = limit_pct(code, name)
    pct = _pct_series(df).fillna(0.0).values
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    lo = df["low"].astype(float).values
    c = df["close"].astype(float).values
    vol = df["volume"].astype(float).values if "volume" in df.columns else np.ones(n)

    cash = float(init_cash)
    shares = 0
    entry: dict | None = None
    trades: list[RoundTrip] = []
    eq_curve: list[float] = []
    logs: list[str] = []
    pending_buy_from = -1
    pending_sell_from = -1
    blocked_buy = blocked_sell = 0

    # 涨停/跌停判定阈值：开盘价距涨跌停 0.3% 以内视作封死
    gap = 0.003
    limit_up_px = np.full(n, np.nan)
    limit_dn_px = np.full(n, np.nan)
    for i in range(n):
        pc = c[i - 1] if i > 0 else c[i]
        limit_up_px[i] = pc * (1.0 + lim)
        limit_dn_px[i] = pc * (1.0 - lim)

    for i in range(n):
        # ---- 1. 处理昨天收盘产生的信号，今天开盘成交
        target = int(sig[i - 1]) if i > 0 else 0

        if target == 1 and shares == 0:
            if pending_buy_from < 0:
                pending_buy_from = i - 1
            delay = i - 1 - pending_buy_from
            tradable = vol[i] > 0
            sealed_up = o[i] >= limit_up_px[i] * (1 - gap) and pct[i] >= lim * 0.9
            if tradable and not sealed_up:
                px = o[i] * (1 + cm.slippage)
                budget = cash * position_pct
                lot = 100
                qty = int(budget / (px * (1 + cm.commission + cm.transfer_fee)) // lot) * lot
                if qty > 0:
                    amt = qty * px
                    fee = cm.buy_cost(amt)
                    if amt + fee <= cash:
                        cash -= (amt + fee)
                        shares = qty
                        entry = {
                            "idx_in": i, "date": _fmt_date(df, i), "time": _row_time(df, i),
                            "price": px, "shares": qty, "amount": amt, "fee": fee,
                            "peak": h[i], "trough": lo[i], "delay": delay,
                            "equity_before": cash + shares * c[i],
                        }
                        pending_buy_from = -1
                    else:
                        blocked_buy += 1
            else:
                blocked_buy += 1
                if delay >= max_delay:
                    pending_buy_from = -1
                    logs.append(f"{_fmt_date(df, i)} 连续 {delay + 1} 日无法买入（停牌/涨停），放弃该次信号")

        elif target == 0 and shares > 0:
            if pending_sell_from < 0:
                pending_sell_from = i - 1
            delay = i - 1 - pending_sell_from
            tradable = vol[i] > 0
            sealed_dn = o[i] <= limit_dn_px[i] * (1 + gap) and pct[i] <= -lim * 0.9
            if tradable and not sealed_dn:
                px = o[i] * (1 - cm.slippage)
                amt = shares * px
                fee = cm.sell_cost(amt)
                cash += (amt - fee)
                trades.append(_close_round(entry, i, df, px, amt, fee, shares,
                                           entry, h, lo, eq_before=cash, reason="信号离场",
                                           signal_name=signal_name, delay=delay))
                shares = 0
                entry = None
                pending_sell_from = -1
            else:
                blocked_sell += 1
                if delay >= max_delay:
                    logs.append(f"{_fmt_date(df, i)} 连续 {delay + 1} 日无法卖出（停牌/跌停），顺延")

        # ---- 2. 持仓期更新浮动极值
        if entry is not None:
            entry["peak"] = max(entry["peak"], h[i])
            entry["trough"] = min(entry["trough"], lo[i])
        if target == 1 and shares == 0:
            pending_sell_from = -1
        if target == 0 and shares > 0:
            pending_buy_from = -1

        eq_curve.append(cash + shares * c[i])

    # ---- 3. 回测结束仍持仓：按末日收盘估值，标记未平仓
    if shares > 0 and entry is not None:
        px = c[-1]
        amt = shares * px
        fee = cm.sell_cost(amt)
        rt = _close_round(entry, n - 1, df, px, amt, fee, shares, entry, h, lo,
                          eq_before=cash, reason="持有中", signal_name=signal_name,
                          delay=0, is_open=True)
        trades.append(rt)

    dates = [_fmt_date(df, i) for i in range(n)]
    equity = [[dates[i], round(float(eq_curve[i]), 2)] for i in range(n)]

    df_out = df.copy()
    df_out["signal"] = sig
    df_out["equity"] = eq_curve
    df_out["position"] = 0
    for t in trades:
        end = t.idx_out if t.idx_out >= 0 else n - 1
        df_out.loc[t.idx_in:end, "position"] = 1

    return {
        "equity": equity,
        "dates": dates,
        "trades": [t.as_dict() for t in trades],
        "df": df_out,
        "cost": asdict(cm),
        "logs": logs,
        "blocked": {"buy": blocked_buy, "sell": blocked_sell},
        "limit_pct": lim,
    }


def _close_round(entry: dict, i: int, df: pd.DataFrame, px: float, amt: float,
                 fee: float, shares: int, e: dict, h: np.ndarray, lo: np.ndarray,
                 eq_before: float = 0.0, reason: str = "信号离场",
                 signal_name: str = "", delay: int = 0, is_open: bool = False) -> RoundTrip:
    ep = e["price"]
    d0 = pd.Timestamp(df["date"].iloc[e["idx_in"]])
    d1 = pd.Timestamp(df["date"].iloc[i])
    gross = (px - ep) * shares
    total_fee = e["fee"] + fee
    net = gross - total_fee
    cost_basis = ep * shares + e["fee"]
    peak, trough = e["peak"], e["trough"]
    mfe = (peak / ep - 1) * 100 if ep else 0.0
    mae = (trough / ep - 1) * 100 if ep else 0.0
    ret = (px * shares - fee) / cost_basis - 1 if cost_basis else 0.0
    return RoundTrip(
        idx_in=e["idx_in"], idx_out=i,
        entry_date=d0.strftime("%Y-%m-%d"), entry_time=e["time"],
        entry_price=round(ep, 4), shares=shares,
        entry_amount=round(e["amount"], 2),
        exit_date=d1.strftime("%Y-%m-%d"), exit_time=_row_time(df, i),
        exit_price=round(px, 4), exit_amount=round(amt, 2),
        hold_days=int((d1.normalize() - d0.normalize()).days),
        hold_bars=int(i - e["idx_in"]),
        gross_pnl=round(gross, 2), fees=round(total_fee, 2), net_pnl=round(net, 2),
        ret_pct=round(ret * 100, 2),
        mfe_pct=round(mfe, 2), mae_pct=round(mae, 2),
        giveback_pct=round(mfe - ret * 100, 2),
        exit_reason=reason, signal_name=signal_name, is_open=is_open,
        delay_days=int(e.get("delay", 0)), equity_before=round(eq_before, 2),
    )


# ---------------------------------------------------------------- 参数扫描

def param_grid(df: pd.DataFrame, strat: str, init_cash: float = 100000.0,
               cost: CostModel | None = None, code: str = "", name: str = "",
               max_runs: int = 200) -> list[dict]:
    """参数网格扫描。返回每组的参数与核心指标，供热力图使用。"""
    from metrics import summarize

    spec = STRATEGIES[strat]
    grids = {}
    for k, (lo_, hi_, st) in spec["params"].items():
        grids[k] = list(range(int(lo_), int(hi_) + 1, max(int(st), 1)))

    keys = list(grids.keys())
    combos = [{}]
    for k in keys:
        combos = [dict(c, **{k: v}) for c in combos for v in grids[k]]
    combos = [c for c in combos if _valid(strat, c)][:max_runs]

    out = []
    for p in combos:
        try:
            sg = make_signal(df, strat, **p)
            res = run(df, sg, init_cash, cost, code, name, signal_name=strat)
            m = summarize(res)
            out.append({"params": p, "annual": m["annual_return"],
                        "maxdd": m["max_drawdown"], "sharpe": m["sharpe"],
                        "trades": m["trade_count"], "calmar": m["calmar"]})
        except Exception:
            continue
    return out


def _valid(strat: str, p: dict) -> bool:
    if strat in ("双均线", "均线+ATR止损") and p.get("fast", 0) >= p.get("slow", 1):
        return False
    if strat in ("MACD",) and p.get("fast", 0) >= p.get("slow", 1):
        return False
    if strat == "RSI" and p.get("low", 0) >= p.get("high", 100):
        return False
    return True


def walk_forward(df: pd.DataFrame, strat: str, params: dict, folds: int = 4,
                 init_cash: float = 100000.0, cost: CostModel | None = None,
                 code: str = "", name: str = "") -> list[dict]:
    """滚动样本外：把区间切成 folds 段，每段独立跑，看收益是否集中在某一段。"""
    n = len(df)
    step = n // folds
    out = []
    for f in range(folds):
        a = f * step
        b = n if f == folds - 1 else (f + 1) * step
        seg = df.iloc[a:b].reset_index(drop=True)
        if len(seg) < 30:
            continue
        try:
            sg = make_signal(seg, strat, **params)
            res = run(seg, sg, init_cash, cost, code, name, signal_name=strat)
            from metrics import summarize
            m = summarize(res)
            out.append({
                "fold": f + 1,
                "start": pd.Timestamp(seg["date"].iloc[0]).strftime("%Y-%m-%d"),
                "end": pd.Timestamp(seg["date"].iloc[-1]).strftime("%Y-%m-%d"),
                "return": m["total_return"], "annual": m["annual_return"],
                "maxdd": m["max_drawdown"], "sharpe": m["sharpe"],
                "trades": m["trade_count"], "win_rate": m["win_rate"],
            })
        except Exception:
            continue
    return out
