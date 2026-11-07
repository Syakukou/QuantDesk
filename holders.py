"""机构持仓：社保 / QFII / 保险 / 基金 / 券商 / 信托的季度重仓，以及 A+H 溢价。

口径必须先说清楚，否则看的人一定会误读：

  · 数据来自基金季报和上市公司定期报告，一年只有四个时点，是季末快照，
    没有盘中调仓这回事。看到「社保增持」只能理解成「上个季度末比上上个季度末多」，
    不代表今天还拿着。
  · 持股市值按报告期末的收盘价算，不是按今天的价。跟现价对比要先看涨跌幅。
  · 北向（陆股通）个股持股明细从 2024-08-18 起交易所停止披露，
    这条线断了以后，追踪外资的官方口径只剩 QFII 季报。界面上会标明，
    免得有人以为「北向没数据」是软件坏了。
  · 十大流通股东里的社保组合名，是按「全国社保基金一一三组合」这种口径披露的，
    同一只票可能被三四只不同组合同时拿着，所以「组合数」和「家数」不是一回事。
"""
from __future__ import annotations

import os
import time
from datetime import date

import pandas as pd

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".quantdesk", "cache", "holders")

KINDS = ("社保", "QFII", "保险", "基金", "券商", "信托")

_KIND_MAP = {
    "社保": "社保持仓",
    "QFII": "QFII持仓",
    "保险": "保险持仓",
    "基金": "基金持仓",
    "券商": "券商持仓",
    "信托": "信托持仓",
}

_KIND_NOTE = {
    "社保": "全国社保基金理事会及各家社保组合，季报只给合并数",
    "QFII": "境外合格机构投资者，北向停发后唯一能追踪外资的官方口径",
    "保险": "险资自营与保险资管产品，长钱，看的是仓位变动方向",
    "基金": "公募基金全部持仓，样本最多，也最容易跟着抱团",
    "券商": "券商自营盘，通常跟着做市和衍生品对冲，参考价值弱于前四类",
    "信托": "信托计划持股，样本少，多是历史遗留通道",
}

# 季度末 → (可用披露月, 日)。年报拖到次年 4 月底，所以跨年。
_DISCLOSE = {"0331": (4, 30), "0630": (8, 31), "0930": (10, 31), "1231": (4, 30)}
_QNAME = {"03": "Q1", "06": "Q2", "09": "Q3", "12": "Q4"}

# 原始「持股变化」字段是几套词汇混用，统一成四个动作
_CHG_MAP = {
    "新进": "新进", "新增": "新进", "新出现在": "新进",
    "增仓": "增持", "加仓": "增持", "增持": "增持", "增加": "增持",
    "减仓": "减持", "减持": "减持", "减少": "减持",
    "不变": "不变", "持平": "不变",
}


# ------------------------------------------------------------------ 基础工具
def _cached(name: str, fn, force: bool = False, ttl: int = 12 * 3600):
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


def recent_periods(n: int = 5, today: date | None = None) -> list[str]:
    """最近 n 个已经披露完的报告期，倒序（最新在前）。

    不能简单按「今天往前推 3 个月」算，因为年报要等次年 4 月底。
    """
    t = today or date.today()
    out: list[str] = []
    for y in range(t.year, t.year - 5, -1):
        for md in ("1231", "0930", "0630", "0331"):
            m, d = _DISCLOSE[md]
            avail_year = y + 1 if md == "1231" else y
            if date(avail_year, m, d) <= t:
                p = f"{y}{md}"
                if p not in out:
                    out.append(p)
    out.sort(reverse=True)
    return out[:n]


def quarter_name(period: str) -> str:
    return f"{period[:4]}{_QNAME.get(period[4:6], '')}"


def kind_note(kind: str) -> str:
    return _KIND_NOTE.get(kind, "")


# ------------------------------------------------------------------ 拉取
def _fetch(kind: str, period: str) -> pd.DataFrame:
    import akshare as ak
    sym = _KIND_MAP.get(kind)
    if sym is None:
        raise ValueError(f"未知机构类型：{kind}")
    return ak.stock_report_fund_hold(symbol=sym, date=period)


def _normalize(raw: pd.DataFrame, kind: str, period: str) -> pd.DataFrame:
    if raw is None or not len(raw):
        return pd.DataFrame()
    d = pd.DataFrame({
        "code": raw["股票代码"].astype(str).str.strip().str.zfill(6),
        "name": raw["股票简称"].astype(str).str.strip(),
        "n_inst": pd.to_numeric(raw.get("持有基金家数"), errors="coerce"),
        "shares": pd.to_numeric(raw.get("持股总数"), errors="coerce"),
        "value": pd.to_numeric(raw.get("持股市值"), errors="coerce"),
        "chg_raw": raw.get("持股变化", pd.Series(dtype=str)).astype(str),
        "chg_shares": pd.to_numeric(raw.get("持股变动数值"), errors="coerce"),
        "chg_pct": pd.to_numeric(raw.get("持股变动比例"), errors="coerce"),
    })
    d["change"] = d["chg_raw"].map(lambda s: _CHG_MAP.get(s.strip(), s.strip()))
    d["kind"] = kind
    d["period"] = period
    return d.drop(columns=["chg_raw"])


def load(kind: str, period: str, force: bool = False) -> pd.DataFrame:
    """某一类机构某一个季度的持仓，已标准化。"""
    raw = _cached(f"{kind}_{period}.pkl", lambda: _fetch(kind, period),
                  force=force, ttl=7 * 86400)
    return _normalize(raw, kind, period)


def _streak(vals: list[float]) -> int:
    """从最新一期往前数，持股数连续增加的季数。

    中间断一期（停牌没披露、或退出前十大）就停，不跨缺口硬算，
    否则「连续 4 季增持」里可能隔着两个空季度。
    """
    n = 0
    for i in range(len(vals) - 1, 0, -1):
        cur, prev = vals[i], vals[i - 1]
        if pd.isna(cur) or pd.isna(prev) or prev <= 0:
            break
        if cur > prev:
            n += 1
        else:
            break
    return n


def panel(kind: str, periods: list[str] | None = None,
          force: bool = False) -> pd.DataFrame:
    """最新一期持仓，外加「连续增持季数」和「现有几季数据」。

    这是主表用的形态：一行一只票，列是当期数字，streak 是历史结论。
    """
    ps = periods or recent_periods(5)
    frames = [load(kind, p, force=force) for p in ps]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()

    latest = frames[0].copy()
    wide = pd.concat(
        [f[["code", "shares"]].assign(p=f["period"].iloc[0]) for f in frames],
        ignore_index=True)
    piv = wide.pivot_table(index="code", columns="p", values="shares",
                           aggfunc="sum")
    piv = piv.reindex(columns=sorted(piv.columns))     # 时间升序
    st = piv.apply(lambda r: _streak(r.tolist()), axis=1)
    n_period = piv.notna().sum(axis=1)

    latest["streak"] = latest["code"].map(st).fillna(0).astype(int)
    latest["n_period"] = latest["code"].map(n_period).fillna(0).astype(int)
    latest["periods"] = len(ps)
    return latest


def history(code: str, kind: str,
            periods: list[str] | None = None) -> pd.DataFrame:
    """单只票在该类机构下的逐季持仓，按时间倒序。"""
    ps = periods or recent_periods(5)
    rows = []
    for p in ps:
        f = load(kind, p)
        if not len(f):
            continue
        hit = f[f["code"] == code]
        if len(hit):
            r = hit.iloc[0]
            rows.append({
                "period": p, "季度": quarter_name(p),
                "n_inst": int(r["n_inst"]) if pd.notna(r["n_inst"]) else 0,
                "shares": float(r["shares"]) if pd.notna(r["shares"]) else 0.0,
                "value": float(r["value"]) if pd.notna(r["value"]) else 0.0,
                "change": r["change"],
                "chg_shares": float(r["chg_shares"]) if pd.notna(r["chg_shares"]) else 0.0,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 交叉重仓
def cross_hold(period: str | None = None,
               force: bool = False) -> pd.DataFrame:
    """六类机构在同一季度的重叠持仓。

    一只票被越多类型的机构同时拿着，说明它的逻辑越不依赖单一买方的偏好。
    这一点比「社保重仓」更有信息量：社保买是长钱，险资和 QFII 也买，
    说明看的不是同一个理由。
    """
    p = period or recent_periods(1)[0]
    frames = []
    for k in KINDS:
        try:
            f = load(k, p, force=force)
        except Exception:  # noqa: BLE001
            continue
        if len(f):
            frames.append(f)
    if not frames:
        return pd.DataFrame()

    d = pd.concat(frames, ignore_index=True)
    g = d.groupby(["code", "name"], as_index=False).agg(
        n_kind=("kind", "nunique"),
        kinds=("kind", lambda s: "、".join(sorted(set(s)))),
        value=("value", "sum"),
        shares=("shares", "sum"),
        n_inst=("n_inst", "sum"),
    )
    return g.sort_values(["n_kind", "value"], ascending=[False, False])


# ------------------------------------------------------------------ 社保组合
def _fetch_gdfx(period: str) -> pd.DataFrame:
    import akshare as ak
    return ak.stock_gdfx_free_holding_detail_em(date=period)


def _split_combo(name: str) -> tuple[str, str]:
    s = str(name).strip()
    if "-" in s:
        left, right = s.split("-", 1)
        return left.strip(), right.strip()
    return "", s


def social_combos(period: str | None = None,
                  force: bool = False) -> pd.DataFrame:
    """十大流通股东里的社保组合明细。

    全市场一次 5.6 万行、25 秒左右，必须缓存。这是唯一能看到
    「是哪个组合在买」的口径，机构持仓那张表只有合并后的总数。
    """
    p = period or recent_periods(1)[0]
    raw = _cached(f"gdfx_{p}.pkl", lambda: _fetch_gdfx(p),
                  force=force, ttl=7 * 86400)
    if raw is None or not len(raw):
        return pd.DataFrame()

    s = raw[raw["股东名称"].astype(str).str.contains("社保", na=False)].copy()
    if not len(s):
        return pd.DataFrame()

    parts = s["股东名称"].map(_split_combo)
    out = pd.DataFrame({
        "combo": [x[1] for x in parts],
        "manager": [x[0] for x in parts],
        "code": s["股票代码"].astype(str).str.zfill(6),
        "name": s["股票简称"].astype(str),
        "shares": pd.to_numeric(s.get("期末持股-数量"), errors="coerce"),
        "chg_shares": pd.to_numeric(s.get("期末持股-数量变化"), errors="coerce"),
        "chg_pct": pd.to_numeric(s.get("期末持股-数量变化比例"), errors="coerce"),
        "value": pd.to_numeric(s.get("期末持股-流通市值"), errors="coerce"),
        "chg_raw": s.get("期末持股-持股变动", pd.Series(dtype=str)).astype(str),
        "announce": s.get("公告日", pd.Series(dtype=str)).astype(str),
        "period": p,
    })
    out["change"] = out["chg_raw"].map(lambda x: _CHG_MAP.get(x.strip(), x.strip()))
    return out.drop(columns=["chg_raw"])


def combos_of(code: str, period: str | None = None) -> pd.DataFrame:
    p = period or recent_periods(1)[0]
    d = social_combos(p)
    if not len(d):
        return pd.DataFrame()
    return d[d["code"] == code].sort_values("value", ascending=False)


def combo_rank(period: str | None = None) -> pd.DataFrame:
    """按社保组合汇总：哪个组合持了多少只、多少市值、本季净买多少。"""
    d = social_combos(period)
    if not len(d):
        return pd.DataFrame()
    g = d.groupby("combo", as_index=False).agg(
        n_stock=("code", "nunique"),
        value=("value", "sum"),
        shares=("shares", "sum"),
        chg_shares=("chg_shares", "sum"),
        add=("change", lambda s: int((s == "增持").sum() + (s == "新进").sum())),
        cut=("change", lambda s: int((s == "减持").sum())),
    )
    return g.sort_values("value", ascending=False)


# ------------------------------------------------------------------ A+H 溢价
def _fetch_ah() -> pd.DataFrame:
    import akshare as ak
    return ak.stock_zh_ah_spot_em()


def ah_premium(force: bool = False) -> pd.DataFrame:
    """同时在内地和香港上市的公司，A 股相对 H 股的溢价。

    「香港投资」这条线，北向持股已经停发，能看的只剩两地价差：
    同一家公司，H 股便宜四成，说明港股那边给的估值更低，
    要么是流动性折价，要么是 A 股在付情绪溢价。
    """
    raw = _cached("ah_spot.pkl", _fetch_ah, force=force, ttl=4 * 3600)
    if raw is None or not len(raw):
        return pd.DataFrame()
    d = pd.DataFrame({
        "a_code": raw["A股代码"].astype(str).str.zfill(6),
        "name": raw["名称"].astype(str),
        "h_code": raw["H股代码"].astype(str).str.zfill(5),
        "h_px": pd.to_numeric(raw.get("最新价-HKD"), errors="coerce"),
        "a_px": pd.to_numeric(raw.get("最新价-RMB"), errors="coerce"),
        "ratio": pd.to_numeric(raw.get("比价"), errors="coerce"),
        "premium": pd.to_numeric(raw.get("溢价"), errors="coerce"),
        "h_chg": pd.to_numeric(raw.get("H股-涨跌幅"), errors="coerce"),
        "a_chg": pd.to_numeric(raw.get("A股-涨跌幅"), errors="coerce"),
    })
    d = d.dropna(subset=["premium"])
    return d.sort_values("premium", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ 汇总口径
def summary(panel_df: pd.DataFrame) -> dict:
    """主表上方那排数字。只描述样本，不做主观判断。"""
    if panel_df is None or not len(panel_df):
        return {"n": 0}
    chg = panel_df["change"].fillna("")
    return {
        "n": len(panel_df),
        "value": float(panel_df["value"].fillna(0).sum()),
        "shares": float(panel_df["shares"].fillna(0).sum()),
        "new": int((chg == "新进").sum()),
        "add": int((chg == "增持").sum()),
        "cut": int((chg == "减持").sum()),
        "flat": int((chg == "不变").sum()),
        "streak2": int((panel_df["streak"] >= 2).sum()),
        "streak3": int((panel_df["streak"] >= 3).sum()),
        "net_shares": float(panel_df["chg_shares"].fillna(0).sum()),
    }
