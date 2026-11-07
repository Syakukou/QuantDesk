"""交易行为审计：把你自己的成交记录当成一只「基金」来解剖。

券商客户端只给你成交明细和当日盈亏，它不会告诉你：
  · 你是不是赚 3% 就跑、亏 15% 才割
  · 你的钱是不是都亏在某一个时段、某一只票上
  · 如果严格按你自己定的规则卖，能多赚多少

这一层就是干这三件事。所有结论都必须带数字证据，不允许只给一句「您存在
处置效应」——那是算命，不是分析。

四个偏误的判定标准借自行为金融实证文献（Odean 的 PGR/PLR 思路），在
没有每日持仓快照的情况下用「持有天数 × 收益分布」做代理，这一点在界面
上会写明，不假装严谨。
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

LEVELS = ("低", "中", "高")


def _med(xs) -> float:
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else 0.0


def _pct(a: float, b: float) -> float:
    return (a / b - 1) * 100 if b else 0.0


# ------------------------------------------------------------------ 四类偏误

def diagnose(records: list[dict], closed: list[dict],
             summary: dict | None = None, bars_lookup=None) -> dict:
    """返回 {biases:[...], groups:{...}, calendar:[...], score, note}"""
    biases = []
    if not closed:
        return {"biases": [], "groups": {}, "calendar": [], "score": 0,
                "note": "没有已平仓记录，无法诊断"}

    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]

    # ---- 1. 处置效应
    wd = _med([t["days"] for t in wins])
    ld = _med([t["days"] for t in losses])
    wr = _med([t["ret"] for t in wins])
    lr = _med([t["ret"] for t in losses])
    ratio = (ld / wd) if wd > 0 else 0.0
    if ratio >= 2.0:
        sev = "高"
    elif ratio >= 1.4:
        sev = "中"
    else:
        sev = "低"
    biases.append({
        "key": "disposition",
        "name": "处置效应",
        "severity": sev,
        "headline": f"亏损单拿 {ld:.0f} 天，盈利单只拿 {wd:.0f} 天",
        "evidence": (f"盈利单中位数 {wd:.0f} 天／{wr:+.1f}%，"
                     f"亏损单中位数 {ld:.0f} 天／{lr:+.1f}%，"
                     f"持有时间比 {ratio:.1f} 倍"),
        "advice": "把「涨了就跑、跌了就扛」倒过来，是第一件能立刻改的事。"
                  "具体做法：买入时就写下止损价，到价无条件走。",
        "value": round(ratio, 2),
    })

    # ---- 2. 过度交易
    months = set()
    for r in records:
        d = pd.Timestamp(r["date"])
        months.add((d.year, d.month))
    n_month = max(len(months), 1)
    n_trade = len([r for r in records if r["side"] in ("B", "S")])
    per_month = n_trade / n_month
    buy_amt = sum(t["buy_amt"] for t in closed)
    fee = sum(t["fee"] for t in closed)
    fee_ratio = fee / buy_amt * 100 if buy_amt > 0 else 0.0
    loss_abs = abs(sum(t["pnl"] for t in losses))
    fee_vs_loss = fee / loss_abs * 100 if loss_abs > 0 else 0.0
    if fee_vs_loss >= 30 or per_month >= 25:
        sev = "高"
    elif fee_vs_loss >= 15 or per_month >= 12:
        sev = "中"
    else:
        sev = "低"
    biases.append({
        "key": "overtrading",
        "name": "过度交易",
        "severity": sev,
        "headline": f"月均 {per_month:.0f} 笔，手续费 {fee:,.0f} 元",
        "evidence": (f"{n_month} 个月成交 {n_trade} 笔，手续费合计 {fee:,.0f} 元，"
                     f"占买入额 {fee_ratio:.2f}%；相当于亏损总额的 {fee_vs_loss:.0f}%"),
        "advice": "费用本身不是大问题，问题是高频往往对应「没想法也要做一笔」。"
                  "试一个月只做清单内的票，笔数砍一半看收益变不变。",
        "value": round(per_month, 1),
    })

    # ---- 3. 追涨杀跌（需要行情）
    chase = _chase_analysis(records, closed, bars_lookup)
    if chase:
        biases.append(chase)

    # ---- 4. 锚定 / 摊薄补仓
    anchor = _anchor_analysis(records, closed)
    if anchor:
        biases.append(anchor)

    groups = analyze_groups(records, closed)
    calendar = pnl_calendar(closed)
    score = _score(biases, closed)
    return {
        "biases": biases,
        "groups": groups,
        "calendar": calendar,
        "score": score,
        "note": ("持有天数比是无每日持仓快照时的代理指标；"
                 "PGR/PLR 完整算法需要逐日持仓明细，本工具不做假装。"),
    }


def _chase_analysis(records, closed, bars_lookup, max_codes: int = 40) -> dict | None:
    """追涨杀跌：买入时已处于近 20 日高位、且买入后 5 日为负的比例。

    只抽样前 max_codes 只标的，避免为了一个指标拉几百只票的行情。
    """
    if bars_lookup is None:
        return None
    buys = [r for r in records if r["side"] == "B"]
    if len(buys) < 5:
        return None
    cache: dict[str, pd.DataFrame] = {}
    hi_cnt = tot = 0
    after_rets = []
    for r in buys[:400]:
        code = r["code"]
        if code not in cache:
            if len(cache) >= max_codes:
                continue
            try:
                cache[code] = bars_lookup(code)
            except Exception:
                cache[code] = pd.DataFrame()
        bars = cache[code]
        if bars is None or bars.empty:
            continue
        d = pd.Timestamp(r["date"]).normalize()
        idx = bars.index[bars["date"].dt.normalize() == d]
        if len(idx) == 0:
            continue
        i = bars.index.get_loc(idx[0])
        if i < 20:
            continue
        c = bars["close"].astype(float)
        pos20 = (c.iloc[max(0, i - 20):i + 1] <= c.iloc[i]).mean()
        tot += 1
        if pos20 >= 0.8:
            hi_cnt += 1
        if i + 5 < len(c):
            after_rets.append(_pct(c.iloc[i + 5], c.iloc[i]))
    if tot < 5:
        return None
    hi_ratio = hi_cnt / tot * 100
    avg_after = float(np.mean(after_rets)) if after_rets else 0.0
    if hi_ratio >= 65 and avg_after < 0:
        sev = "高"
    elif hi_ratio >= 50 and avg_after < 0.5:
        sev = "中"
    else:
        sev = "低"
    return {
        "key": "chasing",
        "name": "追涨",
        "severity": sev,
        "headline": f"{hi_ratio:.0f}% 的买入发生在近 20 日价格高位",
        "evidence": (f"抽样 {tot} 笔买入，其中 {hi_cnt} 笔买在近 20 日价格区间的 "
                     f"80% 分位以上；这批买入后 5 日平均收益 {avg_after:+.2f}%"),
        "advice": "买在高位不是错，错在高位买完没有对应的高波动准备。"
                  "把仓位按「距 20 日均线的偏离度」反向缩放。",
        "value": round(hi_ratio, 1),
    }


def _anchor_analysis(records, closed) -> dict | None:
    """锚定：同一轮里反复补仓摊薄，且价格越补越低。"""
    rounds = []
    cur = {}
    for r in records:
        code = r["code"]
        if r["side"] == "B":
            st = cur.setdefault(code, {"prices": [], "qty": 0.0})
            if r["price"]:
                st["prices"].append(r["price"])
            st["qty"] += r["qty"]
        elif r["side"] == "S" and code in cur:
            st = cur.pop(code)
            if len(st["prices"]) >= 2:
                rounds.append((code, st["prices"]))
    if len(rounds) < 3:
        return None
    avg_down = 0
    for _, prices in rounds:
        if prices[-1] < prices[0] * 0.99:
            avg_down += 1
    ratio = avg_down / len(rounds) * 100
    mult = [len(p) for _, p in rounds]
    avg_n = float(np.mean(mult))
    # 这些摊薄轮次的盈亏
    codes_down = {c for c, p in rounds if p[-1] < p[0] * 0.99}
    pnl_down = sum(t["pnl"] for t in closed if t["code"] in codes_down)
    if pnl_down > 0:
        sev = "低"
        headline = f"{ratio:.0f}% 的轮次越跌越买，但结果是赚的"
        advice = ("数据不支持「补仓有害」这个结论，这批摊薄轮次合计是盈利的。"
                  "要盯的是下一次：摊薄赚钱容易让人把「运气」当成「方法」。")
    else:
        if ratio >= 50:
            sev = "高"
        elif ratio >= 35:
            sev = "中"
        else:
            sev = "低"
        headline = f"{ratio:.0f}% 的持仓轮次出现越跌越买"
        advice = ("补仓需要理由（基本面变化、估值到位），不是「比成本便宜」。"
                  "把补仓条件写成可验证的条款，而不是价格直觉。")
    return {
        "key": "anchoring",
        "name": "锚定补仓",
        "severity": sev,
        "headline": headline,
        "evidence": (f"{len(rounds)} 个建仓轮次平均买 {avg_n:.1f} 次，"
                     f"其中 {avg_down} 轮最后一次买入价低于首次 1% 以上，"
                     f"这批轮次合计 {pnl_down:+,.0f} 元"),
        "advice": advice,
        "value": round(ratio, 1),
    }


# ------------------------------------------------------------------ 分组找漏

def _bucket_hold(d: int) -> str:
    if d <= 1:
        return "1 天内"
    if d <= 3:
        return "2–3 天"
    if d <= 7:
        return "4–7 天"
    if d <= 30:
        return "8–30 天"
    if d <= 90:
        return "31–90 天"
    return "90 天以上"


def _bucket_hour(ts) -> str:
    if ts is None or pd.isna(ts):
        return "未知"
    h = pd.Timestamp(ts).hour
    if h < 10:
        return "09:30–10:00"
    if h < 11:
        return "10:00–11:00"
    if h < 12:
        return "11:00–11:30"
    if h < 14:
        return "13:00–14:00"
    if h < 15:
        return "14:00–15:00"
    return "15:00 后"


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    pnl = sum(t["pnl"] for t in rows)
    wins = [t for t in rows if t["pnl"] > 0]
    return {
        "n": n, "pnl": round(pnl, 0),
        "win_rate": round(len(wins) / n * 100, 1) if n else 0.0,
        "avg": round(pnl / n, 0) if n else 0.0,
        "avg_days": round(float(np.mean([t["days"] for t in rows])), 1) if n else 0.0,
        # 原始成交留着：界面上点了某一组，右边的日历要按这一组的平仓日重算，
        # 只有汇总数字没法还原到日期，所以这里必须带上明细。
        "trades": rows,
    }


def analyze_groups(records: list[dict], closed: list[dict],
                   min_n: int = 3) -> dict:
    """按标的 / 星期 / 持有期 / 时段分组，找出亏得最集中的那一格。"""
    if not closed:
        return {}

    def group_by(keyfn, title):
        buckets = defaultdict(list)
        for t in closed:
            buckets[keyfn(t)].append(t)
        rows = []
        for k, v in buckets.items():
            a = _agg(v)
            a["key"] = k
            rows.append(a)
        rows.sort(key=lambda x: x["pnl"])
        valid = [r for r in rows if r["n"] >= min_n]
        worst = valid[0] if valid else (rows[0] if rows else None)
        return {"title": title, "rows": rows, "worst": worst}

    def wk(t):
        return ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][
            pd.Timestamp(t["entry_date"]).weekday()]

    out = {
        "symbol": group_by(lambda t: f"{t['code']} {t['name']}", "按标的"),
        "weekday": group_by(wk, "按入场星期"),
        "hold": group_by(lambda t: _bucket_hold(int(t["days"])), "按持有天数"),
    }

    # 时段要从成交记录里取（配对结果没有时间）
    # 券商导出的「成交时间」常是清算批次时间，先验证有效性
    good_hours = 0
    hours = []
    for r in records:
        ts = r.get("ts")
        if ts is None or pd.isna(ts):
            continue
        h = pd.Timestamp(ts).hour + pd.Timestamp(ts).minute / 60.0
        hours.append(h)
        if 9.4 <= h <= 15.05:
            good_hours += 1
    if hours and (len(hours) - good_hours) / len(hours) <= 0.1:
        first_buy = {}
        for r in records:
            if r["side"] == "B":
                first_buy.setdefault((r["code"], pd.Timestamp(r["date"]).normalize()),
                                     r.get("ts"))
        tb = defaultdict(list)
        for t in closed:
            k = (t["code"], pd.Timestamp(t["entry_date"]).normalize())
            tb[_bucket_hour(first_buy.get(k))].append(t)
        rows = []
        for k, v in tb.items():
            a = _agg(v)
            a["key"] = k
            rows.append(a)
        rows.sort(key=lambda x: x["pnl"])
        valid = [r for r in rows if r["n"] >= min_n]
        out["hour"] = {"title": "按入场时段", "rows": rows,
                       "worst": valid[0] if valid else None}
    else:
        out["hour_disabled"] = (
            f"交割单里的成交时间只有 {good_hours}/{len(hours)} 落在 9:30–15:00，"
            "其余是券商清算批次时间，不足以支撑时段分析")
    return out


def pnl_calendar(closed: list[dict]) -> list[dict]:
    """按平仓日汇总当日已实现盈亏，用来画日历热力。"""
    by_day = defaultdict(float)
    cnt = defaultdict(int)
    for t in closed:
        d = pd.Timestamp(t["exit_date"]).strftime("%Y-%m-%d")
        by_day[d] += t["pnl"]
        cnt[d] += 1
    return [{"date": d, "pnl": round(v, 0), "n": cnt[d]}
            for d, v in sorted(by_day.items())]


def _score(biases: list[dict], closed: list[dict]) -> int:
    """纪律分（0-100）。只做相对参考，不是评级。"""
    score = 100
    w = {"高": 18, "中": 10, "低": 4}
    for b in biases:
        score -= w.get(b["severity"], 0)
    if len(closed) < 30:
        score = min(score, 60)
    return max(0, min(100, score))


# ------------------------------------------------------------------ 影子账户

def extract_rules(closed: list[dict]) -> dict:
    """从你自己的成交里反推「你实际的卖出规则」。"""
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    tp = _med([t["ret"] for t in wins]) if wins else 10.0
    sl = _med([t["ret"] for t in losses]) if losses else -8.0
    days = _med([t["days"] for t in closed]) or 10
    return {
        "take_profit": round(max(tp, 1.0), 2),
        "stop_loss": round(min(sl, -1.0), 2),
        "max_hold": int(max(days * 1.5, 5)),
        "sample": len(closed),
        "win_n": len(wins),
        "loss_n": len(losses),
    }


def shadow_replay(closed: list[dict], bars_lookup, rules: dict | None = None,
                  max_items: int = 300) -> dict:
    """保留你的选股和买入时点，只把卖出换成规则化执行，看差额。

    这是「卖出纪律的成本」的量化：同一只票、同一天买，按止盈/止损/超期
    三条线机械卖出，和你的实际操作比，差多少。
    """
    if not closed:
        return {"ok": False, "msg": "没有已平仓记录"}
    rules = rules or extract_rules(closed)
    tp, sl, mh = rules["take_profit"] / 100.0, abs(rules["stop_loss"]) / 100.0, rules["max_hold"]

    cache: dict[str, pd.DataFrame] = {}
    rows = []
    actual_total = shadow_total = hold_total = 0.0
    used = 0
    for t in closed[:max_items]:
        code = t["code"]
        if code not in cache:
            try:
                cache[code] = bars_lookup(code)
            except Exception:
                cache[code] = pd.DataFrame()
        bars = cache[code]
        if bars is None or bars.empty:
            continue
        d0 = pd.Timestamp(t["entry_date"]).normalize()
        idx = bars.index[bars["date"].dt.normalize() == d0]
        if len(idx) == 0:
            continue
        i0 = bars.index.get_loc(idx[0])
        entry_px = float(bars["close"].iloc[i0])
        if entry_px <= 0:
            continue
        exit_px = None
        reason = "超期"
        seg = bars.iloc[i0:i0 + mh + 1]
        for j in range(len(seg)):
            hi = float(seg["high"].iloc[j]) if "high" in seg else float(seg["close"].iloc[j])
            lo = float(seg["low"].iloc[j]) if "low" in seg else float(seg["close"].iloc[j])
            if j > 0 and lo <= entry_px * (1 - sl):
                exit_px = entry_px * (1 - sl)
                reason = "止损"
                break
            if j > 0 and hi >= entry_px * (1 + tp):
                exit_px = entry_px * (1 + tp)
                reason = "止盈"
                break
        if exit_px is None:
            exit_px = float(seg["close"].iloc[-1])
            reason = "超期"
        shadow_ret = exit_px / entry_px - 1
        shadow_pnl = t["buy_amt"] * shadow_ret
        # 第二个基准：买入后不择时，机械持有到 max_hold 天
        hold_px = float(bars["close"].iloc[min(i0 + mh, len(bars) - 1)])
        hold_pnl = t["buy_amt"] * (hold_px / entry_px - 1)
        rows.append({
            "code": code, "name": t["name"], "entry_date": t["entry_date"],
            "actual": round(t["pnl"], 0), "shadow": round(shadow_pnl, 0),
            "hold": round(hold_pnl, 0),
            "diff": round(shadow_pnl - t["pnl"], 0),
            "hold_diff": round(hold_pnl - t["pnl"], 0),
            "actual_ret": t["ret"], "shadow_ret": round(shadow_ret * 100, 2),
            "hold_ret": round((hold_px / entry_px - 1) * 100, 2),
            "actual_days": t["days"], "reason": reason,
        })
        actual_total += t["pnl"]
        shadow_total += shadow_pnl
        hold_total += hold_pnl
        used += 1

    if used == 0:
        return {"ok": False, "msg": "没有可用的行情数据，无法回放（需要联网）"}
    rows.sort(key=lambda x: x["diff"])
    return {
        "ok": True,
        "rules": rules,
        "rows": rows,
        "used": used,
        "actual": round(actual_total, 0),
        "shadow": round(shadow_total, 0),
        "hold": round(hold_total, 0),
        "diff": round(shadow_total - actual_total, 0),
        "hold_diff": round(hold_total - actual_total, 0),
        "msg": (f"同一批买入、共 {used} 轮：实际 {actual_total:,.0f} 元；"
                f"机械止盈 {rules['take_profit']}%／止损 {abs(rules['stop_loss'])}%／"
                f"持 {mh} 天 → {shadow_total:,.0f} 元；"
                f"完全不择时、持 {mh} 天 → {hold_total:,.0f} 元"),
    }


# ------------------------------------------------------------------ 综合判读

def overall(closed: list[dict], summary: dict) -> list[dict]:
    """把胜率、盈亏比、期望值放在一起读，给出结构性结论。"""
    if not closed:
        return []
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    wr = len(wins) / len(closed)
    avg_w = sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
    avg_l = sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0
    exp = sum(t["pnl"] for t in closed) / len(closed)
    out = []

    if wr >= 0.65 and abs(avg_l) > avg_w * 1.2:
        cover = abs(avg_l) / avg_w if avg_w else 0
        out.append({
            "title": "典型的小赢大亏结构",
            "text": (f"{wr * 100:.0f}% 的胜率是靠不停止盈换来的：平均赚 {avg_w:,.0f} 元、"
                     f"亏 {avg_l:,.0f} 元，一笔亏损要 {cover:.1f} 笔盈利才填得平。"
                     f"这种结构最怕连亏，因为三次亏损就能抹掉十次盈利。"),
        })
    elif wr < 0.4 and avg_w > abs(avg_l) * 2:
        out.append({
            "title": "低胜率高赔率结构",
            "text": (f"胜率只有 {wr * 100:.0f}%，但赚的时候平均拿 {avg_w:,.0f} 元、"
                     f"亏的时候只有 {avg_l:,.0f} 元。这种结构能赚钱，但连续亏损段会很长，"
                     f"必须提前算好能扛几笔。"),
        })

    out.append({
        "title": "每笔交易的期望值",
        "text": (f"{exp:+,.0f} 元。{len(closed)} 轮累计 {sum(t['pnl'] for t in closed):+,.0f} 元，"
                 f"费用 {sum(t['fee'] for t in closed):,.0f} 元。"
                 + ("期望为正，问题在波动和仓位。"
                    if exp > 0 else "期望为负，做得越多亏得越多，先解决方向问题。")),
    })

    # 最好最差各占多少
    srt = sorted(closed, key=lambda t: t["pnl"])
    if len(closed) >= 10:
        bottom = sum(t["pnl"] for t in srt[:len(srt) // 10])
        total = sum(t["pnl"] for t in closed)
        if total > 0 and bottom < 0:
            out.append({
                "title": "最差的 10% 交易",
                "text": (f"这 {max(len(srt) // 10, 1)} 轮合计 {bottom:+,.0f} 元，"
                         f"占全部收益的 {abs(bottom) / total * 100:.0f}%。"
                         "把这部分剔掉收益能翻倍，说明亏损是集中在少数几笔上的。"),
            })
    return out
