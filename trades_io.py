"""交割单解析 + 实盘交易复盘引擎。

券商导出的交割单各家列名不同（成交日期/发生日期、操作/买卖标志、
成交金额/发生金额 …），这里做别名归一，重建成"持仓 -> 平仓回合"。

配对口径：移动加权平均成本法，与券商 App 显示的持仓成本一致。
一轮 = 从 0 持仓开始买入，到再次清仓为止；中途加仓、减仓都并入同一轮。
手续费（佣金 + 印花税 + 过户费）计入成本与卖出净收入。
"""
from __future__ import annotations

import io
import os
import re
import pandas as pd

# --------------------------------------------------------------------------- #
# 列名归一
# --------------------------------------------------------------------------- #
ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("成交日期", "发生日期", "交易日期", "委托日期", "清算日期", "日期"),
    "time": ("成交时间", "交易时间", "委托时间"),
    "code": ("证券代码", "股票代码", "证券编码", "代码"),
    "name": ("证券名称", "股票名称", "证券简称", "名称"),
    "biz": ("业务名称", "业务类型", "业务说明", "摘要", "操作", "买卖标志",
            "买卖方向", "交易类别", "委托类别", "成交类型"),
    "price": ("成交价格", "成交均价", "成交价", "成交单价", "价格", "委托价格"),
    "qty": ("成交数量", "发生数量", "成交股数", "数量", "股数"),
    "amount": ("成交金额", "发生金额", "清算金额", "成交额", "金额"),
    "fee": ("手续费", "佣金", "费用合计", "交易费用", "总费用"),
    "stamp": ("印花税", "印花税费"),
    "transfer": ("过户费", "规费", "经手费", "证管费"),
}

# 不是买卖成交的行：打新申号、红利税、撤单、利息、逆回购等
SKIP_WORDS = ("申购", "中签", "配号", "缴款", "撤单", "利息", "结息",
              "银行", "托管", "指定", "回购")

# 转债面值：可转债「新股入账」无缴款记录时按面值补成本
BOND_PAR = 100.0


def _norm(s) -> str:
    return re.sub(r"\s+", "", str(s or "")).replace("（", "(").replace("）", ")")


def _num(v) -> float:
    """金额/数量清洗：去掉千分位、货币符号、括号负数。"""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        try:
            return float(v) if pd.notna(v) else 0.0
        except (TypeError, ValueError):
            return 0.0
    s = str(v).strip()
    if not s:
        return 0.0
    neg = s.startswith("(") and s.endswith(")") or s.startswith("-")
    s = s.replace(",", "").replace("，", "").replace("¥", "").replace("￥", "")
    s = s.replace("+", "").replace("(", "").replace(")", "").replace(" ", "")
    if s in ("", "-", "--", "—"):
        return 0.0
    try:
        f = float(s)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if not m:
            return 0.0
        f = float(m.group())
    return -abs(f) if neg else f


def _to_date(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return pd.Timestamp(v).normalize()
    s = str(v).strip()
    if not s:
        return None
    m = re.match(r"^(\d{4})[-/.年]?(\d{1,2})[-/.月]?(\d{1,2})", s)
    if m:
        try:
            return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    try:
        d = pd.to_datetime(s, errors="coerce")
        return None if pd.isna(d) else pd.Timestamp(d).normalize()
    except (ValueError, TypeError):
        return None


def classify(biz: str, amount: float, qty: float) -> tuple[str, str]:
    """业务名称 -> (动作, 说明)。

    动作：
      B 买入（含港股申报、新股缴款）
      S 卖出
      G 红股/送股入账（0 成本加股）
      N 转债中签入账（按面值补成本）
      D 现金分红（计入该轮收益）
      T 红利税补缴（冲减该轮收益）
      '' 非交易，跳过
    """
    t = _norm(biz)
    if not t:
        # 没有业务名称列时靠金额符号兜底（买入多为负数）
        if qty and amount:
            return ("B" if amount < 0 else "S"), ""
        return "", ""

    if "港股" in t and "买" in t:
        return "B", "港股申报"
    if "港股" in t and "卖" in t:
        return "S", "港股申报"

    # 先判明确的非交易事件，避免被「买」「卖」误捕
    if "红股" in t or "送股" in t or "转增" in t:
        return "G", "红股入账"
    if "新股入账" in t:
        return "N", "中签入账"
    if "税" in t and ("红利" in t or "股息" in t):
        return "T", "红利税"
    if "股息" in t or "红利" in t or "派息" in t:
        return "D", "现金分红"

    if "申购或认购" in t or "认购" in t:
        return "B", "新股缴款"
    if any(w in t for w in SKIP_WORDS):
        return "", ""

    if "卖" in t:
        return "S", ""
    if "买" in t:
        return "B", ""
    return "", ""


def _side(biz: str, amount: float, qty: float) -> str:
    """兼容旧调用：只要买卖方向。"""
    act, _ = classify(biz, amount, qty)
    return act if act in ("B", "S") else ""


# --------------------------------------------------------------------------- #
# 读文件
# --------------------------------------------------------------------------- #
def _read_text(path: str) -> pd.DataFrame:
    """按文本读：券商常见 GBK / GB18030 / UTF-8 BOM，制表符或逗号分隔。"""
    last_err = None
    for enc in ("utf-8-sig", "gbk", "gb18030", "utf-8"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                text = f.read()
            head = text[:4000]
            sep = "\t" if head.count("\t") > head.count(",") else ","
            return pd.read_csv(io.StringIO(text), header=None, dtype=object,
                               sep=sep, engine="python", skip_blank_lines=False)
        except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError, OSError) as e:
            last_err = e
            continue
    raise RuntimeError(f"无法识别文件编码（{last_err}），请另存为 UTF-8 或 xlsx 后重试")


def _read_raw(path: str) -> pd.DataFrame:
    """原样读入，表头行留给 _header_row 处理。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        return pd.read_excel(path, header=None, dtype=object)
    if ext == ".xls":
        # 招商等券商常把 GBK 文本直接存成 .xls，先按真 Excel 试，失败再按文本读
        try:
            df = pd.read_excel(path, header=None, dtype=object)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        return _read_text(path)
    return _read_text(path)


def _header_row(raw: pd.DataFrame, need: int = 2) -> int:
    """在前 40 行里找表头：命中别名最多的那一行。"""
    best, best_hit = -1, 0
    for i in range(min(40, len(raw))):
        cells = [_norm(v) for v in raw.iloc[i].tolist()]
        hit = 0
        for keys in ALIASES.values():
            if any(c in keys for c in cells):
                hit += 1
        if hit > best_hit:
            best, best_hit = i, hit
    if best_hit < need:
        raise RuntimeError("没找到交割单表头，请确认导出的是「交割单 / 对账单」明细")
    return best


def _map_columns(cols: list[str]) -> dict[str, int]:
    """别名 -> 列序号。"""
    mapping: dict[str, int] = {}
    norm = [_norm(c) for c in cols]
    for key, keys in ALIASES.items():
        for k in keys:
            for i, c in enumerate(norm):
                if c == k and key not in mapping:
                    mapping[key] = i
                    break
            if key in mapping:
                break
    # 模糊兜底：列名里含关键词
    for key, keys in ALIASES.items():
        if key in mapping:
            continue
        for k in keys:
            for i, c in enumerate(norm):
                if k in c and i not in mapping.values():
                    mapping[key] = i
                    break
            if key in mapping:
                break
    return mapping


def _dedup_issue(records: list[dict]) -> None:
    """打新记录去重。

    券商常把一次中签拆成「申购委托(金额0)」+「中签入账(金额>0)」两条，
    甚至更多。同代码只保留金额最大的那条，其余剔除。
    """
    keep: dict[tuple, int] = {}
    drop: set[int] = set()
    for i, r in enumerate(records):
        if r.get("note") != "新股缴款":
            continue
        k = (r["code"], r["side"])
        j = keep.get(k)
        if j is None:
            keep[k] = i
        else:
            if records[j]["amount"] >= r["amount"]:
                drop.add(i)
            else:
                drop.add(j)
                keep[k] = i
    for i in sorted(drop, reverse=True):
        records.pop(i)


def load_statement(path: str) -> tuple[list[dict], dict]:
    """解析交割单，返回 (标准化成交记录, 元信息)。"""
    raw = _read_raw(path)
    if raw is None or raw.empty:
        raise RuntimeError("文件是空的")

    hdr = _header_row(raw)
    cols = [_norm(v) if v is not None else "" for v in raw.iloc[hdr].tolist()]
    body = raw.iloc[hdr + 1:].reset_index(drop=True)
    idx = _map_columns(cols)

    for need in ("date", "code", "qty"):
        if need not in idx:
            raise RuntimeError(f"交割单缺少必要列（{need}），现有列：{'、'.join(c for c in cols if c)}")

    def cell(row, key, default=None):
        i = idx.get(key)
        if i is None or i >= len(row):
            return default
        v = row[i]
        return default if (v is None or (isinstance(v, float) and pd.isna(v))) else v

    records: list[dict] = []
    skipped = 0
    for _, row in body.iterrows():
        d = _to_date(cell(row, "date"))
        code_raw = str(cell(row, "code", "") or "").strip()
        code_raw = code_raw.strip().strip('"').replace("=", "")
        code = re.sub(r"\D", "", code_raw)
        if d is None or not code:
            continue
        qty = abs(_num(cell(row, "qty", 0)))
        price = _num(cell(row, "price", 0))
        amount = _num(cell(row, "amount", 0))
        biz = str(cell(row, "biz", "") or "")
        act, note = classify(biz, amount, qty)

        if not act:
            skipped += 1
            continue
        if act in ("G", "D", "T") and qty <= 0 and act != "D" and act != "T":
            skipped += 1
            continue

        market = "HK" if ("港股" in _norm(biz) or (len(code) == 5 and note == "港股申报")) else "A"
        full_code = ("HK" + code.zfill(5)) if market == "HK" else code.zfill(6)

        fee = abs(_num(cell(row, "fee", 0)))
        stamp = abs(_num(cell(row, "stamp", 0)))
        transfer = abs(_num(cell(row, "transfer", 0)))
        # 「手续费」若已是合计（≥印花税+过户费）就不要再叠加
        total_fee = fee if fee >= stamp + transfer and fee > 0 else fee + stamp + transfer

        if not price and amount and qty:
            price = abs(amount) / qty
        ts = d
        tstr = str(cell(row, "time", "") or "").strip()
        mt = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?", tstr)
        if mt:
            ts = d + pd.Timedelta(hours=int(mt.group(1)),
                                  minutes=int(mt.group(2)),
                                  seconds=int(mt.group(3) or 0))
        records.append({
            "date": d,
            "ts": ts,
            "code": full_code,
            "name": str(cell(row, "name", "") or "").strip(),
            "side": act,
            "market": market,
            "price": round(abs(price), 4),
            "qty": qty,
            # 红股入账没有现金对价，券商金额列是 0，不能按价格补算
            "amount": 0.0 if act == "G" else (abs(amount) if amount else round(abs(price) * qty, 2)),
            "fee": round(total_fee, 2),
            "biz": biz,
            "note": note,
        })

    if not records:
        raise RuntimeError("没解析出任何买卖成交，可能是表头或列名不匹配")

    # 转债「新股入账」没有缴款流水时，按面值补上成本
    bonds = {r["code"] for r in records
             if r["code"].startswith(("11", "12")) and r["market"] == "A"}
    for r in records:
        if r["side"] == "N" and r["code"] in bonds:
            r["amount"] = round(BOND_PAR * r["qty"], 2)
            r["price"] = BOND_PAR

    _dedup_issue(records)
    # 有成交时间就按时间还原真实先后，否则同日买入优先
    records.sort(key=lambda r: (r.get("ts") or r["date"], 0 if r["side"] == "B" else 1))
    has_fee = any(r["fee"] > 0 for r in records)
    meta = {
        "path": path,
        "header_row": hdr,
        "columns": {k: cols[v] for k, v in idx.items() if v < len(cols)},
        "records": len(records),
        "skipped": skipped,
        "start": str(records[0]["date"].date()),
        "end": str(records[-1]["date"].date()),
        "codes": len({r["code"] for r in records}),
        "has_fee": has_fee,
        "n_a": len({r["code"] for r in records if r["market"] == "A"}),
        "n_hk": len({r["code"] for r in records if r["market"] == "HK"}),
        "events": {k: sum(1 for r in records if r["side"] == k) for k in "BSGDNT"},
    }
    return records, meta


# --------------------------------------------------------------------------- #
# 配对重建
# --------------------------------------------------------------------------- #
def rebuild(records: list[dict], comm_rate: float = 0.00025,
            comm_min: float = 5.0, est_fee: bool = None) -> dict:
    """按移动加权平均成本重建持仓与平仓回合。

    卖出前无持仓的记录不再静默丢弃，单独归入 initial（期初持仓卖出），
    因为导出区间之前的买入不在文件里，成本无从得知。
    """
    state: dict[str, dict] = {}
    closed: list[dict] = []
    initial: list[dict] = []
    warns: list[str] = []
    need_est = (not any(r["fee"] > 0 for r in records)) if est_fee is None else est_fee

    def fee_of(r):
        if r["fee"] > 0:
            return r["fee"]
        if not need_est:
            return 0.0
        amt = r["amount"] or r["price"] * r["qty"]
        if r.get("market") == "HK":
            # 港股：印花税 0.1% + 交易费/征费约 0.0085% + 佣金
            return round(amt * 0.001 + max(amt * comm_rate, 3.0) + amt * 0.000085, 2)
        if r["code"].startswith(("11", "12")) and r["side"] in ("B", "S"):
            # 转债：佣金万 0.1、无最低、免印花税
            return round(max(amt * 0.00001, 0.05), 2)
        comm = max(amt * comm_rate, comm_min)
        stamp = amt * 0.0005 if r["side"] == "S" else 0.0
        return round(comm + stamp + amt * 0.00001, 2)

    for r in records:
        code = r["code"]
        st = state.setdefault(code, {
            "qty": 0.0, "cost": 0.0, "name": r["name"], "market": r.get("market", "A"),
            "open_date": None, "buy_amt": 0.0, "sell_amt": 0.0,
            "n_buy": 0, "n_sell": 0, "fee": 0.0, "first": None,
            "dividend": 0.0, "tax": 0.0, "realized": 0.0,
        })
        if r["name"] and len(r["name"]) < len(st["name"] or ""):
            st["name"] = r["name"]          # 「国富量子(五百)」这类长名归到短名
        elif r["name"] and not st["name"]:
            st["name"] = r["name"]

        act = r["side"]

        if act in ("G", "N"):                       # 红股 / 中签入账：加股，成本按入账额
            if st["qty"] <= 1e-9 and st["open_date"] is None:
                st["open_date"] = r["date"]
            st["qty"] += r["qty"]
            st["cost"] += r["amount"]
            if r["amount"]:                         # 转债中签有面值成本，红股为 0
                st["buy_amt"] += r["amount"]
            continue

        if act in ("D", "T"):                       # 分红 / 红利税：计入本轮现金流
            cash = r["amount"] if act == "D" else -r["amount"]
            if st["qty"] > 1e-9:
                st["sell_amt"] += cash
                st["realized"] += cash
                st["dividend" if act == "D" else "tax"] += abs(r["amount"])
            else:
                # 已清仓后到账（税补缴常见），挂到这笔票最近一轮上
                last = None
                for t in reversed(closed):
                    if t["code"] == code:
                        last = t
                        break
                if last is not None:
                    last["pnl"] = round(last["pnl"] + cash, 2)
                    last["sell_amt"] = round(last["sell_amt"] + cash, 2)
                    last["ret"] = round(last["pnl"] / last["buy_amt"] * 100, 2) if last["buy_amt"] else 0.0
                    last["dividend" if act == "D" else "tax"] = round(
                        last.get("dividend", 0.0) + abs(r["amount"]), 2)
            continue

        if act == "B":
            if st["qty"] <= 1e-9:                 # 从空仓起买 -> 开新一轮
                st.update(open_date=r["date"], buy_amt=0.0, sell_amt=0.0,
                          n_buy=0, n_sell=0, cost=0.0, fee=0.0,
                          dividend=0.0, tax=0.0, realized=0.0)
                if st["first"] is None:
                    st["first"] = r["date"]
            gross = r["amount"] or r["price"] * r["qty"]
            f = fee_of(r)
            st["qty"] += r["qty"]
            st["cost"] += gross + f
            st["buy_amt"] += gross + f
            st["fee"] += f
            st["n_buy"] += 1
        else:                                      # 卖出
            if st["qty"] <= 1e-9:
                if r["qty"] > 0:
                    initial.append({
                        "code": code, "name": st["name"], "date": r["date"],
                        "qty": r["qty"], "amount": round(r["amount"], 2),
                    })
                    warns.append(f"{r['date'].date()} {code} {st['name']} 卖出 {r['qty']:.0f} 股，"
                                 f"但区间内无对应买入（期初持仓，成本未知）")
                continue
            q = min(r["qty"], st["qty"])
            if r["qty"] > st["qty"] + 1e-6:
                warns.append(f"{r['date'].date()} {code} 卖出 {r['qty']:.0f} 股超过持仓 "
                             f"{st['qty']:.0f}，按持仓截断")
            gross = r["amount"] or r["price"] * r["qty"]
            f = fee_of(r)
            net = gross - f
            avg = st["cost"] / st["qty"] if st["qty"] else 0.0
            st["cost"] -= avg * q
            st["realized"] += net - avg * q
            st["qty"] -= q
            st["sell_amt"] += net
            st["fee"] += f
            st["n_sell"] += 1
            if st["qty"] <= 1e-9:
                _close(st, closed, code, r)

    open_pos = []
    for code, st in state.items():
        if st["qty"] > 1e-9:
            open_pos.append({
                "code": code,
                "name": st["name"],
                "market": st.get("market", "A"),
                "qty": st["qty"],
                "cost": round(st["cost"], 2),
                "cost_price": round(st["cost"] / st["qty"], 4),
                "open_date": st["open_date"],
                "fee": round(st["fee"], 2),
                "dividend": round(st.get("dividend", 0.0), 2),
                "realized": round(st["realized"], 2),
                "sell_amt": round(st["sell_amt"], 2),
                "buy_amt": round(st["buy_amt"], 2),
                "n_buy": st["n_buy"],
                "n_sell": st["n_sell"],
            })

    closed.sort(key=lambda t: t["exit_date"])
    return {"closed": closed, "open": open_pos, "initial": initial, "warns": warns,
            "est_fee": need_est, "summary": summarize(closed, open_pos, initial, need_est)}


def _close(st: dict, closed: list, code: str, r: dict) -> None:
    pnl = st["realized"]
    closed.append({
        "code": code,
        "name": st["name"],
        "market": st.get("market", "A"),
        "entry_date": st["open_date"],
        "exit_date": r["date"],
        "buy_amt": round(st["buy_amt"], 2),
        "sell_amt": round(st["sell_amt"], 2),
        "pnl": round(pnl, 2),
        "ret": round(pnl / st["buy_amt"] * 100, 2) if st["buy_amt"] else 0.0,
        "days": int((r["date"] - st["open_date"]).days) if st["open_date"] else 0,
        "n_buy": st["n_buy"],
        "n_sell": st["n_sell"],
        "fee": round(st["fee"], 2),
        "dividend": round(st.get("dividend", 0.0), 2),
    })
    st.update(qty=0.0, cost=0.0)


def summarize(closed: list[dict], open_pos: list[dict],
              initial: list[dict] = None, est_fee: bool = False) -> dict:
    initial = initial or []
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    tot = sum(t["pnl"] for t in closed)
    avg_w = sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
    avg_l = sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0
    # 连续亏损最长纪录
    streak = worst_streak = 0
    for t in closed:
        streak = streak + 1 if t["pnl"] <= 0 else 0
        worst_streak = max(worst_streak, streak)
    return {
        "n_closed": len(closed),
        "n_open": len(open_pos),
        "n_initial": len(initial),
        "initial_amt": round(sum(x["amount"] for x in initial), 2),
        "win_rate": round(len(wins) / len(closed) * 100, 2) if closed else 0.0,
        "realized": round(tot, 2),
        "realized_all": round(tot + sum(o.get("realized", 0.0) for o in open_pos), 2),
        "avg_win": round(avg_w, 2),
        "avg_loss": round(avg_l, 2),
        "pl_ratio": round(abs(avg_w / avg_l), 2) if avg_l else 0.0,
        "profit_factor": round(sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses)), 2)
        if losses and sum(t["pnl"] for t in losses) else 0.0,
        "best": round(max((t["pnl"] for t in closed), default=0.0), 2),
        "worst": round(min((t["pnl"] for t in closed), default=0.0), 2),
        "avg_days": round(sum(t["days"] for t in closed) / len(closed), 1) if closed else 0.0,
        "days_win": round(sum(t["days"] for t in wins) / len(wins), 1) if wins else 0.0,
        "days_loss": round(sum(t["days"] for t in losses) / len(losses), 1) if losses else 0.0,
        "total_fee": round(sum(t["fee"] for t in closed), 2),
        "fee_est": bool(est_fee),
        "invested": round(sum(t["buy_amt"] for t in closed), 2),
        "worst_streak": worst_streak,
        "avg_ret": round(sum(t["ret"] for t in closed) / len(closed), 2) if closed else 0.0,
    }


def equity_curve(closed: list[dict]) -> list[tuple[str, float]]:
    """累计已实现盈亏曲线（按平仓日）。"""
    out, acc = [], 0.0
    for t in sorted(closed, key=lambda x: x["exit_date"]):
        acc += t["pnl"]
        out.append((str(pd.Timestamp(t["exit_date"]).date()), round(acc, 2)))
    return out


# --------------------------------------------------------------------------- #
# 造一份样例交割单（仅自测用，不参与运行）
# --------------------------------------------------------------------------- #
def _make_sample() -> list[dict]:
    def rec(d, code, name, side, px, qty, amt, fee):
        return {"date": pd.Timestamp(d), "code": code, "name": name, "side": side,
                "market": "A", "price": px, "qty": qty, "amount": amt,
                "fee": fee, "biz": "", "note": ""}
    return [
        rec("2026-02-10", "603011", "合锻智能", "B", 26.4, 1000, 26400.0, 8.0),
        rec("2026-03-17", "603011", "合锻智能", "S", 23.69, 1000, 23690.0, 30.0),
        rec("2026-04-15", "002407", "多氟多", "B", 21.32, 2000, 42640.0, 13.0),
        rec("2026-04-20", "002407", "多氟多", "B", 19.0, 1000, 19000.0, 6.0),
        rec("2026-04-29", "002407", "多氟多", "S", 20.27, 3000, 60810.0, 76.0),
    ]
