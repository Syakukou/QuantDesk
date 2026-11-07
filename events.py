"""事件雷达：公告分类、龙虎榜、解禁、增减持、回购、新股、资金流，附财务质量打分。

这些数据券商客户端都有零散入口，但没有一个地方把它们按「对我持仓的影响」
排成一列。这里的排序逻辑是：离今天越近、指向我持仓越明确，越靠前。

所有全量接口都落盘缓存，TTL 12 小时。东财全量增减持要 170 秒，不能每次点都重拉。
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".quantdesk", "cache")
TTL = 12 * 3600


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


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


# ---------------------------------------------------------------- 公告雷达

# 关键词 -> (标签, 方向, 强度 1-3)
RULES = [
    ("减持", "股东减持", "risk", 3),
    ("增持", "股东增持", "chance", 2),
    ("回购", "股份回购", "chance", 2),
    ("立案", "立案调查", "risk", 3),
    ("处罚", "监管处罚", "risk", 3),
    ("问询", "交易所问询", "risk", 2),
    ("警示", "风险警示", "risk", 3),
    ("退市", "退市风险", "risk", 3),
    ("质押", "股权质押", "risk", 2),
    ("冻结", "股份冻结", "risk", 2),
    ("诉讼", "诉讼仲裁", "risk", 2),
    ("预亏", "业绩预亏", "risk", 3),
    ("预减", "业绩预减", "risk", 2),
    ("首亏", "首次亏损", "risk", 3),
    ("减值", "资产减值", "risk", 2),
    ("预增", "业绩预增", "chance", 3),
    ("扭亏", "扭亏为盈", "chance", 3),
    ("中标", "重大中标", "chance", 3),
    ("签订", "重大合同", "chance", 2),
    ("收购", "收购重组", "chance", 2),
    ("激励", "股权激励", "chance", 2),
    ("分红", "利润分配", "chance", 1),
    ("解禁", "限售解禁", "risk", 2),
    ("停牌", "停牌", "risk", 2),
    ("复牌", "复牌", "chance", 1),
]


def notice_radar(date: str | None = None, force: bool = False) -> pd.DataFrame:
    """当日全市场公告，按风险/机会打标并按强度排序。"""
    d = date or _today()
    import akshare as ak
    df = _cached(f"notice_{d}.pkl",
                 lambda: ak.stock_notice_report(symbol="全部", date=d),
                 force=force, ttl=3 * 3600)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    title = df["公告标题"].astype(str)

    def _tag(t: str):
        hits = [(r[1], r[2], r[3]) for r in RULES if r[0] in t]
        if not hits:
            return pd.Series({"标签": "常规公告", "方向": "neutral", "强度": 0})
        hits.sort(key=lambda x: -x[2])
        return pd.Series({"标签": hits[0][0], "方向": hits[0][1], "强度": hits[0][2]})

    tags = title.apply(_tag)
    df = pd.concat([df.reset_index(drop=True), tags.reset_index(drop=True)], axis=1)
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    df = df.sort_values(["强度", "方向"], ascending=[False, True])
    return df


def notice_text(url: str, force: bool = False) -> dict:
    """按公告详情页地址取正文全文。

    东财详情页是 JS 渲染的，直接抓 HTML 只有空壳；正文在
    np-cnotice-stock 的接口里，art_code 从网址尾巴上就能拿到。
    正文落盘缓存 7 天，一条公告只拉一次。
    """
    m = re.search(r"(AN\d+)\.html", str(url or ""))
    if not m:
        return {"ok": False, "msg": "这条公告没有可解析的编号，只能看标题"}
    art = m.group(1)
    path = os.path.join(CACHE_DIR, f"ann_{art}.txt")
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not force and os.path.exists(path):
        if time.time() - os.path.getmtime(path) < 7 * 24 * 3600:
            try:
                body = open(path, encoding="utf-8").read()
                if body.strip():
                    return {"ok": True, "art": art, "text": body, "pdf": "",
                            "from_cache": True}
            except OSError:
                pass

    import requests
    api = "https://np-cnotice-stock.eastmoney.com/api/content/ann"
    try:
        r = requests.get(api, params={"art_code": art, "client_source": "web",
                                      "page_index": 1},
                         headers={"User-Agent": "Mozilla/5.0",
                                  "Referer": "https://data.eastmoney.com/"},
                         timeout=15)
        data = (r.json() or {}).get("data") or {}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "msg": f"拉正文失败：{exc}"}

    raw = str(data.get("notice_content") or "")
    # 源文本每行前塞满对齐空格，直接展示会全是空白，先压掉连续空格
    lines = [re.sub(r"[ \t\u3000]{2,}", " ", ln).strip() for ln in raw.splitlines()]
    body = "\n".join([ln for ln in lines if ln])
    pdf = str(data.get("attach_url") or "")
    if not body:
        return {"ok": False, "msg": "接口没有返回正文，可能只有 PDF 附件",
                "pdf": pdf, "art": art}
    try:
        open(path, "w", encoding="utf-8").write(body)
    except OSError:
        pass
    return {"ok": True, "art": art, "text": body, "pdf": pdf,
            "pages": data.get("page_size")}


def notice_summary(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"total": 0}
    risk = df[df["方向"] == "risk"]
    chance = df[df["方向"] == "chance"]
    return {
        "total": len(df),
        "risk": len(risk),
        "chance": len(chance),
        "risk_strong": int((risk["强度"] >= 3).sum()),
        "by_tag": (df[df["方向"] != "neutral"]
                   .groupby(["标签", "方向"]).size()
                   .sort_values(ascending=False).head(12).to_dict()),
    }


# ---------------------------------------------------------------- 龙虎榜

def lhb(start: str | None = None, end: str | None = None,
        days: int = 20, force: bool = False) -> pd.DataFrame:
    """龙虎榜明细。接口自带「上榜后 1/2/5/10 日」收益，可直接验证跟单有效性。"""
    import akshare as ak
    e = end or _today()
    s = start or (datetime.strptime(e, "%Y%m%d") - timedelta(days=days)).strftime("%Y%m%d")
    df = _cached(f"lhb_{s}_{e}.pkl",
                 lambda: ak.stock_lhb_detail_em(start_date=s, end_date=e),
                 force=force)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    for c in ["上榜后1日", "上榜后2日", "上榜后5日", "上榜后10日"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def lhb_seats(code: str, date: str, force: bool = False) -> dict:
    """某只票某天的龙虎榜席位明细：买方前五、卖方前五。

    接口给的比例是小数，这里统一换成百分比；「机构专用」「沪股通/深股通专用」
    单独打标，因为跟单跟的是这两类钱的动向，不是营业部名字好不好听。
    """
    import akshare as ak
    d = str(date).replace("-", "")[:8]
    c = str(code).zfill(6)
    out = {"code": c, "date": d, "buy": pd.DataFrame(), "sell": pd.DataFrame(),
           "err": ""}

    def _one(flag: str) -> pd.DataFrame:
        def _fetch():
            return ak.stock_lhb_stock_detail_em(symbol=c, date=d, flag=flag)
        try:
            df = _cached(f"lhbseat_{c}_{d}_{flag}.pkl", _fetch, force=force,
                         ttl=7 * 24 * 3600)
        except Exception as exc:  # noqa: BLE001
            out["err"] = f"{exc}"
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        x = df.copy()
        for col in ["买入金额", "卖出金额", "净额",
                    "买入金额-占总成交比例", "卖出金额-占总成交比例"]:
            if col in x.columns:
                x[col] = pd.to_numeric(x[col], errors="coerce")
        for col in ["买入金额-占总成交比例", "卖出金额-占总成交比例"]:
            if col in x.columns and x[col].abs().max() <= 1.5:
                x[col] = x[col] * 100
        name = x["交易营业部名称"].astype(str)

        def _kind(s: str) -> str:
            if "机构专用" in s:
                return "机构"
            if "沪股通" in s or "深股通" in s:
                return "北向"
            if "东方财富" in s and "拉萨" in s:
                return "散户大营"
            return "游资/营业部"

        x["性质"] = name.map(_kind)
        return x

    out["buy"] = _one("买入")
    out["sell"] = _one("卖出")
    return out


def lhb_verdict(df: pd.DataFrame) -> list[dict]:
    """按上榜原因/席位解读分组，看跟单到底赚不赚。只有样本够的组才给结论。

    分组名不截断：截断会让两条不同的长原因名撞成一条，下钻时对不上。
    """
    if df is None or df.empty:
        return []
    out = []
    col = "上榜后5日" if "上榜后5日" in df.columns else "上榜后1日"
    if col not in df.columns:
        return []
    d = df.copy()

    if "上榜原因" in d.columns:
        g = d.groupby("上榜原因")[col].agg(["count", "mean", "median"])
        g = g[g["count"] >= 5].sort_values("mean", ascending=False)
        for reason, row in g.iterrows():
            out.append({
                "维度": "上榜原因",
                "分组": str(reason),
                "样本": int(row["count"]),
                "均值": round(float(row["mean"]), 2),
                "中位": round(float(row["median"]), 2),
            })

    if "解读" in d.columns:
        g = d.groupby("解读")[col].agg(["count", "mean", "median"])
        g = g[g["count"] >= 5].sort_values("mean", ascending=False)
        for reason, row in g.iterrows():
            out.append({
                "维度": "席位解读",
                "分组": str(reason),
                "样本": int(row["count"]),
                "均值": round(float(row["mean"]), 2),
                "中位": round(float(row["median"]), 2),
            })
    return out


# ---------------------------------------------------------------- 解禁

def unlock(start: str | None = None, end: str | None = None,
           days: int = 30, force: bool = False) -> pd.DataFrame:
    """未来解禁明细，按「解禁市值占流通市值比」排冲击力。"""
    import akshare as ak
    s = start or _today()
    e = end or (datetime.strptime(s, "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d")
    df = _cached(f"unlock_{s}_{e}.pkl",
                 lambda: ak.stock_restricted_release_detail_em(start_date=s, end_date=e),
                 force=force, ttl=6 * 3600)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    for c in ["实际解禁市值", "占解禁前流通市值比例", "解禁前20日涨跌幅"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 占比口径修正：接口给的是小数，转成百分比
    if "占解禁前流通市值比例" in df.columns:
        v = df["占解禁前流通市值比例"]
        if v.abs().max() < 1:
            df["占流通%"] = v * 100
        else:
            df["占流通%"] = v
    df["冲击"] = (df["实际解禁市值"].fillna(0) / 1e8) * df.get(
        "占流通%", pd.Series(0, index=df.index)).fillna(0).clip(lower=0.01)
    return df.sort_values("实际解禁市值", ascending=False)


def unlock_daily(start: str | None = None, days: int = 30) -> pd.DataFrame:
    """按日汇总解禁家数与市值，用来排未来日历。"""
    import akshare as ak
    s = start or _today()
    e = (datetime.strptime(s, "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d")
    try:
        df = ak.stock_restricted_release_summary_em(
            symbol="全部股票", start_date=s, end_date=e)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["解禁市值亿"] = pd.to_numeric(df["实际解禁市值"], errors="coerce") / 1e8
    return df.sort_values("解禁时间")


# ---------------------------------------------------------------- 增减持 / 回购 / 新股

def hold_changes(days: int = 30, force: bool = False) -> pd.DataFrame:
    """董监高及股东增减持明细。全量 17 万条要 170 秒，必须缓存。

    方向以「变动股数」的正负为准，不看「变动原因」的文字：同一个原因
    「集中竞价」既能是买也能是卖，只有股数符号是可靠的。
    """
    import akshare as ak
    df = _cached("hold_mgmt.pkl", lambda: ak.stock_hold_management_detail_em(),
                 force=force, ttl=24 * 3600)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    cut = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    df = df[df["日期"] >= cut]
    for c in ["变动股数", "变动金额", "成交均价", "变动后持股数", "变动比例"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    sh = df["变动股数"].fillna(0)
    df["方向"] = np.where(sh > 0, "增持", np.where(sh < 0, "减持", "持平"))
    # 金额统一成「正数=增持、负数=减持」，避免各源符号口径不一致
    if "变动金额" in df.columns:
        amt = df["变动金额"].fillna(0).abs()
        df["金额"] = np.where(sh >= 0, amt, -amt)
    else:
        df["金额"] = 0.0
    return df.sort_values("日期", ascending=False)


def hold_changes_summary(df: pd.DataFrame) -> pd.DataFrame:
    """按股票汇总：增持笔数、减持笔数、增持额、减持额、净额。"""
    if df is None or df.empty or "金额" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["增持笔"] = (d["方向"] == "增持").astype(int)
    d["减持笔"] = (d["方向"] == "减持").astype(int)
    d["增持额"] = np.where(d["方向"] == "增持", d["金额"].abs(), 0.0)
    d["减持额"] = np.where(d["方向"] == "减持", d["金额"].abs(), 0.0)
    g = d.groupby(["代码", "名称"]).agg(
        笔数=("金额", "size"),
        增持笔=("增持笔", "sum"),
        减持笔=("减持笔", "sum"),
        增持额=("增持额", "sum"),
        减持额=("减持额", "sum"),
        净额=("金额", "sum"),
    ).reset_index()
    return g.sort_values("净额", ascending=False)


def repurchase(force: bool = False) -> pd.DataFrame:
    import akshare as ak
    df = _cached("repurchase.pkl", lambda: ak.stock_repurchase_em(), force=force)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    if "计划回购金额区间-上限" in df.columns:
        df["计划金额亿"] = pd.to_numeric(
            df["计划回购金额区间-上限"], errors="coerce") / 1e8
    return df


def ipo_recent(force: bool = False) -> pd.DataFrame:
    import akshare as ak
    df = _cached("ipo.pkl", lambda: ak.stock_xgsr_ths(), force=force)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["上市日期"] = pd.to_datetime(df["上市日期"], errors="coerce")
    df = df.sort_values("上市日期", ascending=False)
    return df


def fund_flow(indicator: str = "今日", force: bool = False) -> pd.DataFrame:
    import akshare as ak
    df = _cached(f"flow_{indicator}.pkl",
                 lambda: ak.stock_individual_fund_flow_rank(indicator=indicator),
                 force=force, ttl=6 * 3600)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    return df


# ---------------------------------------------------------------- 财务质量

def f_score(code: str) -> dict:
    """Piotroski F-score（9 项）。数据源是新浪财务指标，按季一行。

    A 股口径下 9 项里有两项只能近似：
    - 「未增发新股」用总股本变化判断，拆股会误判
    - 「经营现金流 > 净利润」用每股经营现金流与 EPS 比较
    近似项单独标注，不混进确定项里充数。
    """
    import akshare as ak
    year = max(datetime.now().year - 2, 2015)
    try:
        df = ak.stock_financial_analysis_indicator(symbol=code, start_year=str(year))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "msg": f"取财务数据失败: {exc}"}
    if df is None or df.empty:
        return {"ok": False, "msg": "该股无财务指标数据"}

    d = df.copy()
    d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
    d = d.sort_values("日期").reset_index(drop=True)
    if len(d) < 5:
        return {"ok": False, "msg": f"只有 {len(d)} 期财务数据，不够做同比"}

    def pick(row, *names):
        for n in names:
            if n in row.index:
                v = pd.to_numeric(pd.Series([row[n]]), errors="coerce").iloc[0]
                if pd.notna(v):
                    return float(v)
        return np.nan

    cur, prev = d.iloc[-1], d.iloc[-5]  # 相隔 4 个季度 = 同比
    items = []

    roa = pick(cur, "总资产净利润率(%)")
    items.append(("资产回报为正", 1 if (pd.notna(roa) and roa > 0) else 0,
                  f"ROA {roa:.2f}%" if pd.notna(roa) else "ROA 缺失"))

    cfo = pick(cur, "每股经营性现金流(元)")
    items.append(("经营现金流为正", 1 if (pd.notna(cfo) and cfo > 0) else 0,
                  f"每股经营现金流 {cfo:.2f}" if pd.notna(cfo) else "缺失"))

    roa_p = pick(prev, "总资产净利润率(%)")
    items.append(("ROA 同比改善",
                  1 if (pd.notna(roa) and pd.notna(roa_p) and roa > roa_p) else 0,
                  f"{roa_p:.2f}% → {roa:.2f}%" if pd.notna(roa_p) and pd.notna(roa) else "缺失"))

    eps = pick(cur, "摊薄每股收益(元)")
    items.append(("现金流覆盖利润",
                  1 if (pd.notna(cfo) and pd.notna(eps) and cfo > eps) else 0,
                  f"现金流 {cfo:.2f} vs EPS {eps:.2f}"
                  if pd.notna(cfo) and pd.notna(eps) else "缺失"))

    lev = pick(cur, "资产负债率(%)")
    lev_p = pick(prev, "资产负债率(%)")
    items.append(("杠杆下降",
                  1 if (pd.notna(lev) and pd.notna(lev_p) and lev < lev_p) else 0,
                  f"{lev_p:.1f}% → {lev:.1f}%" if pd.notna(lev_p) and pd.notna(lev) else "缺失"))

    cr = pick(cur, "流动比率")
    cr_p = pick(prev, "流动比率")
    items.append(("流动性改善",
                  1 if (pd.notna(cr) and pd.notna(cr_p) and cr > cr_p) else 0,
                  f"{cr_p:.2f} → {cr:.2f}" if pd.notna(cr_p) and pd.notna(cr) else "缺失"))

    gm = pick(cur, "销售毛利率(%)")
    gm_p = pick(prev, "销售毛利率(%)")
    items.append(("毛利率改善",
                  1 if (pd.notna(gm) and pd.notna(gm_p) and gm > gm_p) else 0,
                  f"{gm_p:.1f}% → {gm:.1f}%" if pd.notna(gm_p) and pd.notna(gm) else "缺失"))

    tat = pick(cur, "总资产周转率(次)")
    tat_p = pick(prev, "总资产周转率(次)")
    items.append(("周转率改善",
                  1 if (pd.notna(tat) and pd.notna(tat_p) and tat > tat_p) else 0,
                  f"{tat_p:.3f} → {tat:.3f}" if pd.notna(tat_p) and pd.notna(tat) else "缺失"))

    # 应计质量：净利润里有多少没收到现金
    ocf_ratio = pick(cur, "经营现金净流量与净利润的比率(%)")
    if pd.notna(ocf_ratio):
        items.append(("利润含现金量", 1 if ocf_ratio > 80 else 0,
                      f"经营现金流/净利润 {ocf_ratio:.0f}%"))
    else:
        items.append(("利润含现金量", 0, "数据缺失，记 0 分"))

    score = sum(i[1] for i in items)
    return {
        "ok": True,
        "code": code,
        "period": str(cur["日期"].date()) if pd.notna(cur["日期"]) else "?",
        "score": int(score),
        "max": len(items),
        "items": items,
        "verdict": _f_verdict(score, len(items)),
    }


def _f_verdict(score: int, mx: int) -> str:
    pct = score / mx if mx else 0
    if pct >= 0.78:
        return f"{score}/{mx}，财务面扎实，可以重点看"
    if pct >= 0.55:
        return f"{score}/{mx}，中上，有几处要盯"
    if pct >= 0.33:
        return f"{score}/{mx}，偏弱，赚钱质量存疑"
    return f"{score}/{mx}，财务面很差，先排除"
