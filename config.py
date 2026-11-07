"""关注股清单 + 自定义股票池持久化。

默认池覆盖能源/核聚变、半导体、面板/封测、光通信、光伏锂电、军工、金融等板块。
用户自己添加的标的会写进 ~/.quantdesk/watchlist.json，重启后仍在。
"""
from __future__ import annotations

import json
import os
import datetime

# 默认关注池（首次运行或配置文件丢失时使用）
WATCHLIST = [
    # 原关注 / 活跃追踪
    {"code": "603011", "name": "合锻智能"},
    {"code": "002407", "name": "多氟多"},
    {"code": "688696", "name": "极米科技"},
    {"code": "002792", "name": "通宇通讯"},
    {"code": "002537", "name": "海联金汇"},
    {"code": "688002", "name": "睿创微纳"},
    {"code": "600869", "name": "远东股份"},
    {"code": "000617", "name": "中油资本"},
    {"code": "002119", "name": "康强电子"},
    {"code": "601636", "name": "旗滨集团"},
    {"code": "603650", "name": "彤程新材"},
    # 光通信 / 算力
    {"code": "300308", "name": "中际旭创"},
    {"code": "300502", "name": "新易盛"},
    {"code": "002281", "name": "光迅科技"},
    # 半导体封测 / 面板
    {"code": "600584", "name": "长电科技"},
    {"code": "002156", "name": "通富微电"},
    {"code": "000725", "name": "京东方A"},
    {"code": "000100", "name": "TCL科技"},
    # 核聚变 / 超导 / 核电
    {"code": "688122", "name": "西部超导"},
    {"code": "600363", "name": "联创光电"},
    {"code": "601985", "name": "中国核电"},
    # 光伏 / 锂电
    {"code": "300274", "name": "阳光电源"},
    {"code": "601012", "name": "隆基绿能"},
    {"code": "002466", "name": "天齐锂业"},
    {"code": "002460", "name": "赣锋锂业"},
    # 军工
    {"code": "600760", "name": "中航沈飞"},
    {"code": "600893", "name": "航发动力"},
    # 金融 / 金融科技
    {"code": "600036", "name": "招商银行"},
    {"code": "300059", "name": "东方财富"},
]

# 默认回测参数
DEFAULT_START = "20240101"
DEFAULT_END = datetime.date.today().strftime("%Y%m%d")
INIT_CASH = 100000.0

# 看盘默认拉取的 K 线根数（日K 约 2 年，周K/月K 更长）
BARS_DAILY_YEARS = 2
BARS_WEEKLY_YEARS = 4
BARS_MONTHLY_YEARS = 12

_CFG_DIR = os.path.join(os.path.expanduser("~"), ".quantdesk")
_CFG_FILE = os.path.join(_CFG_DIR, "watchlist.json")


def _norm(item) -> dict | None:
    try:
        code = str(item.get("code", "")).strip().zfill(6)
        if not code.isdigit() or len(code) != 6:
            return None
        return {"code": code, "name": str(item.get("name", "")).strip() or code}
    except Exception:
        return None


def load_watchlist() -> list[dict]:
    """读取股票池：优先用户配置，无则用默认池。"""
    try:
        with open(_CFG_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        items = [x for x in (_norm(d) for d in raw if isinstance(d, dict)) if x]
        # 去重保序
        seen, out = set(), []
        for it in items:
            if it["code"] in seen:
                continue
            seen.add(it["code"])
            out.append(it)
        if out:
            return out
    except Exception:
        pass
    return [dict(x) for x in WATCHLIST]


def save_watchlist(items: list[dict]) -> bool:
    """把股票池写盘。失败不抛异常（只影响下次启动的默认池）。"""
    try:
        clean = [x for x in (_norm(d) for d in items) if x]
        os.makedirs(_CFG_DIR, exist_ok=True)
        tmp = _CFG_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _CFG_FILE)
        return True
    except Exception:
        return False


def reset_watchlist() -> list[dict]:
    """恢复默认池。"""
    items = [dict(x) for x in WATCHLIST]
    save_watchlist(items)
    return items
