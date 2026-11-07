"""离线自检（不联网、不开窗口）。

供 GitHub Actions 与本地共用同一份判定逻辑，避免 CI 与本地两套标准漂移。

覆盖四层：
  1. 语法层   —— compileall 全量编译，捕捉语法错误
  2. 导入层   —— 逐个 import 全部模块，捕捉循环依赖 / 缺依赖 / 顶层副作用爆炸
  3. 内核层   —— 用合成 OHLCV 跑通信号生成 + 撮合回测 + 指标汇总，不依赖行情接口
  4. 纯函数层 —— 表格排序数值解析、财报期推算等易回归的小逻辑

用法：
    python ci_check.py          全部执行，失败返回码 1
    python ci_check.py -v       打印每一步细节

注意：真·联网自检（真实行情 + 交割单链路）仍是 `python main.py --selftest`，
      因为依赖东财接口可用性，不进 CI 的每次推送流程。
"""
from __future__ import annotations

import os
import sys
import tokenize
import traceback
from datetime import date

# Qt 无显示环境：必须在 import PyQt6 之前设置
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv
ROOT = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {".venv", "venv", "build", "dist", "__pycache__", ".git", ".idea"}

MODULES = [
    "config", "data_feed", "datasource", "engine", "metrics", "optimizer",
    "riskx", "trades_io", "tablekit", "selection", "rotation", "multimkt",
    "events", "holders", "audit", "backtest", "ui_tabs", "main",
]

_fails: list[str] = []


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def bad(msg: str, exc: BaseException | None = None) -> None:
    _fails.append(msg)
    print(f"  [FAIL] {msg}")
    if exc is not None:
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(f"         {detail}")
        if VERBOSE:
            traceback.print_exc()


def step_syntax() -> None:
    """纯内存编译：不落 .pyc，Windows/Linux 行为一致（os.devnull 当 cfile 在 Windows 会炸）。"""
    print("[1/4] 语法编译")
    count = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT)
            try:
                with tokenize.open(path) as f:
                    compile(f.read(), rel, "exec")
                count += 1
            except Exception as exc:  # SyntaxError / UnicodeDecodeError
                bad(f"编译失败 {rel}", exc)
    ok(f"{count} 个 .py 文件编译通过")


def step_import() -> None:
    print("[2/4] 模块导入")
    import importlib

    for name in MODULES:
        try:
            importlib.import_module(name)
            if VERBOSE:
                ok(f"import {name}")
        except Exception as exc:
            bad(f"import {name}", exc)
    if not any(f.startswith("import ") for f in _fails):
        ok(f"{len(MODULES)} 个模块导入通过")


def step_engine() -> None:
    print("[3/4] 回测内核（合成数据，不联网）")
    try:
        import numpy as np
        import pandas as pd

        import engine
        import metrics
    except Exception as exc:
        bad("内核依赖导入", exc)
        return

    # 造一段有涨有跌、含放量缩量的日线，避免单边行情让均线信号退化
    n = 260
    rng = np.random.default_rng(20261107)
    drift = np.sin(np.linspace(0, 6 * np.pi, n)) * 0.012
    ret = drift + rng.normal(0, 0.015, n)
    close = 20.0 * np.cumprod(1 + ret)
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.004, n))
    vol = rng.integers(8_000_000, 60_000_000, n).astype(float)
    dates = pd.bdate_range("2025-01-02", periods=n).strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "date": dates, "open": open_, "high": np.maximum.reduce([high, open_, close]),
        "low": np.minimum.reduce([low, open_, close]), "close": close,
        "volume": vol,
    })

    try:
        sig = engine.make_signal(df, "双均线", fast=5, slow=20)
        if len(sig) != len(df):
            bad(f"信号长度 {len(sig)} != 行情长度 {len(df)}")
            return
        ok(f"信号生成 {int(sig.sum())}/{len(sig)} 根持仓")
    except Exception as exc:
        bad("make_signal 双均线", exc)
        return

    try:
        res = engine.run(df, sig, 100000.0, code="000001", name="自检标的")
        if not res.get("trades"):
            bad("回测未产生任何交易（撮合链路疑似中断）")
            return
        m = metrics.summarize(res, 100000.0)
        need = ["total_return", "annual_return", "max_drawdown", "win_rate",
                "trade_count", "sharpe", "calmar", "profit_loss_ratio",
                "profit_factor", "expectancy", "exposure", "turnover"]
        miss = [k for k in need if k not in m]
        if miss:
            bad(f"指标缺失 {miss}")
            return
        t = res["trades"][0]
        tneed = ["entry_date", "entry_price", "shares", "exit_date",
                 "exit_price", "hold_days", "net_pnl", "mfe_pct", "giveback_pct"]
        tmiss = [k for k in tneed if k not in t]
        if tmiss:
            bad(f"交易级字段缺失 {tmiss}")
            return
        ok(f"回测 {m['trade_count']} 笔，总收益 {m['total_return']}%，"
           f"回撤 {m['max_drawdown']}%，夏普 {m['sharpe']}")
        ok(f"交易级字段 {len(t)} 个齐备，结论翻译 {len(metrics.verdict(m))} 条")
    except Exception as exc:
        bad("run / summarize", exc)


def step_pure() -> None:
    print("[4/4] 纯函数（排序解析 / 财报期 / 指标口径）")
    try:
        import tablekit
    except Exception as exc:
        bad("import tablekit", exc)
    else:
        cases = [
            ("1,234.5 万", 1234.5), ("-12.3%", -12.3), ("+8.0pp", 8.0),
            ("3.2亿", 3.2), ("—", None), ("", None), (None, None),
        ]
        wrong = []
        for text, want in cases:
            got = tablekit.as_number(text)
            if want is None:
                if got is not None:
                    wrong.append(f"{text!r} 期望空值，得到 {got}")
            elif got is None or abs(float(got) - want) > 1e-6:
                wrong.append(f"{text!r} 期望 {want}，得到 {got}")
        if wrong:
            bad("as_number 解析回归：" + "；".join(wrong))
        else:
            ok(f"as_number {len(cases)} 个用例（含空值沉底）")

    try:
        import holders
        got = holders.recent_periods(4, today=date(2026, 11, 7))
        want = ["20260930", "20260630", "20260331", "20251231"]
        if got != want:
            bad(f"recent_periods 期望 {want}，得到 {got}")
        else:
            ok("recent_periods 季报期推算（含年报跨年）")
    except Exception as exc:
        bad("recent_periods", exc)


def main() -> int:
    print("=== QuantDesk 离线自检（CI 标准，不联网）===")
    for step in (step_syntax, step_import, step_engine, step_pure):
        try:
            step()
        except Exception as exc:  # 单步炸掉不阻断其余步骤
            bad(f"{step.__name__} 异常中断", exc)
        print()

    if _fails:
        print(f"=== 失败 {len(_fails)} 项 ===")
        for f in _fails:
            print(f"  - {f}")
        return 1
    print("=== 全部通过 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
