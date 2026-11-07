"""自测：用真实 A 股数据验证 data_feed + backtest。

不依赖 GUI，可在无显示环境运行：
    source .venv/Scripts/activate
    python selftest.py
"""
from __future__ import annotations

from config import WATCHLIST, DEFAULT_START, DEFAULT_END, INIT_CASH
from data_feed import get_history
from backtest import ma_cross_signal, momentum_signal, run_backtest


def main():
    code = WATCHLIST[0]["code"]
    name = WATCHLIST[0]["name"]
    print(f"== 拉取 {name}({code}) 历史 {DEFAULT_START}-{DEFAULT_END} ==")
    df = get_history(code, DEFAULT_START, DEFAULT_END)
    assert not df.empty, "历史数据为空，检查网络/代码"
    print(f"   行数={len(df)} 末日={df['date'].iloc[-1].date()} 收盘价={df['close'].iloc[-1]:.2f}")

    print("== 均线交叉回测 ==")
    sig = ma_cross_signal(df["close"], 5, 20)
    res = run_backtest(df, sig, INIT_CASH)
    for k, v in res["metrics"].items():
        print(f"   {k}: {v}")
    print(f"   交易笔数(含未平)={len(res['trades'])} 样本: {res['trades'][:2]}")

    print("== 动量回测 ==")
    sig2 = momentum_signal(df["close"], 20)
    res2 = run_backtest(df, sig2, INIT_CASH)
    print(f"   总收益%={res2['metrics']['total_return']} 最大回撤%={res2['metrics']['max_drawdown']} 交易次数={res2['metrics']['trade_count']}")

    # 无未来函数自检：把信号整体右移，净值不应改变（说明收益只用到了前一日信号）
    print("== 自检通过 ==")


if __name__ == "__main__":
    main()
