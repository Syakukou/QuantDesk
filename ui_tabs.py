"""八个页签：策略回测、可转债轮动、多市场净值、全市场选股、事件雷达、机构持仓、
行为审计、组合风险。

放在单独文件里，main.py 只负责挂载，避免那个 2400 行的文件继续膨胀。

界面原则：
  · 所有分析都靠按钮触发，不需要命令行
  · 每个结论旁边必须有数字，不给空话
  · 数据档位（精确/近似）始终显示，不拿近似数据冒充精确
"""
from __future__ import annotations

import os
import traceback

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt6.QtGui import QColor, QBrush, QFont, QDesktopServices
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QTextBrowser, QFileDialog, QMessageBox,
    QAbstractItemView, QGroupBox, QCheckBox, QTabWidget, QDialog, QDialogButtonBox,
)

import audit
import datasource
import data_feed
import engine
import events
import holders
import metrics
import multimkt
import optimizer
import riskx
import rotation
import selection
import trades_io
from config import DEFAULT_START, DEFAULT_END, INIT_CASH, load_watchlist
import tablekit
from tablekit import SortTable, CellItem, DetailDialog, row_fields

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.ticker as mticker

UP = QColor("#d0342c")      # 涨 红
DOWN = QColor("#0f8a4d")    # 跌 绿
MUTED = QColor("#7a7a7a")
FG = QColor("#222222")

# 多曲线配色：红蓝绿橙紫青灰，红绿都留给「涨跌」语义，其余做区分
PALETTE = ["#1f6fd0", "#e08a1e", "#7b4ecb", "#0aa0a0",
           "#c0392b", "#2e8b57", "#8a8a8a", "#b8860b"]


class Worker(QThread):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    def run(self):
        try:
            self.done.emit(self.fn(*self.args))
        except Exception as exc:
            self.fail.emit(f"{exc}\n\n{traceback.format_exc()}")


def _table(headers, stretch=True, height=None, row_h=22, widths=None,
           stretch_col=-1) -> SortTable:
    """可排序表格：点表头排序、列宽可拖、列可换位，数字按数值比。

    widths 给关键列一个像样的初宽（按内容自动算会把长标题列压扁）；
    stretch_col 指定吸收剩余宽度的那一列，通常是文字最长的一列。
    """
    t = SortTable(headers, row_h=row_h,
                  last_stretch=(stretch and stretch_col < 0),
                  widths=widths, stretch_col=stretch_col)
    if height:
        t.setMaximumHeight(height)
    return t


def _put(t, r, c, text, color=None, align_right=False, tooltip=None):
    tablekit.put(t, r, c, text, color=color, align_right=align_right,
                 tooltip=tooltip)


def _money_color(v: float) -> QColor:
    return UP if v > 0 else (DOWN if v < 0 else FG)


def _fnum(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _col_of(v):
    return MUTED if v != v else _money_color(v)


def _n(v, fmt="{:,.0f}", dash="—") -> str:
    """数值格式化：NaN 与 None 一律显示破折号，别让 nan 混进表格。"""
    f = _fnum(v)
    return dash if f != f else fmt.format(f)


def _legacy_view(res: dict, m: dict) -> dict:
    """把新引擎结果改造成净值图控件认识的形状。"""
    out = dict(res)
    out["metrics"] = {"init_cash": m["init_cash"]}
    tr = []
    for t in res["trades"]:
        d = dict(t)
        d["return"] = None if t.get("is_open") else t["ret_pct"]
        d["exit_date"] = "持有中" if t.get("is_open") else t["exit_date"]
        tr.append(d)
    out["trades"] = tr
    return out


# ===================================================================== 回测

class BacktestTab(QWidget):
    def __init__(self, chart, parent=None):
        super().__init__(parent)
        self.chart = chart
        self.items = load_watchlist()
        self.res = None
        self.m = None
        self._worker = None
        self.param_widgets: dict[str, QSpinBox] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(7)

        # ---- 第一行：标的 / 策略 / 区间
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self.symbol = QComboBox()
        self.symbol.addItems([f"{it['name']}({it['code']})" for it in self.items])
        self.symbol.setMinimumWidth(160)
        self.strategy = QComboBox()
        self.strategy.addItems(list(engine.STRATEGIES.keys()))
        self.strategy.setMinimumWidth(120)
        self.strategy.currentTextChanged.connect(lambda *_: self._rebuild_params())
        self.start = QLineEdit(DEFAULT_START)
        self.start.setFixedWidth(92)
        self.end = QLineEdit(DEFAULT_END)
        self.end.setFixedWidth(92)
        self.cash = QDoubleSpinBox()
        self.cash.setRange(10000, 1e9)
        self.cash.setValue(INIT_CASH)
        self.cash.setDecimals(0)
        self.cash.setSingleStep(10000)
        self.cash.setFixedWidth(110)
        self.cash.setPrefix("¥ ")

        for w in (QLabel("标的"), self.symbol, QLabel("策略"), self.strategy,
                  QLabel("起"), self.start, QLabel("止"), self.end,
                  QLabel("初始资金"), self.cash):
            row1.addWidget(w)
        row1.addStretch(1)
        self.run_btn = QPushButton("运行回测")
        self.run_btn.setMinimumWidth(100)
        self.run_btn.clicked.connect(lambda *_: self.run())
        row1.addWidget(self.run_btn)
        root.addLayout(row1)

        # ---- 第二行：参数 + 工具
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        self.param_box = QHBoxLayout()
        self.param_box.setSpacing(4)
        row2.addLayout(self.param_box)
        row2.addSpacing(10)
        self.grid_btn = QPushButton("参数扫描")
        self.grid_btn.setToolTip("遍历参数组合，看收益是否只在一小片参数上成立")
        self.grid_btn.clicked.connect(lambda *_: self.run_grid())
        self.wf_btn = QPushButton("滚动样本外")
        self.wf_btn.setToolTip("把区间切四段分别跑，看收益是否集中在某一段")
        self.wf_btn.clicked.connect(lambda *_: self.run_wf())
        row2.addWidget(self.grid_btn)
        row2.addWidget(self.wf_btn)
        row2.addStretch(1)
        self.ds_label = QLabel(datasource.status_line())
        self.ds_label.setStyleSheet("color:#8a6d3b;")
        row2.addWidget(self.ds_label)
        self.status = QLabel("")
        self.status.setObjectName("status")
        row2.addWidget(self.status)
        root.addLayout(row2)
        self._rebuild_params()

        # ---- 净值图
        root.addWidget(self.chart, 4)

        # ---- 指标（3 组 × 8 行；列宽按内容收紧，末尾空列吸收剩余宽度）
        self.metrics_tb = _table(
            ["指标", "数值", "指标", "数值", "指标", "数值", ""],
            height=8 * 23 + 26)
        self.metrics_tb.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        hd = self.metrics_tb.horizontalHeader()
        for c in range(6):
            hd.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hd.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.metrics_tb, 0)

        # ---- 交易明细
        self.trades_tb = _table(metrics.TRADE_COLUMNS, height=150)
        self.trades_tb.itemSelectionChanged.connect(lambda *_: self._pick_trade())
        self.trades_tb.cellClicked.connect(lambda *_: self._pick_trade())
        self._filling = False
        root.addWidget(self.trades_tb, 2)

        # ---- 结论
        self.verdict = QTextBrowser()
        self.verdict.setMaximumHeight(96)
        self.verdict.setOpenExternalLinks(False)
        root.addWidget(self.verdict, 0)

        QTimer.singleShot(0, lambda: self.chart.show_placeholder("点击「运行回测」生成净值曲线"))

    # ---------------------------------------------------------------- 参数
    _DEFAULTS = {"fast": 5, "slow": 20, "n": 20, "low": 30, "high": 70,
                 "k": 2.0, "exit_n": 10, "signal": 9, "atr_k": 2.5}

    def _rebuild_params(self):
        while self.param_box.count():
            w = self.param_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.param_widgets.clear()
        strat = self.strategy.currentText()
        spec = engine.STRATEGIES.get(strat, {})
        for name, (lo, hi, st) in spec.get("params", {}).items():
            lab = QLabel(name)
            # 步长带小数（布林 k、ATR 倍数）必须用浮点框，整数框会把步长设成 0
            is_float = (isinstance(lo, float) or isinstance(hi, float)
                        or float(st) % 1 != 0)
            if is_float:
                sp = QDoubleSpinBox()
                sp.setDecimals(1)
                sp.setRange(float(lo), float(hi))
                sp.setSingleStep(float(st))
                sp.setValue(float(self._DEFAULTS.get(name, lo)))
            else:
                sp = QSpinBox()
                sp.setRange(int(lo), int(hi))
                sp.setSingleStep(max(int(st), 1))
                sp.setValue(int(self._DEFAULTS.get(name, lo)))
            sp.setFixedWidth(74)
            sp.setToolTip(f"{name} 范围 {lo}–{hi}")
            self.param_widgets[name] = sp
            self.param_box.addWidget(lab)
            self.param_box.addWidget(sp)
        tip = QLabel(spec.get("desc", ""))
        tip.setStyleSheet("color:#888;")
        self.param_box.addWidget(tip)

    def _params(self) -> dict:
        return {k: w.value() for k, w in self.param_widgets.items()}

    def _code(self):
        txt = self.symbol.currentText()
        return txt.split("(")[-1].rstrip(")"), txt.split("(")[0]

    # ---------------------------------------------------------------- 运行
    def run(self):
        code, name = self._code()
        strat = self.strategy.currentText()
        start, end = self.start.text().strip(), self.end.text().strip()
        cash = self.cash.value()
        params = self._params()

        def job():
            df = data_feed.get_history(code, start, end)
            if df is None or df.empty:
                raise RuntimeError(f"{code} 没有取到行情，检查代码或区间")
            sig = engine.make_signal(df, strat, **params)
            res = engine.run(df, sig, cash, code=code, name=name,
                             signal_name=strat)
            m = metrics.summarize(res, cash)
            return df, res, m

        self.status.setText("回测中…")
        self.run_btn.setEnabled(False)
        w = Worker(job)
        w.done.connect(lambda out: self.on_done(*out))
        w.fail.connect(self.on_fail)
        self._worker = w
        w.start()

    def on_fail(self, msg):
        self.run_btn.setEnabled(True)
        self.status.setText("出错")
        QMessageBox.warning(self, "回测失败", msg[:800])

    def on_done(self, df, res, m):
        self.res, self.m = res, m
        self.run_btn.setEnabled(True)
        self.status.setText(
            f"{len(df)} 根 K 线｜{m['trade_count']} 笔｜顺延买 {res['blocked']['buy']} "
            f"卖 {res['blocked']['sell']}")
        try:
            self.chart.set_result(_legacy_view(res, m))
        except Exception:
            traceback.print_exc()
        self._fill_metrics(m)
        self._fill_trades(res["trades"])
        self._fill_verdict(m, res)

    def _fill_metrics(self, m):
        pairs = [
            ("总收益", f"{m['total_return']:+.2f}%", _money_color(m["total_return"])),
            ("年化", f"{m['annual_return']:+.2f}%", _money_color(m["annual_return"])),
            ("年化波动", f"{m['volatility']:.2f}%", None),
            ("最大回撤", f"{m['max_drawdown']:.2f}%", DOWN),
            ("回撤修复", f"{m['dd_recover_days']} 日", None),
            ("最长水下", f"{m['longest_underwater']} 日", None),
            ("夏普", f"{m['sharpe']:.2f}", None),
            ("索提诺", f"{m['sortino']:.2f}", None),
            ("卡玛", f"{m['calmar']:.2f}", None),
            ("交易笔数", f"{m['trade_count']}", None),
            ("胜率", f"{m['win_rate']:.1f}%", None),
            ("盈亏比", f"{m['profit_loss_ratio']:.2f}", None),
            ("盈利因子", f"{m['profit_factor']:.2f}", None),
            ("每笔期望", f"{m['expectancy']:+,.0f}", _money_color(m["expectancy"])),
            ("赢的持股", f"{m['avg_win_hold']:.1f} 日", None),
            ("亏的持股", f"{m['avg_loss_hold']:.1f} 日", None),
            ("年交易数", f"{m['trades_per_year']:.0f}", None),
            ("持仓暴露", f"{m['exposure']:.1f}%", None),
            ("换手", f"{m['turnover']:.1f}x", None),
            ("总费用", f"{m['total_fees']:,.0f}", None),
            ("费用/收益", f"{m['fee_ratio']:.1f}%", None),
            ("最大连亏", f"{m['max_loss_streak']} 笔", None),
            ("最大连盈", f"{m['max_win_streak']} 笔", None),
            ("期末权益", f"{m['final_equity']:,.0f}", None),
        ]
        rows = (len(pairs) + 2) // 3
        self.metrics_tb.setRowCount(rows)
        for i, (k, v, col) in enumerate(pairs):
            r, c = divmod(i, 3)
            _put(self.metrics_tb, r, c * 2, k, MUTED)
            _put(self.metrics_tb, r, c * 2 + 1, v, col)

    def _fill_trades(self, trades):
        self._filling = True
        rows = metrics.trade_table(trades)
        self.trades_tb.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                col = None
                if c in (10, 11):
                    try:
                        col = _money_color(float(str(v).replace(",", "").rstrip("%")))
                    except ValueError:
                        col = None
                _put(self.trades_tb, r, c, v, col, align_right=(2 <= c <= 14))
        self._trade_rows = trades
        self._filling = False

    def _pick_trade(self):
        if self._filling or not self.res:
            return
        sel = self.trades_tb.selectionModel().selectedRows()
        if not sel:
            return
        r = sel[0].row()
        if r >= len(self._trade_rows):
            return
        t = self._trade_rows[r]
        i_in = t.get("idx_in", 0)
        i_out = None if t.get("is_open") else t.get("idx_out")
        tt = dict(t)
        tt["return"] = None if t.get("is_open") else t["ret_pct"]
        tt["exit_date"] = "持有中" if t.get("is_open") else t["exit_date"]
        self.chart.focus_trade(i_in, i_out, tt)

    def _fill_verdict(self, m, res):
        parts = ["<b>结论</b><ul style='margin-left:14px;'>"]
        for k, v in metrics.verdict(m):
            parts.append(f"<li><b>{k}</b>：{v}</li>")
        if res.get("blocked", {}).get("buy"):
            parts.append(f"<li><b>成交受阻</b>：有 {res['blocked']['buy']} 次买入"
                         f"因涨停/停牌未能成交，已按规则顺延</li>")
        if res.get("logs"):
            parts.append(f"<li><b>顺延</b>：{res['logs'][0]}</li>")
        parts.append("</ul>")
        self.verdict.setHtml("".join(parts))

    # ---------------------------------------------------------------- 稳健性
    def run_grid(self):
        code, name = self._code()
        strat = self.strategy.currentText()

        def job():
            df = data_feed.get_history(code, self.start.text().strip(), self.end.text().strip())
            return df, engine.param_grid(df, strat, self.cash.value(), code=code, name=name)

        self.status.setText("参数扫描中…")
        w = Worker(job)
        w.done.connect(self._show_grid)
        w.fail.connect(self.on_fail)
        self._worker = w
        w.start()

    def _show_grid(self, payload):
        df, rows = payload
        self.status.setText(f"扫描 {len(rows)} 组")
        if not rows:
            return
        dlg = QTableWidget(len(rows), 6)
        dlg.setWindowTitle("参数扫描结果（按夏普排序）")
        dlg.setHorizontalHeaderLabels(["参数", "年化%", "最大回撤%", "夏普", "卡玛", "笔数"])
        dlg.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        dlg.resize(720, 460)
        rows = sorted(rows, key=lambda x: -x["sharpe"])
        for i, r in enumerate(rows):
            ps = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            _put(dlg, i, 0, ps)
            _put(dlg, i, 1, f"{r['annual']:.2f}", _money_color(r["annual"]), True)
            _put(dlg, i, 2, f"{r['maxdd']:.2f}", None, True)
            _put(dlg, i, 3, f"{r['sharpe']:.2f}", None, True)
            _put(dlg, i, 4, f"{r['calmar']:.2f}", None, True)
            _put(dlg, i, 5, str(r["trades"]), None, True)
        dlg.show()
        self._grid_dlg = dlg

    def run_wf(self):
        code, name = self._code()
        strat = self.strategy.currentText()
        params = self._params()

        def job():
            df = data_feed.get_history(code, self.start.text().strip(), self.end.text().strip())
            return engine.walk_forward(df, strat, params, 4, self.cash.value(),
                                       code=code, name=name)

        self.status.setText("分段回测中…")
        w = Worker(job)
        w.done.connect(self._show_wf)
        w.fail.connect(self.on_fail)
        self._worker = w
        w.start()

    def _show_wf(self, rows):
        self.status.setText(f"样本外 {len(rows)} 段")
        dlg = QTableWidget(len(rows) + 1, 6)
        dlg.setWindowTitle("滚动样本外：收益是否集中在某一段")
        dlg.setHorizontalHeaderLabels(["区间", "起", "止", "收益%", "最大回撤%", "笔数"])
        dlg.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        dlg.resize(760, 260)
        for i, r in enumerate(rows):
            _put(dlg, i, 0, f"第 {r['fold']} 段")
            _put(dlg, i, 1, r["start"])
            _put(dlg, i, 2, r["end"])
            _put(dlg, i, 3, f"{r['return']:+.2f}", _money_color(r["return"]), True)
            _put(dlg, i, 4, f"{r['maxdd']:.2f}", None, True)
            _put(dlg, i, 5, str(r["trades"]), None, True)
        if rows:
            pos = sum(1 for r in rows if r["return"] > 0)
            _put(dlg, len(rows), 0, f"{pos}/{len(rows)} 段盈利", None)
        dlg.show()
        self._wf_dlg = dlg


# ===================================================================== 审计

class AuditTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recs = []
        self.closed = []
        self.summary = {}
        self.result = None
        self._worker = None
        self._shadow_dlg = None
        self._scope = None                 # (分组名, 该组成交)，空表示看全部

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        bar = QHBoxLayout()
        self.open_btn = QPushButton("导入交割单")
        self.open_btn.clicked.connect(lambda *_: self.on_open())
        self.shadow_btn = QPushButton("跑影子对账")
        self.shadow_btn.setToolTip("保留你的买入，只把卖出换成规则执行，看差额")
        self.shadow_btn.clicked.connect(lambda *_: self.run_shadow())
        self.shadow_btn.setEnabled(False)
        self.file_label = QLabel("未导入")
        self.file_label.setStyleSheet("color:#888;")
        self.score_label = QLabel("")
        bar.addWidget(self.open_btn)
        bar.addWidget(self.shadow_btn)
        bar.addWidget(self.file_label, 1)
        bar.addWidget(self.score_label)
        root.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        # 左：偏误
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("四类行为偏误（严重度按数字证据判定）"))
        self.bias_tb = _table(["偏误", "程度", "结论", "证据"], stretch=True)
        self.bias_tb.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.bias_tb.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        lv.addWidget(self.bias_tb, 3)
        lv.addWidget(QLabel("分组找漏：哪一类交易在亏钱"))
        self.dim = QComboBox()
        self.dim.addItems(["按标的", "按入场星期", "按持有天数", "按入场时段"])
        self.dim.currentTextChanged.connect(lambda *_: self._fill_groups())
        lv.addWidget(self.dim)
        self.group_tb = _table(["分组", "笔数", "净盈亏", "胜率", "平均持有"])
        self.group_tb.setToolTip("点一行：右边的日历只算这一组的平仓；再点一次取消。"
                                 "双击看这一组的每一笔")
        self.group_tb.cellClicked.connect(self._on_group_click)
        self.group_tb.cellDoubleClicked.connect(self._group_detail)
        lv.addWidget(self.group_tb, 4)
        split.addWidget(left)

        # 右：判读 + 日历
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QLabel("综合判读"))
        self.read = QTextBrowser()
        rv.addWidget(self.read, 2)
        calbar = QHBoxLayout()
        self.cal_title = QLabel("P&L 日历（按平仓日，最多 250 天）")
        self.cal_title.setStyleSheet("color:#444;font-weight:600;")
        self.cal_clear = QPushButton("看全部")
        self.cal_clear.setFixedHeight(22)
        self.cal_clear.setToolTip("清掉左边的筛选，回到全部平仓")
        self.cal_clear.clicked.connect(lambda *_: self._clear_scope())
        self.cal_clear.setEnabled(False)
        calbar.addWidget(self.cal_title, 1)
        calbar.addWidget(self.cal_clear)
        rv.addLayout(calbar)
        self.cal_note = QLabel("")
        self.cal_note.setStyleSheet("color:#8b96a3;font-size:11px;")
        self.cal_note.setWordWrap(True)
        rv.addWidget(self.cal_note)
        self.cal_tb = _table(["日期", "当日盈亏", "笔数"],
                             widths={0: 104, 1: 104, 2: 72})
        rv.addWidget(self.cal_tb, 3)
        split.addWidget(right)
        split.setSizes([720, 560])
        root.addWidget(split, 1)

    # ---------------------------------------------------------------- 导入
    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择券商交割单", os.path.expanduser("~"),
            "交割单 (*.xls *.xlsx *.csv *.txt)")
        if not path:
            return
        self.file_label.setText("解析中…")

        def job():
            recs, meta = trades_io.load_statement(path)
            res = trades_io.rebuild(recs)
            prov = datasource.active()
            cache = {}

            def lookup(code):
                if code not in cache:
                    cache[code] = prov.bars(code, "20240101", "20261231")
                return cache[code]

            diag = audit.diagnose(recs, res["closed"], res.get("summary"),
                                  bars_lookup=lookup)
            ov = audit.overall(res["closed"], res.get("summary") or {})
            return path, recs, meta, res, diag, ov

        w = Worker(job)
        w.done.connect(self.on_loaded)
        w.fail.connect(lambda m: QMessageBox.warning(self, "解析失败", m[:800]))
        self._worker = w
        w.start()

    def on_loaded(self, payload):
        path, recs, meta, res, diag, ov = payload
        self.recs = recs
        self.closed = res["closed"]
        self.summary = res.get("summary") or {}
        self.result = diag
        self.ov = ov
        self.shadow_btn.setEnabled(bool(self.closed))
        self.file_label.setText(
            f"{os.path.basename(path)}｜成交 {len(recs)} 条｜清仓 {len(self.closed)} 轮"
            f"｜持仓 {len(res['open'])} 只")
        self.score_label.setText(
            f"<b style='font-size:15px;'>纪律分 {diag['score']}</b>"
            f"<span style='color:#888;'>（仅相对参考）</span>")
        self.score_label.setTextFormat(Qt.TextFormat.RichText)
        self._scope = None                 # (筛选说明, 该组成交)
        self._fill_bias(diag)
        self._fill_groups()
        self._fill_calendar(diag["calendar"])
        self._fill_read(diag, ov)

    def _fill_bias(self, d):
        rows = d["biases"]
        self.bias_tb.setRowCount(len(rows))
        for i, b in enumerate(rows):
            col = {"高": UP, "中": QColor("#b8860b"), "低": DOWN}.get(b["severity"], FG)
            _put(self.bias_tb, i, 0, b["name"])
            _put(self.bias_tb, i, 1, b["severity"], col)
            _put(self.bias_tb, i, 2, b["headline"])
            _put(self.bias_tb, i, 3, b["evidence"])
        self.bias_tb.resizeRowsToContents()

    def _cur_group(self):
        key = {"按标的": "symbol", "按入场星期": "weekday",
               "按持有天数": "hold", "按入场时段": "hour"}[self.dim.currentText()]
        return self.result["groups"].get(key) if self.result else None

    def _fill_groups(self):
        # 换维度就把上一维的筛选清掉，否则日历还挂在旧分组上，看着像没反应
        self._clear_scope(quiet=True)
        if not self.result:
            return
        g = self._cur_group()
        if not g:
            self.group_tb.setRowCount(1)
            _put(self.group_tb, 0, 0, self.result["groups"].get("hour_disabled", "该维度不可用"))
            for c in range(1, 5):
                _put(self.group_tb, 0, c, "")
            return
        rows = g["rows"]
        worst_key = (g.get("worst") or {}).get("key")
        self.group_tb.setRowCount(len(rows))
        for i, r in enumerate(rows):
            _put(self.group_tb, i, 0, r["key"])
            _put(self.group_tb, i, 1, str(r["n"]), None, True)
            _put(self.group_tb, i, 2, f"{r['pnl']:+,.0f}", _money_color(r["pnl"]), True)
            _put(self.group_tb, i, 3, f"{r['win_rate']:.0f}%", None, True)
            _put(self.group_tb, i, 4, f"{r['avg_days']:.0f} 天", None, True)
            # 表能排序，行号会变，所以把组序号挂在格子上
            self.group_tb.bind(i, i)
            # 标出「笔数够、亏得最多」的那一格，避免只做一次的偶然亏损被当成规律
            if worst_key is not None and r["key"] == worst_key:
                for c in range(5):
                    it = self.group_tb.item(i, c)
                    if it:
                        it.setBackground(QBrush(QColor("#fbe4e4")))
                        f = it.font()
                        f.setBold(True)
                        it.setFont(f)

    def _fill_calendar(self, calendar):
        # 一年多的平仓日全留，靠滚动看不靠砍；砍掉会让「合计」对不上
        cal = sorted(calendar, key=lambda x: x["date"], reverse=True)[:250]
        self.cal_tb.setRowCount(len(cal))
        for i, r in enumerate(cal):
            _put(self.cal_tb, i, 0, r["date"])
            _put(self.cal_tb, i, 1, f"{r['pnl']:+,.0f}", _money_color(r["pnl"]), True)
            _put(self.cal_tb, i, 2, str(r["n"]), None, True)

    # ------------------------------------------------------- 分组 → 日历联动
    def _on_group_click(self, row, _col):
        g = self._cur_group()
        if not g or row < 0:
            return
        idx = self.group_tb.ref(row)
        if not isinstance(idx, int) or not (0 <= idx < len(g["rows"])):
            return
        r = g["rows"][idx]
        # 同一个组再点一次就当取消，省得去找复位按钮
        if self._scope is not None and self._scope[0] == r["key"]:
            self._clear_scope()
            return
        trades = r.get("trades") or []
        self._set_scope(r["key"], trades,
                        f"{self.dim.currentText()} · {r['key']}")

    def _set_scope(self, key, trades, label):
        self._scope = (key, trades)
        self._fill_calendar(audit.pnl_calendar(trades))
        wins = [t for t in trades if t["pnl"] > 0]
        net = sum(t["pnl"] for t in trades)
        self.cal_title.setText(f"P&L 日历 · {label}")
        self.cal_note.setText(
            f"{len(trades)} 笔｜净 {net:+,.0f} 元｜胜率 "
            f"{len(wins) / len(trades) * 100:.0f}%" if trades else "该组没有可展示的平仓")
        self.cal_clear.setEnabled(True)
        # 图例上的日历只留给这一组，标题里写清楚是按什么筛的，别让人以为是全量

    def _clear_scope(self, quiet=False):
        self._scope = None
        self.cal_title.setText("P&L 日历（按平仓日，最多 250 天）")
        self.cal_note.setText("")
        self.cal_clear.setEnabled(False)
        self.group_tb.clearSelection()
        if self.result:
            self._fill_calendar(self.result["calendar"])

    def _group_detail(self, row, _col):
        g = self._cur_group()
        if not g or row < 0:
            return
        idx = self.group_tb.ref(row)
        if not isinstance(idx, int) or not (0 <= idx < len(g["rows"])):
            return
        r = g["rows"][idx]
        trades = sorted(r.get("trades") or [], key=lambda t: t["exit_date"], reverse=True)
        if not trades:
            return
        rows = []
        for t in trades:
            rows.append([
                t["code"], t["name"], t["entry_date"], t["exit_date"],
                f"{t['ret']:+.2f}", f"{t['pnl']:+,.0f}", str(t["days"]),
            ])
        wins = [t for t in trades if t["pnl"] > 0]
        DetailDialog(
            self, title=f"{self.dim.currentText()} · {r['key']}",
            subtitle=f"{r['n']} 笔｜净 {r['pnl']:+,.0f} 元｜胜率 {r['win_rate']:.0f}%"
                     f"｜平均持有 {r['avg_days']:.0f} 天",
            tables=[{"caption": "这一组的每一笔（按平仓日倒序）",
                     "headers": ["代码", "名称", "买入", "卖出", "收益%", "盈亏元", "天数"],
                     "rows": rows, "right": [4, 5, 6]}]).exec()

    def _fill_read(self, d, ov):
        parts = []
        for o in ov:
            parts.append(f"<p><b>{o['title']}</b><br>{o['text']}</p>")
        for b in d["biases"]:
            if b["severity"] in ("高", "中"):
                parts.append(f"<p style='color:#8a3b3b;'><b>{b['name']} · "
                             f"{b['severity']}</b><br>{b['advice']}</p>")
        parts.append(f"<p style='color:#888;font-size:11px;'>{d['note']}</p>")
        self.read.setHtml("".join(parts))

    # ---------------------------------------------------------------- 影子
    def run_shadow(self):
        if not self.closed:
            return
        rules = audit.extract_rules(self.closed)
        closed = self.closed
        codes = list({t["code"] for t in closed})

        def job():
            cache = {}

            def lookup(code):
                if code not in cache:
                    cache[code] = datasource.active().bars(code, "20250101", "20261231")
                return cache[code]

            return audit.shadow_replay(closed, lookup, rules)

        self.shadow_btn.setEnabled(False)
        self.file_label.setText(self.file_label.text() + "｜影子对账中…")
        w = Worker(job)
        w.done.connect(self._show_shadow)
        w.fail.connect(lambda m: QMessageBox.warning(self, "影子对账失败", m[:600]))
        self._worker = w
        w.start()

    def _show_shadow(self, sp):
        self.shadow_btn.setEnabled(True)
        self.file_label.setText(self.file_label.text().replace("｜影子对账中…", ""))
        if not sp.get("ok"):
            QMessageBox.information(self, "影子对账", sp.get("msg", "无法完成"))
            return
        dlg = QWidget()
        dlg.setWindowTitle("影子对账：你的卖出纪律值多少钱")
        dlg.resize(900, 560)
        v = QVBoxLayout(dlg)
        head = QLabel(sp["msg"])
        head.setWordWrap(True)
        head.setStyleSheet("font-size:13px;padding:6px;background:#f6f3ea;")
        v.addWidget(head)
        tb = _table(["标的", "买入日", "实际盈亏", "规则基准", "买入不动", "规则差额", "退出方式"])
        tb.setRowCount(len(sp["rows"]))
        for i, r in enumerate(sp["rows"]):
            _put(tb, i, 0, r["name"])
            _put(tb, i, 1, r["entry_date"])
            _put(tb, i, 2, f"{r['actual']:+,.0f}", _money_color(r["actual"]), True)
            _put(tb, i, 3, f"{r['shadow']:+,.0f}", _money_color(r["shadow"]), True)
            _put(tb, i, 4, f"{r['hold']:+,.0f}", _money_color(r["hold"]), True)
            _put(tb, i, 5, f"{r['diff']:+,.0f}", _money_color(r["diff"]), True)
            _put(tb, i, 6, r["reason"])
        v.addWidget(tb, 1)
        note = QLabel(
            f"规则来自你自己的成交中位数：止盈 {sp['rules']['take_profit']}%、"
            f"止损 {sp['rules']['stop_loss']}%、最长持 {sp['rules']['max_hold']} 天。"
            f"覆盖 {sp['used']} 轮。买入不动 = 同一天买入后完全不操作，持满 "
            f"{sp['rules']['max_hold']} 天。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#777;font-size:11px;")
        v.addWidget(note)
        dlg.show()
        self._shadow_dlg = dlg


# ===================================================================== 风险

class RiskTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self.holdings = []
        self.result = None
        self._ind_scope = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        bar = QHBoxLayout()
        self.import_btn = QPushButton("从交割单读持仓")
        self.import_btn.clicked.connect(lambda *_: self.on_import())
        self.demo_btn = QPushButton("用关注池做等权组合")
        self.demo_btn.clicked.connect(lambda *_: self.use_watchlist())
        self.info = QLabel("未加载持仓")
        self.info.setStyleSheet("color:#888;")
        bar.addWidget(self.import_btn)
        bar.addWidget(self.demo_btn)
        bar.addWidget(self.info, 1)
        root.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        hbar = QHBoxLayout()
        self.hold_title = QLabel("持仓权重与风险贡献（风险贡献 ≠ 仓位，这是关键）")
        self.hold_title.setStyleSheet("color:#444;font-weight:600;")
        self.hold_clear = QPushButton("显示全部")
        self.hold_clear.setFixedHeight(22)
        self.hold_clear.clicked.connect(lambda *_: self._clear_ind())
        self.hold_clear.setEnabled(False)
        hbar.addWidget(self.hold_title, 1)
        hbar.addWidget(self.hold_clear)
        lv.addLayout(hbar)
        self.hold_tb = _table(["标的", "权重", "风险贡献", "行业", "盈亏"])
        self.hold_tb.setToolTip("点一行：下面只留这个行业的同行持仓")
        lv.addWidget(self.hold_tb, 3)
        self.ind_lb = QLabel("行业暴露（点一行，只看这个行业的持仓）")
        self.ind_lb.setStyleSheet("color:#444;font-weight:600;")
        lv.addWidget(self.ind_lb)
        self.ind_tb = _table(["行业", "权重", "市值"], height=150)
        self.ind_tb.cellClicked.connect(self._on_ind_click)
        lv.addWidget(self.ind_tb, 2)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        rtabs = QTabWidget()
        # --- 页一：风险度量
        p1 = QWidget()
        v1 = QVBoxLayout(p1)
        v1.setContentsMargins(0, 4, 0, 0)
        v1.addWidget(QLabel("风险度量"))
        self.var_tb = _table(["指标", "数值"], stretch=True, height=130)
        v1.addWidget(self.var_tb, 0)
        v1.addWidget(QLabel("同涨同跌组（相关性聚类，双击看成员与权重）"))
        self.cluster_tb = _table(["合计权重", "平均相关", "成员"], height=120)
        self.cluster_tb.cellDoubleClicked.connect(self._cluster_detail)
        v1.addWidget(self.cluster_tb, 1)
        rtabs.addTab(p1, "风险度量")

        # --- 页二：组合优化
        p2 = QWidget()
        v2 = QVBoxLayout(p2)
        v2.setContentsMargins(0, 4, 0, 0)
        row = QHBoxLayout()
        self.opt_method = QComboBox()
        self.opt_method.addItems(list(optimizer.METHODS.keys()))
        self.opt_method.setCurrentText("风险平价")
        self.opt_btn = QPushButton("跑组合优化")
        self.opt_btn.clicked.connect(lambda *_: self._optimize())
        row.addWidget(QLabel("方法"))
        row.addWidget(self.opt_method)
        row.addWidget(self.opt_btn)
        row.addStretch(1)
        v2.addLayout(row)
        self.opt_info = QLabel("未运行。先在上面读入持仓，再点这里")
        self.opt_info.setStyleSheet("color:#888;")
        v2.addWidget(self.opt_info)
        v2.addWidget(QLabel("各方案对比（收益数字用历史均值外推，别当真，看波动和有效只数）"))
        self.opt_tb = _table(["方案", "年化收益%", "年化波动%", "夏普", "有效只数", "单票最大%"])
        v2.addWidget(self.opt_tb, 2)
        v2.addWidget(QLabel("调仓建议（按建议权重排）"))
        self.adj_tb = _table(["代码", "名称", "当前%", "建议%", "调整pp", "风险贡献%"])
        v2.addWidget(self.adj_tb, 3)
        rtabs.addTab(p2, "优化建议")

        rv.addWidget(rtabs, 3)
        rv.addWidget(QLabel("提醒"))
        self.warn = QTextBrowser()
        rv.addWidget(self.warn, 2)
        split.addWidget(right)
        split.setSizes([700, 580])
        root.addWidget(split, 1)

    def on_import(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择券商交割单", os.path.expanduser("~"),
            "交割单 (*.xls *.xlsx *.csv *.txt)")
        if not path:
            return
        self.info.setText("解析中…")

        def job():
            recs, _ = trades_io.load_statement(path)
            res = trades_io.rebuild(recs)
            return res["open"]

        w = Worker(job)
        w.done.connect(self.on_holdings)
        w.fail.connect(lambda m: QMessageBox.warning(self, "读取失败", m[:600]))
        self._worker = w
        w.start()

    def use_watchlist(self):
        items = load_watchlist()[:12]
        self.holdings = [{"code": it["code"], "name": it["name"], "qty": 1000,
                          "cost_price": 0} for it in items]
        self._analyze()

    def on_holdings(self, open_pos):
        if not open_pos:
            QMessageBox.information(self, "组合风险", "这份交割单里没有未平仓持仓")
            return
        self.holdings = [{"code": o["code"], "name": o["name"], "qty": o["qty"],
                          "cost_price": o["cost_price"]} for o in open_pos]
        self._analyze()

    def _analyze(self):
        hold = list(self.holdings)
        self.info.setText(f"{len(hold)} 只持仓，取行情中…")

        def job():
            prov = datasource.active()
            codes = [h["code"] for h in hold]
            ind = prov.industry_map(codes)
            cache = {}

            def lookup(code):
                if code not in cache:
                    cache[code] = prov.bars(code, "20250101", "20261231")
                return cache[code]

            return riskx.analyze(hold, lookup, lambda c: ind.get(c, ""), lookback=250)

        w = Worker(job)
        w.done.connect(self.on_result)
        w.fail.connect(lambda m: QMessageBox.warning(self, "分析失败", m[:600]))
        self._worker = w
        w.start()

    def on_result(self, r):
        if not r.get("ok"):
            self.info.setText(r.get("msg", "失败"))
            return
        self.result = r
        self.info.setText(
            f"{r['n']} 只｜总市值 {r['total_mv']:,.0f}｜HHI {r['hhi']}｜"
            f"有效分散度 {r['eff_n']} 只｜有行情 {r['n_with_bars']} 只")

        self._ind_scope = None
        self.hold_clear.setEnabled(False)
        self.hold_title.setText("持仓权重与风险贡献（风险贡献 ≠ 仓位，这是关键）")
        self._fill_hold(r["rows"])

        ind = r["industry"]
        self.ind_tb.setRowCount(len(ind))
        for i, x in enumerate(ind):
            _put(self.ind_tb, i, 0, x["name"] or "未分类")
            _put(self.ind_tb, i, 1, f"{x['weight']:.1f}%", None, True)
            _put(self.ind_tb, i, 2, f"{x['mv']:,.0f}", None, True)
            self.ind_tb.bind(i, x["name"] or "未分类")

        v = r.get("var") or {}
        items = [("年化波动", f"{v.get('ann_vol', 0):.2f}%"),
                 ("VaR 95%（单日）", f"{v.get('var95', 0):.2f}%"),
                 ("VaR 99%（单日）", f"{v.get('var99', 0):.2f}%"),
                 ("ES 95%（尾部均值）", f"{v.get('es95', 0):.2f}%"),
                 ("VaR95 金额", f"{v.get('var95_amt', 0):,.0f}"),
                 ("样本天数", str(v.get("days", 0)))]
        self.var_tb.setRowCount(len(items))
        for i, (k, val) in enumerate(items):
            _put(self.var_tb, i, 0, k, MUTED)
            _put(self.var_tb, i, 1, val, None, True)

        cl = r["clusters"]
        self.cluster_tb.setRowCount(max(len(cl), 1))
        if cl:
            for i, c in enumerate(cl):
                _put(self.cluster_tb, i, 0, f"{c['weight']:.1f}%", None, True)
                _put(self.cluster_tb, i, 1, f"{c['avg_corr']:.2f}", None, True)
                _put(self.cluster_tb, i, 2, "、".join(m["name"] for m in c["members"]))
        else:
            _put(self.cluster_tb, 0, 0, "—")
            _put(self.cluster_tb, 0, 1, "—")
            _put(self.cluster_tb, 0, 2, "没有相关性 0.6 以上的持仓配对")

        html = ["<ul style='margin-left:14px;'>"]
        for w_ in r["warns"]:
            html.append(f"<li>{w_}</li>")
        html.append("</ul>")
        if v.get("worst_days"):
            html.append("<p style='color:#888;font-size:11px;'>历史最差 5 天：" +
                        "、".join(f"{d['date']} {d['ret']}%" for d in v["worst_days"]) +
                        "</p>")
        html.append("<p style='color:#888;font-size:11px;'>VaR 用历史模拟法，"
                    "只描述过去 250 个交易日的分布，不代表未来上限。</p>")
        self.warn.setHtml("".join(html))

    def _fill_hold(self, rows):
        self.hold_tb.setRowCount(len(rows))
        for i, x in enumerate(rows):
            _put(self.hold_tb, i, 0, f"{x['name']} {x['code']}")
            _put(self.hold_tb, i, 1, f"{x['weight']:.1f}%", None, True)
            _put(self.hold_tb, i, 2, f"{x.get('risk_contrib', 0):.1f}%",
                 UP if x.get("risk_contrib", 0) > x["weight"] * 1.5 else None, True)
            _put(self.hold_tb, i, 3, x["industry"])
            _put(self.hold_tb, i, 4, f"{x['pnl']:+,.0f}", _money_color(x["pnl"]), True)
            self.hold_tb.bind(i, x["code"])

    def _on_ind_click(self, row, _col):
        if not self.result or row < 0:
            return
        name = self.ind_tb.ref(row)
        if name is None:
            it = self.ind_tb.item(row, 0)
            name = it.text() if it else ""
        if getattr(self, "_ind_scope", None) == name:   # 再点一次取消
            self._clear_ind()
            return
        sub = [x for x in self.result["rows"]
               if (x["industry"] or "未分类") == name]
        if not sub:
            return
        self._ind_scope = name
        self._fill_hold(sub)
        wsum = sum(x["weight"] for x in sub)
        pnl = sum(x["pnl"] for x in sub)
        self.hold_title.setText(
            f"只看「{name}」：{len(sub)} 只，权重合计 {wsum:.1f}%，"
            f"盈亏合计 {pnl:+,.0f} 元")
        self.hold_clear.setEnabled(True)

    def _clear_ind(self):
        self._ind_scope = None
        self.hold_clear.setEnabled(False)
        self.hold_title.setText("持仓权重与风险贡献（风险贡献 ≠ 仓位，这是关键）")
        self.ind_tb.clearSelection()
        if self.result:
            self._fill_hold(self.result["rows"])

    def _cluster_detail(self, row, _col):
        if not self.result:
            return
        cl = self.result["clusters"]
        if row < 0 or row >= len(cl):
            return
        c = cl[row]
        wmap = {x["code"]: x for x in self.result["rows"]}
        rows = []
        for m in c["members"]:
            x = wmap.get(m["code"], {})
            rows.append([m["code"], m["name"],
                         f"{x.get('weight', 0):.1f}",
                         f"{x.get('risk_contrib', 0):.1f}"])
        DetailDialog(
            self, title=f"同涨同跌组：合计权重 {c['weight']:.1f}%",
            subtitle=f"组内平均相关 {c['avg_corr']:.2f}，{len(c['members'])} 只。"
                     "相关性高的一堆票，分散的是名字，不是风险",
            tables=[{"caption": "成员", "headers": ["代码", "名称", "权重%", "风险贡献%"],
                     "rows": rows, "right": [2, 3]}]).exec()

    # ------------------------------------------------------------ 组合优化
    def _optimize(self):
        if not self.holdings:
            QMessageBox.information(self, "组合优化", "先读入持仓，再点这里")
            return
        method = self.opt_method.currentText()
        hold = [{"code": h["code"], "name": h["name"],
                 "mv": float(h.get("qty", 0)) * float(h.get("cost_price") or 0)}
                for h in self.holdings]
        self.opt_btn.setEnabled(False)
        self.opt_info.setText("取行情并求解中（每只票一次请求）…")

        def job():
            prov = datasource.active()
            cache = {}

            def lookup(code, start, end):
                k = (code, start, end)
                if k not in cache:
                    cache[k] = prov.bars(code, start, end)
                return cache[k]

            return optimizer.optimize(hold, lookup, method=method, start="20250101")

        w = Worker(job)
        w.done.connect(self._on_opt)
        w.fail.connect(self._on_opt_fail)
        self._worker = w
        w.start()

    def _on_opt(self, r):
        self.opt_btn.setEnabled(True)
        if not r.get("ok"):
            self.opt_info.setText("无法优化：" + r.get("msg", ""))
            return

        self.opt_tb.setRowCount(len(r["rows"]))
        for i, row in enumerate(r["rows"]):
            cur = row["方案"] == "当前持仓"
            _put(self.opt_tb, i, 0, row["方案"] + ("（基准）" if cur else ""))
            _put(self.opt_tb, i, 1, f"{row['ret']:.2f}",
                 UP if row["ret"] > 0 else DOWN, True)
            _put(self.opt_tb, i, 2, f"{row['vol']:.2f}", None, True)
            _put(self.opt_tb, i, 3, f"{row['sharpe']:.2f}",
                 UP if row["sharpe"] > 0 else DOWN, True)
            _put(self.opt_tb, i, 4, f"{row['eff_n']:.1f}", None, True)
            _put(self.opt_tb, i, 5, f"{row['top_w']:.1f}", None, True)

        d = r["detail"]
        self.adj_tb.setRowCount(len(d))
        for i, (_, x) in enumerate(d.iterrows()):
            _put(self.adj_tb, i, 0, x["code"], None, True)
            _put(self.adj_tb, i, 1, x["name"])
            _put(self.adj_tb, i, 2, f"{x['当前%']:.1f}", None, True)
            _put(self.adj_tb, i, 3, f"{x['建议%']:.1f}", None, True)
            adj = x["调整pp"]
            _put(self.adj_tb, i, 4, f"{adj:+.1f}",
                 None if abs(adj) < 2 else (UP if adj > 0 else DOWN), True)
            _put(self.adj_tb, i, 5, f"{x['风险贡献%']:.1f}", None, True)

        txt = f"{r['method']}｜可用 {r['n']} 只｜单边换手 {r['turnover']}%"
        if r.get("wild"):
            txt += f"｜已剔除极端波动 {len(r['wild'])} 只（{'、'.join(r['wild'][:4])}）"
        self.opt_info.setText(txt)

        html = ["<ul style='margin-left:14px;'>"]
        for v in r["verdict"]:
            html.append(f"<li><b>{v['title']}</b><br>{v['text']}</li>")
        html.append("</ul>")
        html.append("<p style='color:#888;font-size:11px;'>优化器看的是协方差结构，"
                    "不是收益预测。目标权重是长期配置参考，不是明天就该调到位。</p>")
        self.warn.setHtml("".join(html))

    def _on_opt_fail(self, msg):
        self.opt_btn.setEnabled(True)
        self.opt_info.setText("优化失败")
        QMessageBox.warning(self, "组合优化失败", msg[:600])
        print(msg)


# ====================================================================== 选股

class SelectTab(QWidget):
    """全市场截面选股：因子打分排行 + 因子有效性检验。

    两步是分开的，因为耗时差两个数量级：
    - 选股排行用当日快照，秒级
    - 因子检验要抽 100 只以上拉两年日线，分钟级
    """

    COLS = ["排名", "代码", "名称", "得分", "现价", "涨跌%",
            "60日%", "换手%", "市值亿", "10万可买"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self.snap = None
        self.scored = None
        self.scan = None
        self.cal_w = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        bar = QHBoxLayout()
        self.btn_score = QPushButton("全市场选股排行")
        self.btn_score.clicked.connect(lambda *_: self.run_score())
        self.btn_scan = QPushButton("跑因子有效性检验")
        self.btn_scan.clicked.connect(lambda *_: self.run_scan())
        self.btn_refresh = QPushButton("强制刷新快照")
        self.btn_refresh.clicked.connect(lambda *_: self.run_score(force=True))

        self.topn = QSpinBox()
        self.topn.setRange(10, 500)
        self.topn.setValue(50)
        self.topn.setPrefix("前 ")
        self.topn.setSuffix(" 名")
        self.topn.setFixedWidth(96)

        self.nsample = QSpinBox()
        self.nsample.setRange(40, 400)
        self.nsample.setValue(140)
        self.nsample.setPrefix("样本 ")
        self.nsample.setSuffix(" 只")
        self.nsample.setFixedWidth(100)
        self.nsample.setToolTip("检验用的随机样本数。样本太少 IC 不稳，太多拉数据久")

        self.use_cal = QCheckBox("用检验结果校准权重")
        self.use_cal.setEnabled(False)
        self.use_cal.setToolTip("先跑一次因子检验，才能启用")

        for w in (self.btn_score, self.btn_scan, self.btn_refresh, self.topn,
                  self.nsample, self.use_cal):
            bar.addWidget(w)
        self.status = QLabel("就绪。点左边按钮开始，首次拉快照约 1 分钟")
        self.status.setStyleSheet("color:#888;")
        bar.addWidget(self.status, 1)
        root.addLayout(bar)

        vsplit = QSplitter(Qt.Orientation.Vertical)
        self.tb = _table(self.COLS)
        self.tb.itemSelectionChanged.connect(lambda: self._show_explain())
        vsplit.addWidget(self.tb)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("因子有效性检验（无偏随机样本，T+1 开盘买入口径）"))
        self.scan_tb = _table(["因子", "IC 均值", "ICIR", "t 值", "胜率%", "结论"])
        lv.addWidget(self.scan_tb, 1)
        bottom.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("这一票凭什么排在这个位置"))
        self.btn_f = QPushButton("看财务质量")
        self.btn_f.setFixedWidth(112)
        self.btn_f.setToolTip("Piotroski F-score：按 9 项财务改善信号打分")
        self.btn_f.clicked.connect(lambda *_: self._fscore())
        hdr.addWidget(self.btn_f)
        rv.addLayout(hdr)
        self.explain = QTextBrowser()
        self.explain.setHtml("<p style='color:#888'>选中上方任意一行查看得分归因</p>")
        rv.addWidget(self.explain, 1)
        bottom.addWidget(right)
        bottom.setSizes([520, 420])
        vsplit.addWidget(bottom)
        vsplit.setSizes([430, 260])
        root.addWidget(vsplit, 1)

    # ------------------------------------------------------------ 选股
    def run_score(self, force: bool = False):
        if self._worker and self._worker.isRunning():
            return
        self.btn_score.setEnabled(False)
        self.status.setText("正在拉全市场快照（约 60 秒），请稍候…")
        self._worker = Worker(self._job_score, force)
        self._worker.done.connect(self._on_score)
        self._worker.fail.connect(self._on_fail)
        self._worker.start()

    def _job_score(self, force):
        snap = selection.snapshot(force=force)
        p = selection.build_panel(snap)
        w = self.cal_w if self.use_cal.isChecked() else None
        s = selection.score(p, weights=w, top_n=self.topn.value())
        return snap, p, s

    def _on_score(self, payload):
        snap, p, s = payload
        self.snap, self.scored = snap, s
        self.btn_score.setEnabled(True)

        rows = s.head(self.topn.value())
        self.tb.setRowCount(len(rows))
        for i, (_, r) in enumerate(rows.iterrows()):
            _put(self.tb, i, 0, int(r["rank"]), None, True)
            _put(self.tb, i, 1, r["code"], None, True)
            _put(self.tb, i, 2, r["name"])
            _put(self.tb, i, 3, f"{r['factor_score']:+.2f}", None, True)
            _put(self.tb, i, 4, f"{r['price']:.2f}" if pd.notna(r["price"]) else "—", None, True)
            c = r["chg"] if pd.notna(r["chg"]) else 0
            _put(self.tb, i, 5, f"{c:+.2f}", UP if c > 0 else DOWN if c < 0 else None, True)
            m60 = r["mom60"] if pd.notna(r["mom60"]) else 0
            _put(self.tb, i, 6, f"{m60:+.1f}", UP if m60 > 0 else DOWN if m60 < 0 else None, True)
            _put(self.tb, i, 7, f"{r['turnover']:.2f}" if pd.notna(r["turnover"]) else "—", None, True)
            _put(self.tb, i, 8, f"{r['mv'] / 1e8:.0f}" if pd.notna(r["mv"]) else "—", None, True)
            _put(self.tb, i, 9, int(r["lot_100k"]), None, True)

        mode = "检验校准权重" if self.use_cal.isChecked() else "默认权重"
        self.status.setText(
            f"快照 {len(snap)} 只 → 过滤后 {len(p)} 只 → 打分完成（{mode}），"
            f"显示前 {len(rows)} 名")

    # ------------------------------------------------------------ 检验
    def run_scan(self):
        if self._worker and self._worker.isRunning():
            return
        n = self.nsample.value()
        self.btn_scan.setEnabled(False)
        self.status.setText(f"正在抽 {n} 只票拉两年日线做检验，约 2-3 分钟…")
        self._worker = Worker(self._job_scan, n)
        self._worker.done.connect(self._on_scan)
        self._worker.fail.connect(self._on_fail)
        self._worker.start()

    def _job_scan(self, n):
        snap = self.snap if self.snap is not None else selection.snapshot()
        p = selection.build_panel(snap)
        rng = np.random.default_rng(20261107)
        pool = p["code"].tolist()
        sample = list(rng.choice(pool, size=min(n, len(pool)), replace=False))
        prov = datasource.active()
        end = DEFAULT_END or pd.Timestamp.now().strftime("%Y%m%d")
        bars = {}
        for c in sample:
            try:
                df = prov.bars(c, "20240101", end)
                if df is not None and len(df) > 200:
                    bars[c] = df
            except Exception:  # noqa: BLE001
                continue
        scan = selection.factor_scan(bars, hold=5)
        w, det = selection.calibrate(scan)
        return scan, w, det, len(bars)

    def _on_scan(self, payload):
        scan, w, det, nbars = payload
        self.scan, self.cal_w = scan, w
        self.btn_scan.setEnabled(True)

        if scan is None or scan.empty:
            self.status.setText("检验失败：样本不足")
            return

        self.scan_tb.setRowCount(len(scan))
        for i, (_, r) in enumerate(scan.iterrows()):
            _put(self.scan_tb, i, 0, r["因子"])
            _put(self.scan_tb, i, 1, f"{r['IC均值']:.4f}" if pd.notna(r["IC均值"]) else "—", None, True)
            _put(self.scan_tb, i, 2, f"{r['ICIR']:.3f}" if pd.notna(r["ICIR"]) else "—", None, True)
            t = r["t值"] if pd.notna(r["t值"]) else 0
            _put(self.scan_tb, i, 3, f"{t:+.2f}", UP if t > 0 else DOWN if t < 0 else None, True)
            _put(self.scan_tb, i, 4, f"{r['胜率%']:.1f}" if pd.notna(r["胜率%"]) else "—", None, True)
            _put(self.scan_tb, i, 5, r["结论"])

        ok = bool(w)
        self.use_cal.setEnabled(ok)
        if ok:
            names = "、".join(f"{d['对应字段']}={d['权重']:+.2f}" for d in det)
            self.status.setText(
                f"检验完成：{nbars} 只有效样本，已生成校准权重（{names}）。"
                "勾选后重跑选股即生效")
        else:
            self.status.setText(f"检验完成：{nbars} 只有效样本，但没有任何因子通过 t>2")

    def _show_explain(self):
        if self.scored is None:
            return
        sel = self.tb.selectedIndexes()
        if not sel:
            return
        i = sel[0].row()
        code = self.tb.item(i, 1).text()
        row = self.scored[self.scored["code"] == code]
        if row.empty:
            return
        r = row.iloc[0]
        w = self.cal_w if self.use_cal.isChecked() else None
        parts = [f"<h3 style='margin:4px 0'>{r['name']}　{r['code']}</h3>",
                 f"<p>综合得分 <b>{r['factor_score']:+.2f}</b>，"
                 f"全市场第 {int(r['rank'])} 名（前 {r['pct']:.2f}%）</p>", "<ul>"]
        for why, sign, c in selection.explain(r, weights=w):
            color = "#d0342c" if c > 0 else "#0f8a4d"
            parts.append(f"<li style='color:{color}'>{sign} {c:+.2f}　{why}</li>")
        parts.append("</ul>")
        parts.append("<p style='color:#888;font-size:11px'>得分是相对排名，不是收益预测。"
                     "反转类因子会把跌过头的票排前面，接飞刀的风险要自己看基本面。</p>")
        self.explain.setHtml("".join(parts))

    def _on_fail(self, msg):
        self.btn_score.setEnabled(True)
        self.btn_scan.setEnabled(True)
        self.status.setText("出错了：" + msg.splitlines()[0])
        print(msg)

    # ------------------------------------------------------------ 财务质量
    def _fscore(self):
        sel = self.tb.selectedIndexes()
        if not sel:
            self.status.setText("先在上面的表里选一行")
            return
        code = self.tb.item(sel[0].row(), 1).text()
        self.btn_f.setEnabled(False)
        self.status.setText(f"正在取 {code} 的财务数据…")
        w = Worker(events.f_score, code)
        w.done.connect(self._on_fscore)
        w.fail.connect(lambda m: (self.btn_f.setEnabled(True),
                                  self.status.setText("取财务数据失败")))
        self._worker_f = w
        w.start()

    def _on_fscore(self, r):
        self.btn_f.setEnabled(True)
        if not r.get("ok"):
            self.status.setText(r.get("msg", "取数失败"))
            return
        parts = [f"<h3 style='margin:4px 0'>Piotroski F-score "
                 f"{r['score']}/{r['max']}</h3>",
                 f"<p>报告期 {r['period']}　{r['verdict']}</p>", "<ul>"]
        for name, got, ev in r["items"]:
            color = "#d0342c" if got else "#0f8a4d"
            mark = "✓" if got else "✗"
            parts.append(f"<li style='color:{color}'>{mark} {name}　"
                         f"<span style='color:#666'>{ev}</span></li>")
        parts.append("</ul>")
        parts.append("<p style='color:#888;font-size:11px'>"
                     "9 项里「未增发新股」用股本变化近似，拆股会误判；"
                     "缺数据的项按 0 分计，不猜。这是财务质量的粗筛，不是估值判断。</p>")
        self.explain.setHtml("".join(parts))
        self.status.setText(f"{code} 财务质量 {r['score']}/{r['max']}")


# ====================================================================== 事件

def _pane(caption: str, tb: QTableWidget) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(3)
    lb = QLabel(caption)
    lb.setStyleSheet("color:#444;font-weight:600;")
    lay.addWidget(lb)
    lay.addWidget(tb, 1)
    return w


def _bind(t: QTableWidget, r: int, obj):
    """把行对应的原始索引挂到第一列，排序后仍然找得到那一行。"""
    it = t.item(r, 0)
    if it is not None:
        it.setData(Qt.ItemDataRole.UserRole, obj)


def _ref(t: QTableWidget, r: int):
    it = t.item(r, 0)
    return None if it is None else it.data(Qt.ItemDataRole.UserRole)


class NoticeDialog(DetailDialog):
    """公告详情：字段 + 正文全文。正文走接口异步拉，不卡界面。"""

    def __init__(self, parent, row):
        link = str(row.get("网址", "") or "")
        self._title = str(row["公告标题"])
        self._link = link
        super().__init__(
            parent, title=self._title,
            subtitle=f"{row['名称']}（{row['代码']}）",
            fields=[
                ("代码", row["代码"]), ("名称", row["名称"]),
                ("公告类型", row.get("公告类型", "")),
                ("标签", row["标签"]),
                ("方向", {"risk": "风险", "chance": "机会"}.get(row["方向"], "中性")),
                ("强度", int(row["强度"])),
                ("公告日期", str(row.get("公告日期", ""))[:10]),
                ("原文", link or "无"),
            ],
            link=link or None,
            copy_text=f"{row['名称']}({row['代码']}) {self._title} {link}",
            show_body=bool(link))
        if link and self.body_box is not None:
            self._w = Worker(events.notice_text, link)
            self._w.done.connect(self._on_body)
            self._w.fail.connect(self._on_body_fail)
            self._w.start()

    def _on_body(self, res):
        if res.get("ok"):
            self.set_body(res["text"])
            self._copy = (f"{self._title}\n\n{res['text']}\n\n{self._link}")
        else:
            msg = res.get("msg", "这条公告只有 PDF 附件，没有可解析的正文")
            self.set_body(msg + "\n\n点右下角「在浏览器打开原文」查看完整公告。",
                          note="正文没取到")
            if res.get("pdf") and not self._link:
                self._link = res["pdf"]

    def _on_body_fail(self, msg):
        self.set_body("拉正文失败：" + msg.splitlines()[0], note="网络或接口异常")


_SEAT_COLS = ["#", "营业部 / 机构", "性质", "买入万", "卖出万", "净额万", "买占%", "卖占%"]
_SEAT_W = {0: 40, 2: 76, 3: 80, 4: 80, 5: 82, 6: 64, 7: 64}


class SeatDialog(QDialog):
    """龙虎榜当日买卖席位。机构专用、北向、散户大营分开打标。

    看这个不是为了跟单，是为了分清当天的钱是谁的：机构买、北向买、
    还是拉萨天团对倒。同一只票同样上榜，这三种钱的后五日表现差很远。
    """

    def __init__(self, parent, row):
        super().__init__(parent)
        code = str(row.get("代码", "")).zfill(6)
        name = str(row.get("名称", ""))
        day = str(row.get("上榜日", ""))[:10]
        self.setWindowTitle(f"{name} {code} {day} 席位")
        self.resize(1040, 660)

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)

        head = QLabel(f"{name}（{code}）　{day} 龙虎榜席位")
        head.setStyleSheet("font-size:15px;font-weight:600;color:#141414;")
        v.addWidget(head)

        def num(k, fmt):
            try:
                v = float(row.get(k))
            except (TypeError, ValueError):
                return "—"
            if pd.isna(v):
                return "—"
            return fmt.format(v)

        smry = QLabel(
            f"收盘 {num('收盘价', '{:.2f}')}　涨跌 {num('涨跌幅', '{:+.2f}')}%　"
            f"换手 {num('换手率', '{:.2f}')}%　"
            f"净买 {num('龙虎榜净买额', '{:,.0f}')} 元　"
            f"买 {num('龙虎榜买入额', '{:,.0f}')} / 卖 {num('龙虎榜卖出额', '{:,.0f}')}　"
            f"净买占成交 {num('净买额占总成交比', '{:+.2f}')}%　"
            f"上榜后5日 {num('上榜后5日', '{:+.2f}')}%")
        smry.setWordWrap(True)
        smry.setStyleSheet("color:#555;")
        v.addWidget(smry)

        why = QLabel(f"上榜原因：{row.get('上榜原因', '—')}　｜　席位解读：{row.get('解读', '—')}")
        why.setWordWrap(True)
        why.setStyleSheet("color:#8a5a00;")
        v.addWidget(why)

        self.status = QLabel("正在拉取当日买卖席位…")
        self.status.setStyleSheet("color:#888;")
        v.addWidget(self.status)

        split = QSplitter(Qt.Orientation.Vertical)
        self.buy_tb = _table(_SEAT_COLS, widths=_SEAT_W, stretch_col=1)
        self.sell_tb = _table(_SEAT_COLS, widths=_SEAT_W, stretch_col=1)
        split.addWidget(_pane("买方前五（按买入金额排）", self.buy_tb))
        split.addWidget(_pane("卖方前五（按卖出金额排）", self.sell_tb))
        split.setSizes([290, 290])
        v.addWidget(split, 1)

        bar = QHBoxLayout()
        bar.addStretch(1)
        b = QPushButton("关闭")
        b.clicked.connect(self.accept)
        bar.addWidget(b)
        v.addLayout(bar)

        self._w = Worker(events.lhb_seats, code, day)
        self._w.done.connect(self._on_data)
        self._w.fail.connect(self._on_fail)
        self._w.start()

    def _on_fail(self, msg):
        self.status.setText("拉席位失败：" + msg.splitlines()[0])
        self.status.setStyleSheet("color:#d0342c;")

    def _on_data(self, res):
        if res.get("err"):
            self.status.setText("拉席位失败：" + str(res["err"])[:120])
            self.status.setStyleSheet("color:#d0342c;")
        buy, sell = res.get("buy"), res.get("sell")
        self._fill(self.buy_tb, buy, "买入金额")
        self._fill(self.sell_tb, sell, "卖出金额")

        inst = 0.0
        for x in (buy, sell):
            if x is None or getattr(x, "empty", True):
                continue
            inst += float(x.loc[x["性质"].isin(["机构", "北向"]), "净额"]
                          .fillna(0).sum())
        self.status.setText(
            f"机构与北向席位净额合计 {inst / 1e4:+,.0f} 万元"
            "（买卖两份榜单各计前五，同一席位可能重复计入）")
        self.status.setStyleSheet("color:#333;")
        self.resize(self.width(), 250 + self.buy_tb.height() + self.sell_tb.height())

    @staticmethod
    def _fill(tb: QTableWidget, df, by: str):
        if df is None or getattr(df, "empty", True):
            tb.setRowCount(1)
            _put(tb, 0, 0, "—", None, True)
            _put(tb, 0, 1, "该榜单没有取到数据（可能当日未单独公布）")
            return
        d = df.sort_values(by, ascending=False)
        tb.setRowCount(len(d))
        for i, (_, r) in enumerate(d.iterrows()):
            kind = str(r.get("性质", ""))
            kc = (UP if kind == "机构"
                  else QColor("#1f6fd0") if kind == "北向"
                  else MUTED if kind == "散户大营" else None)
            _put(tb, i, 0, i + 1, None, True)
            _put(tb, i, 1, r["交易营业部名称"], None, False,
                 tooltip=str(r["交易营业部名称"]))
            _put(tb, i, 2, kind, kc, True)
            _put(tb, i, 3, f"{float(r['买入金额']) / 1e4:,.0f}", None, True)
            _put(tb, i, 4, f"{float(r['卖出金额']) / 1e4:,.0f}", None, True)
            net = float(r["净额"]) / 1e4
            _put(tb, i, 5, f"{net:+,.0f}", _money_color(net), True)
            _put(tb, i, 6, f"{float(r.get('买入金额-占总成交比例', 0)):.2f}", None, True)
            _put(tb, i, 7, f"{float(r.get('卖出金额-占总成交比例', 0)):.2f}", None, True)
        tb.fit_columns(max_w=520)
        tb.setFixedHeight(tb.horizontalHeader().height() + max(len(d), 1) * 23 + 8)


class EventTab(QWidget):
    """事件雷达：公告、解禁、龙虎榜、增减持。

    每张表都能点表头排序、拖列宽、换列序；每一行双击都能看到底稿。
    """

    NOTICE_COLS = ["日期", "代码", "名称", "分类", "方向", "强度", "公告标题"]
    UNLOCK_COLS = ["解禁日", "代码", "名称", "解禁市值亿", "占流通%", "类型"]
    LHB_COLS = ["上榜日", "代码", "名称", "上榜原因", "解读",
                "净买额万", "买入万", "卖出万", "净买占成交%", "换手%", "上榜后5日%"]
    HC_COLS = ["日期", "代码", "名称", "变动人", "关系", "方向", "变动股数万",
               "变动金额万", "变动比例%", "变动后持股万", "职务", "变动原因"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._notice_view = pd.DataFrame()
        self._unlock = pd.DataFrame()
        self._lb = pd.DataFrame()
        self._lhb_view = pd.DataFrame()
        self._hc = pd.DataFrame()
        self._hc_view = pd.DataFrame()

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        bar = QHBoxLayout()
        self.btn = QPushButton("刷新全部事件")
        self.btn.clicked.connect(lambda *_: self.run_load())
        bar.addWidget(self.btn)
        bar.addWidget(QLabel("回看天数"))
        self.days = QSpinBox()
        self.days.setRange(5, 120)
        self.days.setValue(25)
        self.days.setSuffix(" 天")
        bar.addWidget(self.days)
        self.status = QLabel("就绪。增减持首次拉全量约 3 分钟，之后走本地缓存")
        self.status.setStyleSheet("color:#888;")
        bar.addWidget(self.status, 1)
        root.addLayout(bar)

        self.sub = QTabWidget()

        # ---------------------------------------------------------- 公告
        w1 = QWidget()
        v1 = QVBoxLayout(w1)
        v1.setContentsMargins(4, 4, 4, 4)
        v1.setSpacing(4)
        self.notice_info = QLabel("—")
        self.notice_info.setStyleSheet("color:#444;")
        v1.addWidget(self.notice_info)
        self.notice_tb = _table(
            self.NOTICE_COLS, stretch_col=6,
            widths={0: 88, 1: 66, 2: 78, 3: 80, 4: 54, 5: 46})
        self.notice_tb.setToolTip("双击任意一行看公告原文")
        self.notice_tb.cellDoubleClicked.connect(self._notice_detail)
        v1.addWidget(self.notice_tb, 1)
        self.sub.addTab(w1, "公告雷达")

        # ---------------------------------------------------------- 解禁
        w2 = QWidget()
        v2 = QVBoxLayout(w2)
        v2.setContentsMargins(4, 4, 4, 4)
        v2.setSpacing(4)
        self.unlock_tb = _table(
            self.UNLOCK_COLS, stretch_col=5,
            widths={0: 88, 1: 66, 2: 78, 3: 96, 4: 78})
        self.unlock_tb.cellDoubleClicked.connect(self._unlock_detail)
        self.uday_tb = _table(["日期", "家数", "解禁市值亿", ""],
                              widths={0: 96, 1: 64, 2: 100})
        self.uday_tb.cellDoubleClicked.connect(self._uday_detail)
        sp2 = QSplitter(Qt.Orientation.Vertical)
        sp2.addWidget(_pane("未来解禁明细（双击看这一笔的全部字段，按解禁市值排）",
                            self.unlock_tb))
        sp2.addWidget(_pane("解禁日历（双击看当天解禁了哪些票）", self.uday_tb))
        sp2.setSizes([420, 220])
        v2.addWidget(sp2, 1)
        self.sub.addTab(w2, "解禁日程")

        # ---------------------------------------------------------- 龙虎榜
        w3 = QWidget()
        v3 = QVBoxLayout(w3)
        v3.setContentsMargins(4, 4, 4, 4)
        v3.setSpacing(4)
        self.lhb_info = QLabel("—")
        self.lhb_info.setStyleSheet("color:#444;")
        v3.addWidget(self.lhb_info)

        f3 = QHBoxLayout()
        f3.addWidget(QLabel("方向"))
        self.lhb_side = QComboBox()
        self.lhb_side.addItems(["全部", "仅净买入", "仅净卖出"])
        self.lhb_side.currentIndexChanged.connect(lambda *_: self._refresh_lhb())
        f3.addWidget(self.lhb_side)
        f3.addWidget(QLabel("上榜原因"))
        self.lhb_reason = QComboBox()
        self.lhb_reason.setMinimumWidth(300)
        self.lhb_reason.setMaximumWidth(420)
        self.lhb_reason.addItem("全部上榜原因")
        self.lhb_reason.currentIndexChanged.connect(lambda *_: self._refresh_lhb())
        f3.addWidget(self.lhb_reason)
        f3.addStretch(1)
        self.lhb_kw = QLineEdit()
        self.lhb_kw.setPlaceholderText("代码 / 名称")
        self.lhb_kw.setMaximumWidth(160)
        self.lhb_kw.textChanged.connect(lambda *_: self._refresh_lhb())
        f3.addWidget(self.lhb_kw)
        v3.addLayout(f3)

        self.lhb_tb = _table(
            self.LHB_COLS, stretch_col=3,
            widths={0: 88, 1: 66, 2: 76, 4: 210, 5: 80, 6: 76, 7: 76,
                    8: 92, 9: 62, 10: 84})
        self.lhb_tb.setToolTip("双击任意一行看当日买卖席位明细")
        self.lhb_tb.cellDoubleClicked.connect(self._lhb_detail)
        self.lhbv_tb = _table(["维度", "分组", "样本", "5日均值%", "中位%"],
                              widths={0: 88, 2: 58, 3: 86, 4: 76}, stretch_col=1)
        self.lhbv_tb.setToolTip("双击某一组，看这组里有哪些票、哪天上榜")
        self.lhbv_tb.cellDoubleClicked.connect(self._lhbv_detail)
        sp3 = QSplitter(Qt.Orientation.Vertical)
        sp3.addWidget(_pane("个股上榜明细（双击看具体机构和金额）", self.lhb_tb))
        sp3.addWidget(_pane("分组有效性：上榜后 5 日到底赚不赚", self.lhbv_tb))
        sp3.setSizes([460, 240])
        v3.addWidget(sp3, 1)
        self.sub.addTab(w3, "龙虎榜")

        # ---------------------------------------------------------- 增减持
        w4 = QWidget()
        v4 = QVBoxLayout(w4)
        v4.setContentsMargins(4, 4, 4, 4)
        v4.setSpacing(4)
        self.hc_info = QLabel("—")
        self.hc_info.setStyleSheet("color:#444;")
        v4.addWidget(self.hc_info)

        f4 = QHBoxLayout()
        f4.addWidget(QLabel("方向"))
        self.hc_side = QComboBox()
        self.hc_side.addItems(["全部", "只看增持", "只看减持"])
        self.hc_side.currentIndexChanged.connect(lambda *_: self._refresh_hc())
        f4.addWidget(self.hc_side)
        self.hc_only_mgmt = QCheckBox("只看本人操作")
        self.hc_only_mgmt.stateChanged.connect(lambda *_: self._refresh_hc())
        f4.addWidget(self.hc_only_mgmt)
        f4.addWidget(QLabel("金额≥"))
        self.hc_min = QDoubleSpinBox()
        self.hc_min.setRange(0, 100000)
        self.hc_min.setDecimals(0)
        self.hc_min.setSingleStep(50)
        self.hc_min.setValue(0)
        self.hc_min.setSuffix(" 万")
        self.hc_min.valueChanged.connect(lambda *_: self._refresh_hc())
        f4.addWidget(self.hc_min)
        self.hc_kw = QLineEdit()
        self.hc_kw.setPlaceholderText("代码 / 名称 / 人名")
        self.hc_kw.setMaximumWidth(180)
        self.hc_kw.textChanged.connect(lambda *_: self._refresh_hc())
        f4.addWidget(self.hc_kw)
        f4.addStretch(1)
        v4.addLayout(f4)

        self.hc_tb = _table(
            self.HC_COLS,
            widths={0: 88, 1: 66, 2: 76, 3: 80, 4: 60, 5: 54, 6: 94, 7: 94,
                    8: 78, 9: 96, 10: 110})
        self.hc_tb.setToolTip("双击任意一笔看完整字段")
        self.hc_tb.cellDoubleClicked.connect(self._hc_detail)
        self.hcs_tb = _table(["代码", "名称", "笔数", "增持笔", "减持笔",
                              "增持万", "减持万", "净额万", ""],
                             widths={0: 70, 1: 90, 2: 58, 3: 68, 4: 68,
                                     5: 88, 6: 88, 7: 92})
        self.hcs_tb.setToolTip("双击某只票，看它的每一笔增减持")
        self.hcs_tb.cellDoubleClicked.connect(self._hcs_detail)
        sp4 = QSplitter(Qt.Orientation.Vertical)
        sp4.addWidget(_pane("逐笔明细：谁、哪天、动了多少股、多少钱", self.hc_tb))
        sp4.addWidget(_pane("按个股汇总：增持和减持分开算，再给净额", self.hcs_tb))
        sp4.setSizes([470, 230])
        v4.addWidget(sp4, 1)
        self.sub.addTab(w4, "增减持")

        root.addWidget(self.sub, 1)

    # ------------------------------------------------------------------ 拉数
    def run_load(self):
        if self._worker and self._worker.isRunning():
            return
        self.btn.setEnabled(False)
        self.status.setText("正在拉公告、解禁、龙虎榜、增减持…首次较慢")
        self._worker = Worker(self._job)
        self._worker.done.connect(self._on_done)
        self._worker.fail.connect(self._on_fail)
        self._worker.start()

    def _job(self):
        days = int(self.days.value())
        out = {"notice": pd.DataFrame(), "nsum": {"total": 0},
               "unlock": pd.DataFrame(), "uday": pd.DataFrame(),
               "lb": pd.DataFrame(), "lbv": [], "hc": pd.DataFrame(),
               "hcs": pd.DataFrame()}
        try:
            out["notice"] = events.notice_radar()
            out["nsum"] = events.notice_summary(out["notice"])
        except Exception:  # noqa: BLE001
            pass
        try:
            out["unlock"] = events.unlock(days=30)
            out["uday"] = events.unlock_daily(days=30)
        except Exception:  # noqa: BLE001
            pass
        try:
            out["lb"] = events.lhb(days=days)
            out["lbv"] = events.lhb_verdict(out["lb"])
        except Exception:  # noqa: BLE001
            pass
        try:
            out["hc"] = events.hold_changes(days=days)
            out["hcs"] = events.hold_changes_summary(out["hc"])
        except Exception:  # noqa: BLE001
            pass
        return out

    def _on_done(self, p):
        self.btn.setEnabled(True)
        self._notice_view = pd.DataFrame()
        self._fill_notice(p.get("notice"), p.get("nsum") or {})
        self._fill_unlock(p.get("unlock"), p.get("uday"))
        self._lb = p.get("lb") if p.get("lb") is not None else pd.DataFrame()
        self._lbv = p.get("lbv") or []
        self._fill_lhbv()
        self._refresh_lhb()
        self._hc = p.get("hc") if p.get("hc") is not None else pd.DataFrame()
        self._refresh_hc()
        hcs = p.get("hcs")
        self._fill_hcs(hcs)

        ns = p.get("nsum") or {}
        self.status.setText(
            f"完成：公告 {ns.get('total', 0)} 条｜解禁 {len(self._unlock)} 笔"
            f"｜龙虎榜 {len(self._lb)} 条｜增减持 {len(self._hc)} 笔")

    def _on_fail(self, msg):
        self.btn.setEnabled(True)
        self.status.setText("出错了：" + msg.splitlines()[0])
        print(msg)

    # ---------------------------------------------------------------- 公告
    def _fill_notice(self, df, nsum):
        tb = self.notice_tb
        if df is None or df.empty:
            tb.setRowCount(0)
            self.notice_info.setText("今天没有取到公告（周末或非交易日可换成最近交易日）")
            return
        risky = df[df["方向"] == "risk"].head(400)
        rest = df[df["方向"] != "risk"].head(200)
        view = pd.concat([risky, rest])
        self._notice_view = view
        tb.setRowCount(len(view))
        for i, (idx, r) in enumerate(view.iterrows()):
            risk = r["方向"] == "risk"
            chance = r["方向"] == "chance"
            _put(tb, i, 0, str(r.get("公告日期", ""))[:10], None, True)
            _put(tb, i, 1, r["代码"], None, True)
            _put(tb, i, 2, r["名称"])
            _put(tb, i, 3, r["标签"], None, True)
            _put(tb, i, 4, "风险" if risk else ("机会" if chance else "中性"),
                 DOWN if risk else (UP if chance else MUTED), True)
            _put(tb, i, 5, int(r["强度"]), None, True)
            title = str(r["公告标题"])
            _put(tb, i, 6, title, None, False, tooltip=title)
            _bind(tb, i, int(idx))
        self.notice_info.setText(
            f"共 {nsum.get('total', 0)} 条公告：风险 {nsum.get('risk', 0)} 条"
            f"（高强度 {nsum.get('risk_strong', 0)}），机会 {nsum.get('chance', 0)} 条。"
            "风险类排前面，强度按关键词打分；双击任一行看全文与原文链接")

    def _notice_detail(self, row, _col):
        idx = _ref(self.notice_tb, row)
        if idx is None or idx not in self._notice_view.index:
            return
        NoticeDialog(self, self._notice_view.loc[idx]).exec()

    # ---------------------------------------------------------------- 解禁
    def _fill_unlock(self, unlock, uday):
        self._unlock = unlock if unlock is not None else pd.DataFrame()
        tb = self.unlock_tb
        if len(self._unlock):
            rows = self._unlock.head(200)
            tb.setRowCount(len(rows))
            for i, (idx, r) in enumerate(rows.iterrows()):
                _put(tb, i, 0, str(r["解禁时间"])[:10], None, True)
                _put(tb, i, 1, r["股票代码"], None, True)
                _put(tb, i, 2, r["股票简称"])
                _put(tb, i, 3, f"{r['实际解禁市值'] / 1e8:,.1f}", None, True)
                p = r.get("占流通%", 0) or 0
                _put(tb, i, 4, f"{p:.1f}", DOWN if p >= 5 else None, True)
                _put(tb, i, 5, str(r["限售股类型"]))
                _bind(tb, i, int(idx))
        else:
            tb.setRowCount(0)

        if uday is not None and len(uday):
            self.uday_tb.setRowCount(len(uday))
            for i, (_, r) in enumerate(uday.iterrows()):
                _put(self.uday_tb, i, 0, str(r["解禁时间"])[:10], None, True)
                _put(self.uday_tb, i, 1, int(r["当日解禁股票家数"]), None, True)
                _put(self.uday_tb, i, 2, f"{r['解禁市值亿']:,.1f}", None, True)
        else:
            self.uday_tb.setRowCount(0)

    def _unlock_detail(self, row, _col):
        idx = _ref(self.unlock_tb, row)
        if idx is None or idx not in self._unlock.index:
            return
        r = self._unlock.loc[idx]
        DetailDialog(self, title=f"{r['股票简称']}（{r['股票代码']}）解禁明细",
                     subtitle=str(r["解禁时间"])[:10],
                     fields=row_fields(r)).exec()

    def _uday_detail(self, row, _col):
        d = self.uday_tb.item(row, 0)
        if d is None or not len(self._unlock):
            return
        day = d.text()
        sub = self._unlock[self._unlock["解禁时间"].astype(str).str[:10] == day]
        rows = []
        for _, r in sub.iterrows():
            rows.append([r["股票代码"], r["股票简称"],
                         f"{r['实际解禁市值'] / 1e8:,.1f}",
                         f"{float(r.get('占流通%', 0) or 0):.1f}",
                         str(r.get("限售股类型", ""))])
        DetailDialog(self, title=f"{day} 解禁 {len(sub)} 家",
                     subtitle="按解禁市值排，占流通比越高、解禁市值越大，抛压越实",
                     tables=[{"caption": "当日解禁清单",
                              "headers": ["代码", "名称", "解禁市值亿", "占流通%", "类型"],
                              "rows": rows, "right": [2, 3]}]).exec()

    # -------------------------------------------------------------- 龙虎榜
    def _fill_lhbv(self):
        lbv = getattr(self, "_lbv", [])
        tb = self.lhbv_tb
        if not lbv:
            tb.setRowCount(0)
            return
        tb.setRowCount(len(lbv))
        for i, x in enumerate(lbv):
            _put(tb, i, 0, x["维度"], None, True)
            _put(tb, i, 1, x["分组"], None, False, tooltip=x["分组"])
            _put(tb, i, 2, x["样本"], None, True)
            v = x["均值"]
            _put(tb, i, 3, f"{v:+.2f}", _money_color(v), True)
            _put(tb, i, 4, f"{x['中位']:+.2f}", _money_color(x["中位"]), True)
            _bind(tb, i, i)

    def _refresh_lhb(self):
        tb = self.lhb_tb
        d = self._lb
        if d is None or getattr(d, "empty", True):
            tb.setRowCount(0)
            self.lhb_info.setText("龙虎榜数据为空")
            return
        f = d
        side = self.lhb_side.currentText()
        if side == "仅净买入":
            f = f[f["龙虎榜净买额"] > 0]
        elif side == "仅净卖出":
            f = f[f["龙虎榜净买额"] < 0]
        rk = self.lhb_reason.currentText()
        if rk and rk != "全部上榜原因" and "上榜原因" in f.columns:
            f = f[f["上榜原因"].astype(str) == rk]
        kw = self.lhb_kw.text().strip()
        if kw:
            hit = (f["代码"].astype(str).str.contains(kw, na=False)
                   | f["名称"].astype(str).str.contains(kw, na=False))
            f = f[hit]
        f = f.sort_values("上榜日", ascending=False)
        self._lhb_view = f

        tb.setRowCount(len(f))
        for i, (idx, r) in enumerate(f.iterrows()):
            _put(tb, i, 0, str(r["上榜日"])[:10], None, True)
            _put(tb, i, 1, r["代码"], None, True)
            _put(tb, i, 2, r["名称"])
            _put(tb, i, 3, str(r.get("上榜原因", "")), None, False,
                 tooltip=str(r.get("上榜原因", "")))
            _put(tb, i, 4, str(r.get("解读", "")), None, False,
                 tooltip=str(r.get("解读", "")))
            net = _fnum(r.get("龙虎榜净买额")) / 1e4
            _put(tb, i, 5, _n(net, "{:+,.0f}"), _col_of(net), True)
            _put(tb, i, 6, _n(_fnum(r.get("龙虎榜买入额")) / 1e4), None, True)
            _put(tb, i, 7, _n(_fnum(r.get("龙虎榜卖出额")) / 1e4), None, True)
            rat8 = _fnum(r.get("净买额占总成交比"))
            _put(tb, i, 8, _n(rat8, "{:+.2f}"), _col_of(rat8), True)
            _put(tb, i, 9, _n(r.get("换手率"), "{:.2f}"), None, True)
            p5 = _fnum(r.get("上榜后5日"))
            _put(tb, i, 10, _n(p5, "{:+.2f}"), _col_of(p5), True)
            _bind(tb, i, int(idx))

        neg = sum(1 for x in getattr(self, "_lbv", []) if x["均值"] < 0)
        tot = len(getattr(self, "_lbv", []))
        self.lhb_info.setText(
            f"当前显示 {len(f)} 条（共 {len(d)} 条）｜{tot} 个分组里 {neg} 组上榜后 5 日是亏的。"
            "双击一行看当天买方前五、卖方前五具体是谁；这张表不是让你跟着买，是让你看跟单的期望值")

    def _lhb_detail(self, row, _col):
        idx = _ref(self.lhb_tb, row)
        if idx is None or idx not in self._lhb_view.index:
            return
        SeatDialog(self, self._lhb_view.loc[idx]).exec()

    def _lhbv_detail(self, row, _col):
        i = _ref(self.lhbv_tb, row)
        lbv = getattr(self, "_lbv", [])
        if i is None or i >= len(lbv) or self._lb is None or self._lb.empty:
            return
        x = lbv[i]
        col = "上榜后5日" if "上榜后5日" in self._lb.columns else "上榜后1日"
        key = "上榜原因" if x["维度"] == "上榜原因" else "解读"
        if key not in self._lb.columns:
            return
        sub = self._lb[self._lb[key].astype(str) == str(x["分组"])]
        sub = sub.sort_values("上榜日", ascending=False).head(300)
        rows = []
        for _, r in sub.iterrows():
            p = r.get(col)
            rows.append([str(r["上榜日"])[:10], r["代码"], r["名称"],
                         f"{float(r.get('龙虎榜净买额', 0) or 0) / 1e4:+,.0f}",
                         "—" if p is None or (isinstance(p, float) and pd.isna(p))
                         else f"{float(p):+.2f}"])
        DetailDialog(
            self, title=f"{x['维度']}：{x['分组']}",
            subtitle=f"样本 {x['样本']} 条，平均 {x['均值']:+.2f}%，中位 {x['中位']:+.2f}%"
                     f"（以{col}计）",
            tables=[{"caption": "这一组包含的上榜记录",
                     "headers": ["上榜日", "代码", "名称", "净买额万", col],
                     "rows": rows, "right": [3, 4]}]).exec()

    # -------------------------------------------------------------- 增减持
    def _refresh_hc(self):
        tb = self.hc_tb
        d = self._hc
        if d is None or getattr(d, "empty", True):
            tb.setRowCount(0)
            self.hc_info.setText("增减持数据为空")
            return
        f = d
        side = self.hc_side.currentText()
        if side == "只看增持":
            f = f[f["方向"] == "增持"]
        elif side == "只看减持":
            f = f[f["方向"] == "减持"]
        if self.hc_only_mgmt.isChecked() and "变动人与董监高的关系" in f.columns:
            # 亲属与受控法人的买卖常常是变相减持，默认混在一起看会低估真实抛压
            f = f[f["变动人与董监高的关系"].astype(str) == "本人"]
        floor = float(self.hc_min.value()) * 1e4
        if floor > 0:
            f = f[f["金额"].abs() >= floor]
        kw = self.hc_kw.text().strip()
        if kw:
            hit = (f["代码"].astype(str).str.contains(kw, na=False)
                   | f["名称"].astype(str).str.contains(kw, na=False)
                   | f["变动人"].astype(str).str.contains(kw, na=False))
            f = f[hit]
        f = f.sort_values("日期", ascending=False).head(3000)
        self._hc_view = f

        tb.setRowCount(len(f))
        for i, (idx, r) in enumerate(f.iterrows()):
            buy = r["方向"] == "增持"
            _put(tb, i, 0, str(r["日期"])[:10], None, True)
            _put(tb, i, 1, r["代码"], None, True)
            _put(tb, i, 2, r["名称"])
            person = str(r.get("变动人", "") or "")
            _put(tb, i, 3, person, None, False, tooltip=person)
            rel = str(r.get("变动人与董监高的关系", "") or "")
            _put(tb, i, 4, rel, MUTED if rel != "本人" else None, True)
            _put(tb, i, 5, r["方向"], UP if buy else DOWN, True)
            sh = _fnum(r.get("变动股数")) / 1e4
            _put(tb, i, 6, _n(sh, "{:+,.2f}"), UP if buy else DOWN, True)
            amt = _fnum(r.get("金额")) / 1e4
            _put(tb, i, 7, _n(amt, "{:+,.1f}"), UP if buy else DOWN, True)
            _put(tb, i, 8, _n(r.get("变动比例"), "{:.3f}"), None, True)
            _put(tb, i, 9, _n(_fnum(r.get("变动后持股数")) / 1e4, "{:,.1f}"), None, True)
            job = str(r.get("职务", "") or "")
            _put(tb, i, 10, job, MUTED if not job else None, False, tooltip=job)
            src = str(r.get("变动原因", "") or "")
            _put(tb, i, 11, src, None, False, tooltip=src)
            _bind(tb, i, int(idx))

        up_n = int((f["方向"] == "增持").sum())
        dn_n = int((f["方向"] == "减持").sum())
        net = float(f.loc[f["方向"] == "增持", "金额"].sum()
                    + f.loc[f["方向"] == "减持", "金额"].sum()) / 1e4
        self.hc_info.setText(
            f"共 {len(d)} 笔，当前显示 {len(f)} 笔：增持 {up_n} 笔、减持 {dn_n} 笔，"
            f"合计净额 {net:+,.0f} 万元，涉及 {f['代码'].nunique()} 只。"
            "双击任一笔看全部字段（含变动前后持股数、职务、与董监高关系）。"
            "关系列不是本人时，多半是亲属或受控法人在动，抛压一样真实")

    def _fill_hcs(self, hcs):
        tb = self.hcs_tb
        if hcs is None or getattr(hcs, "empty", True):
            tb.setRowCount(0)
            return
        rows = hcs.head(300)
        tb.setRowCount(len(rows))
        for i, (_, r) in enumerate(rows.iterrows()):
            _put(tb, i, 0, r["代码"], None, True)
            _put(tb, i, 1, r["名称"])
            _put(tb, i, 2, int(r["笔数"]), None, True)
            _put(tb, i, 3, int(r["增持笔"]), None, True)
            _put(tb, i, 4, int(r["减持笔"]), None, True)
            _put(tb, i, 5, f"{r['增持额'] / 1e4:,.0f}", UP, True)
            _put(tb, i, 6, f"{r['减持额'] / 1e4:,.0f}", DOWN, True)
            net = r["净额"] / 1e4
            _put(tb, i, 7, f"{net:+,.0f}", _money_color(net), True)

    def _hcs_detail(self, row, _col):
        it = self.hcs_tb.item(row, 0)
        if it is None or self._hc is None or self._hc.empty:
            return
        code = it.text()
        sub = self._hc[self._hc["代码"] == code].sort_values("日期", ascending=False)
        if sub.empty:
            return
        rows = []
        for _, r in sub.iterrows():
            rows.append([
                str(r["日期"])[:10], str(r.get("变动人", "") or ""),
                str(r.get("变动人与董监高的关系", "") or ""), r["方向"],
                f"{float(r.get('变动股数', 0) or 0) / 1e4:+,.2f}",
                f"{float(r.get('金额', 0) or 0) / 1e4:+,.1f}",
                str(r.get("职务", "") or ""), str(r.get("变动原因", "") or ""),
            ])
        net = sum(float(r.get("金额", 0) or 0) for _, r in sub.iterrows()) / 1e4
        DetailDialog(
            self, title=f"{sub.iloc[0]['名称']}（{code}）共 {len(sub)} 笔增减持",
            subtitle=f"净额 {net:+,.1f} 万元，增持 {int((sub['方向'] == '增持').sum())} 笔、"
                     f"减持 {int((sub['方向'] == '减持').sum())} 笔",
            tables=[{"caption": "逐笔明细",
                     "headers": ["日期", "变动人", "关系", "方向", "股数万",
                                 "金额万", "职务", "原因"],
                     "rows": rows, "right": [4, 5]}]).exec()

    def _hc_detail(self, row, _col):
        idx = _ref(self.hc_tb, row)
        if idx is None or idx not in self._hc_view.index:
            return
        r = self._hc_view.loc[idx]
        buy = r["方向"] == "增持"
        DetailDialog(
            self, title=f"{r['名称']}（{r['代码']}）{r['变动人']} {r['方向']}",
            subtitle=f"{str(r['日期'])[:10]}　{('+' if buy else '')}"
                     f"{float(r.get('变动金额', 0) or 0) / 1e4:,.1f} 万元",
            fields=row_fields(r)).exec()
# ============================================================ 多曲线净值图

class MultiLineChart(QWidget):
    """多条归一化净值曲线叠加。支持滚轮缩放、拖动平移、双击复位、悬停看数值。

    和 EquityChart 的区别：那个是单序列 + 买卖点，这个是多序列对比。
    多市场净值和可转债轮动都要「我的曲线 vs 基准」，所以单独做一个。
    """

    MIN_SPAN = 8.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dates: list[str] = []
        self.lines: list[dict] = []
        self._bg = None
        self._drag = None
        self._hl = []

        self.fig = Figure(figsize=(6, 2.8), dpi=100, facecolor="#ffffff")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.subplots_adjust(left=0.085, right=0.995, top=0.885, bottom=0.175)
        self._style()

        self.v = self.h = self.info = None

        hint = QLabel("滚轮缩放 · 按住拖动平移 · 双击复位 · 悬停看各条数值")
        hint.setStyleSheet("color:#8b96a3; font-size:11px; padding:1px 2px;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(hint)
        lay.addWidget(self.canvas, 1)

        c = self.canvas
        c.mpl_connect("motion_notify_event", self._on_move)
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("button_release_event", self._on_release)
        c.mpl_connect("scroll_event", self._on_scroll)
        c.mpl_connect("figure_leave_event", self._on_leave)
        c.mpl_connect("resize_event", self._on_resize)

    def _style(self):
        ax = self.ax
        ax.set_facecolor("#ffffff")
        ax.grid(True, color="#e6eaee", lw=0.6)
        ax.tick_params(labelsize=8, colors="#55606b", length=3)
        for sp in ax.spines.values():
            sp.set_color("#d7dce2")

    def show_placeholder(self, text):
        self.dates, self.lines, self._bg = [], [], None
        self.ax.cla()
        self._style()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(False)
        self.ax.text(0.5, 0.5, text, ha="center", va="center",
                     transform=self.ax.transAxes, fontsize=12, color="#98a2ae")
        self.v = self.h = self.info = None
        if self.isVisible():
            self._draw_now()

    def set_data(self, dates: list[str], lines: list[dict], keep_view=False):
        """lines: [{"name":..., "values":[从 1.0 起的净值], "color":..., "lw":..., "ls":...}]

        keep_view=True 用于勾选/取消某条曲线时保持当前缩放位置，
        否则每次加点东西都会被拉回全区间。
        """
        same = keep_view and self.dates == list(dates) and self.dates
        self.dates = list(dates)
        self.lines = [ln for ln in lines if ln.get("values")]
        if not self.dates or not self.lines:
            self.show_placeholder("无数据")
            return
        self._render()
        if not same:
            self._reset_view()
        self._autoscale_y()
        self._draw_now()
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)

    def _render(self):
        ax = self.ax
        ax.cla()
        self._style()
        n = len(self.dates)
        x = np.arange(n, dtype=float)
        for ln in self.lines:
            v = np.asarray(ln["values"], dtype=float)
            m = min(len(v), n)
            col = ln.get("color", "#1f6fd0")
            ax.plot(x[:m], (v[:m] - 1.0) * 100.0, color=col,
                    lw=ln.get("lw", 1.3), ls=ln.get("ls", "-"),
                    label=ln["name"], zorder=ln.get("z", 4))
        ax.axhline(0, color="#9aa4ae", ls="--", lw=0.8, zorder=2)
        ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9,
                  ncol=min(3, len(self.lines)), borderpad=0.4,
                  handlelength=1.6, columnspacing=1.2)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _p=None: f"{v:+.0f}%"))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, integer=True,
                                                       min_n_ticks=3))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(self._fmt_x))

        self.v = ax.axvline(0, color="#5b6b7c", lw=0.8, ls="--",
                            visible=False, zorder=7)
        self.h = ax.axhline(0, color="#5b6b7c", lw=0.8, ls="--",
                            visible=False, zorder=7)
        self.info = ax.text(
            0.012, 0.03, "", transform=ax.transAxes, va="bottom", ha="left",
            fontsize=8.5, color="#222222", visible=False, zorder=9,
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.42", fc="#ffffff", ec="#c8cdd4",
                      lw=0.7, alpha=0.96))

    def _fmt_x(self, val, _pos=None):
        if not self.dates:
            return ""
        i = int(round(val))
        if i < 0 or i >= len(self.dates):
            return ""
        d = str(self.dates[i])
        if len(d) < 10:
            return d
        # 跨度超过 400 个交易日时给年月就够，否则给「月-日」，
        # 原来一直取 [2:7] 只能拿到「26-08」，同一月里刻度全一样，没有信息量
        return d[2:7] if len(self.dates) > 400 else d[5:10]

    def _reset_view(self, n_last: int | None = None):
        n = len(self.dates)
        if not n:
            return
        if n_last is None:
            n_last = n
        n_last = max(self.MIN_SPAN, min(n, n_last))
        self.ax.set_xlim(max(-0.5, n - n_last - 0.5), n - 0.5)

    def reset_view(self):
        if not self.dates:
            return
        self._reset_view()
        self._autoscale_y()
        self._draw_now()

    def _visible_range(self):
        x0, x1 = self.ax.get_xlim()
        i0 = max(0, int(np.floor(x0)))
        i1 = min(len(self.dates) - 1, int(np.ceil(x1)))
        if i1 <= i0:
            return 0, len(self.dates) - 1
        return i0, i1

    def _autoscale_y(self):
        if not self.lines:
            return
        i0, i1 = self._visible_range()
        vals = []
        for ln in self.lines:
            v = np.asarray(ln["values"], dtype=float)[i0:i1 + 1]
            if len(v):
                vals.append((v - 1.0) * 100.0)
        if not vals:
            return
        allv = np.concatenate(vals)
        allv = allv[np.isfinite(allv)]
        if not len(allv):
            return
        lo, hi = float(allv.min()), float(allv.max())
        pad = max((hi - lo) * 0.14, 1.2)
        self.ax.set_ylim(lo - pad, hi + pad)

    def _set_overlay_visible(self, vis):
        for a in (self.v, self.h, self.info):
            if a is not None:
                a.set_visible(vis)

    def _draw_now(self):
        self._set_overlay_visible(False)
        try:
            self.canvas.draw()
            self._bg = self.canvas.copy_from_bbox(self.ax.bbox)
        except Exception:
            self._bg = None

    def _blit(self):
        if self._bg is None:
            try:
                self.canvas.draw_idle()
            except Exception:
                pass
            return
        try:
            cv = self.canvas
            cv.restore_region(self._bg)
            for a in (self.v, self.h, self.info):
                if a is not None and a.get_visible():
                    self.ax.draw_artist(a)
            cv.blit(self.ax.bbox)
        except Exception:
            self._bg = None
            try:
                self.canvas.draw_idle()
            except Exception:
                pass

    def _on_resize(self, _event):
        if self.lines:
            self._draw_now()

    def _hide_cross(self):
        if self.v is None or not self.v.get_visible():
            return
        self._set_overlay_visible(False)
        self._blit()

    def _on_leave(self, _event):
        self._hide_cross()

    def _update_cross(self, i, y=None):
        xv = float(i)
        self.v.set_xdata([xv, xv])
        self.v.set_visible(True)
        if y is not None and np.isfinite(y):
            self.h.set_ydata([y, y])
            self.h.set_visible(True)
        else:
            self.h.set_visible(False)

        when = str(self.dates[i])
        rows = [when]
        for ln in self.lines:
            v = ln.get("values") or []
            if i < len(v) and np.isfinite(v[i]):
                rows.append(f"{ln['name'][:10]:<10} {(v[i] - 1) * 100:>+9.2f}%")
        x0, x1 = self.ax.get_xlim()
        on_left = i < (x0 + x1) / 2
        self.info.set_x(0.988 if on_left else 0.012)
        self.info.set_ha("right" if on_left else "left")
        self.info.set_text("\n".join(rows))
        self.info.set_visible(True)
        self._blit()

    def _on_move(self, event):
        if not self.lines:
            return
        if self._drag is not None:
            if event.xdata is None:
                return
            x_press, old0, old1 = self._drag
            dx = event.xdata - x_press
            span = old1 - old0
            n = len(self.dates)
            new0 = old0 - dx
            if new0 < -0.5:
                new0 = -0.5
            if new0 + span > n - 0.5:
                new0 = max(-0.5, n - 0.5 - span)
            self.ax.set_xlim(new0, new0 + span)
            self._autoscale_y()
            self._draw_now()
            return
        if event.inaxes is not self.ax or event.xdata is None:
            self._hide_cross()
            return
        i = int(round(event.xdata))
        if i < 0 or i >= len(self.dates):
            self._hide_cross()
            return
        self._update_cross(i, event.ydata)

    def _on_press(self, event):
        if not self.lines or event.inaxes is not self.ax:
            return
        if event.button == 1:
            self._drag = (event.xdata, *self.ax.get_xlim())

    def _on_release(self, _event):
        self._drag = None

    def _on_scroll(self, event):
        if not self.lines or event.inaxes is not self.ax or event.xdata is None:
            return
        x0, x1 = self.ax.get_xlim()
        n = len(self.dates)
        span = x1 - x0
        factor = 0.82 if event.button == "up" else 1.22
        new_span = max(self.MIN_SPAN, min(float(n), span * factor))
        frac = (event.xdata - x0) / span if span else 0.5
        new0 = event.xdata - new_span * frac
        if new0 < -0.5:
            new0 = -0.5
        if new0 + new_span > n - 0.5:
            new0 = max(-0.5, n - 0.5 - new_span)
        self.ax.set_xlim(new0, new0 + new_span)
        self._autoscale_y()
        self._draw_now()

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.reset_view()
        super().mouseDoubleClickEvent(event)


def _pct(v, nd=2):
    return f"{v * 100:+.{nd}f}%"


class WorkerSig(QThread):
    """带进度回调的 Worker。拉可转债面板要几十秒，不给进度会让人以为卡死。"""
    done = pyqtSignal(object)
    fail = pyqtSignal(str)
    tick = pyqtSignal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    def run(self):
        try:
            self.done.emit(self.fn(self.tick.emit, *self.args))
        except Exception as exc:  # noqa: BLE001
            self.fail.emit(f"{exc}\n\n{traceback.format_exc()}")


def _metric_strip(items: list[tuple[str, str, QColor]]) -> QLabel:
    """一行横排的指标条，指标名浅色、数值深色。"""
    parts = []
    for name, val, col in items:
        c = col.name() if isinstance(col, QColor) else str(col)
        parts.append(f"<span style='color:#7a7a7a;'>{name}</span> "
                     f"<span style='color:{c};font-weight:600;'>{val}</span>")
    lab = QLabel("　｜　".join(parts))
    lab.setTextFormat(Qt.TextFormat.RichText)
    lab.setStyleSheet("font-size:12px; padding:3px 2px;")
    return lab


# ================================================================== 可转债轮动

class RotationTab(QWidget):
    """可转债轮动回测：按因子排序定期换仓，跟中证转债指数对账。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self.data = None          # 缓存面板，换策略不用重拉
        self.result = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        bar = QHBoxLayout()
        self.strategy = QComboBox()
        for k, v in rotation.STRATEGIES.items():
            self.strategy.addItem(f"{v['name']}", k)
        self.freq = QComboBox()
        for k, v in rotation.FREQS.items():
            self.freq.addItem(v, k)
        self.top_n = QSpinBox()
        self.top_n.setRange(5, 50)
        self.top_n.setValue(15)
        self.top_n.setFixedWidth(64)
        self.rating = QComboBox()
        for r in ("不限", "A", "AA-", "AA", "AA+", "AAA"):
            self.rating.addItem(r)
        self.rating.setCurrentText("AA-")
        self.start = QLineEdit("20240101")
        self.start.setFixedWidth(90)
        self.end = QLineEdit(DEFAULT_END)
        self.end.setFixedWidth(90)
        self.btn = QPushButton("跑轮动回测")
        self.btn.clicked.connect(lambda *_: self.run())

        bar.addWidget(QLabel("策略"))
        bar.addWidget(self.strategy)
        bar.addWidget(QLabel("调仓"))
        bar.addWidget(self.freq)
        bar.addWidget(QLabel("持仓"))
        bar.addWidget(self.top_n)
        bar.addWidget(QLabel("评级下限"))
        bar.addWidget(self.rating)
        bar.addWidget(QLabel("区间"))
        bar.addWidget(self.start)
        bar.addWidget(QLabel("→"))
        bar.addWidget(self.end)
        bar.addWidget(self.btn)
        bar.addStretch(1)
        root.addLayout(bar)

        self.status = QLabel("首次运行要拉全市场可转债历史，约 30 秒；之后走缓存")
        self.status.setStyleSheet("color:#888; font-size:12px;")
        root.addWidget(self.status)

        self.strip = _metric_strip([("总收益", "—", MUTED)])
        root.addWidget(self.strip)

        self.chart = MultiLineChart()
        self.chart.show_placeholder("点右上角「跑轮动回测」")
        root.addWidget(self.chart, 3)

        body = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("调仓记录（点某一行，右边换成那一期的持仓）"))
        self.period_tb = _table(
            ["调仓日", "持仓", "均价", "均溢价", "换手", "净值"],
            widths={0: 92, 1: 46, 2: 58, 3: 64, 4: 50}, stretch_col=5)
        self.period_tb.cellClicked.connect(self._on_period_click)
        lv.addWidget(self.period_tb, 1)
        body.addWidget(left)

        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        self.hold_lb = QLabel("最新一期持仓（按因子值从小到大）")
        self.hold_lb.setStyleSheet("color:#444;font-weight:600;")
        mv.addWidget(self.hold_lb)
        self.hold_tb = _table(
            ["#", "代码", "名称", "价格", "溢价率", "因子值", "评级"],
            widths={0: 30, 1: 60, 3: 58, 4: 60, 5: 56, 6: 44}, stretch_col=2)
        mv.addWidget(self.hold_tb, 1)
        body.addWidget(mid)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QLabel("判读"))
        self.read = QTextBrowser()
        rv.addWidget(self.read, 1)
        body.addWidget(right)

        body.setSizes([420, 420, 420])
        root.addWidget(body, 2)

    # ---------------- 运行 ----------------
    def run(self):
        if self._worker and self._worker.isRunning():
            return
        key = self.strategy.currentData()
        freq = self.freq.currentData()
        top_n = self.top_n.value()
        rating = self.rating.currentText()
        rating = "" if rating == "不限" else rating
        start = self.start.text().strip()
        end = self.end.text().strip()
        if self.data is not None and self.data.get("start") == start \
                and self.data.get("end") == end:
            self._start_backtest(key, freq, top_n, rating)
        else:
            self.btn.setEnabled(False)
            self.status.setText("正在拉可转债历史…")
            self._worker = WorkerSig(self._job_build, start, end, key, freq,
                                     top_n, rating)
            self._wire(rating)

    def _start_backtest(self, key, freq, top_n, rating):
        self.btn.setEnabled(False)
        self.status.setText("计算中…")
        self._worker = WorkerSig(self._job_run, key, freq, top_n, rating)
        self._wire(rating)

    def _wire(self, rating):
        w = self._worker
        w.tick.connect(lambda m: self.status.setText(m))
        w.done.connect(self.on_done)
        w.fail.connect(self.on_fail)
        w.start()

    def _job_build(self, progress, start, end, key, freq, top_n, rating):
        def prog(_done, _total, msg):
            progress(msg)
        data = rotation.build_panel(start, end, progress=prog)
        bench = rotation.load_bench(start, end)
        res = rotation.backtest(data["panel"], data["meta"], strategy=key,
                                freq=freq, top_n=top_n, min_rating=rating,
                                bench=bench)
        return data, res

    def _job_run(self, _progress, key, freq, top_n, rating):
        start, end = self.data["start"], self.data["end"]
        bench = rotation.load_bench(start, end)
        res = rotation.backtest(self.data["panel"], self.data["meta"],
                                strategy=key, freq=freq, top_n=top_n,
                                min_rating=rating, bench=bench)
        return self.data, res

    def on_done(self, payload):
        self.btn.setEnabled(True)
        data, res = payload
        self.data = data
        if not res.get("ok"):
            self.status.setText("没算出来：" + res.get("msg", ""))
            return
        self.result = res
        self._fill(res)
        self.status.setText(
            f"完成：{data['n']} 只转债参与，失败 {len(data['failed'])} 只，"
            f"数据耗时 {data['cost']}s｜区间 {res['dates'][0]} → {res['dates'][-1]}")

    def on_fail(self, msg):
        self.btn.setEnabled(True)
        self.status.setText("出错了：" + msg.splitlines()[0])
        print(msg)

    # ---------------- 渲染 ----------------
    def _fill(self, res: dict):
        m = res["metrics"]
        bm = res.get("bench_metrics")
        items = [
            ("总收益", _pct(m["total_return"]), _money_color(m["total_return"])),
            ("年化", _pct(m["annual_return"]), _money_color(m["annual_return"])),
            ("波动", f"{m['vol'] * 100:.1f}%", FG),
            ("夏普", f"{m['sharpe']:.2f}", _money_color(m["sharpe"])),
            ("最大回撤", f"{m['max_drawdown'] * 100:.1f}%", DOWN),
        ]
        if bm:
            items.append(("基准年化", _pct(bm["annual_return"]),
                          _money_color(bm["annual_return"])))
        if res.get("excess"):
            items.append(("超额", f"{res['excess']['annual'] * 100:+.2f}pp",
                          _money_color(res["excess"]["annual"])))
        items.append(("调仓次数", str(res["stats"]["rebalances"]), FG))
        items.append(("平均换手", f"{res['stats']['avg_turnover']:.0f}%", FG))
        self.strip.setText(_metric_strip(items).text())

        lines = [{"name": f"{res['strategy']}（{res['freq']}调仓）",
                  "values": res["nav"], "color": PALETTE[0], "lw": 1.9, "z": 5}]
        if res.get("bench"):
            lines.append({"name": res["bench_name"], "values": res["bench"],
                          "color": "#8a8a8a", "lw": 1.1, "ls": "--", "z": 3})
        self.chart.set_data(res["dates"], lines)

        # 调仓记录（倒序：最近的在上）。期序号挂在行上，排序后照样找得到那一期
        per = list(reversed(res["periods"]))
        self._periods = per
        self.period_tb.setRowCount(len(per))
        for i, p in enumerate(per):
            _put(self.period_tb, i, 0, p["date"], None, True)
            _put(self.period_tb, i, 1, str(p["n"]), None, True)
            _put(self.period_tb, i, 2, f"{p['avg_price']:.2f}", None, True)
            _put(self.period_tb, i, 3, f"{p['avg_premium']:.1f}%", None, True)
            _put(self.period_tb, i, 4, f"{p['turnover']:.0f}%", None, True)
            _put(self.period_tb, i, 5, f"{p['nav']:.3f}", None, True)
            self.period_tb.bind(i, i)
        self._fill_holds(res["holdings"],
                         f"最新一期持仓 · {res['periods'][-1]['date']}"
                         "（按因子值从小到大）" if res["periods"] else "最新一期持仓")

        html = []
        for v in res["verdict"]:
            html.append(f"<p style='margin:6px 0 2px 0;'><b>{v['title']}</b><br>"
                        f"<span style='color:#333;'>{v['text']}</span></p>")
        html.append("<p style='color:#888;font-size:11px;margin-top:8px;'>"
                    "强赎或退市的债，数据断在哪天就按那天的收盘冻结，下一期调仓剔除。"
                    "评级取的是当前值，开了评级下限会带来一点前视偏差。</p>")
        self.read.setHtml("".join(html))


    def _fill_holds(self, holds, label):
        self.hold_lb.setText(label)
        self.hold_tb.setRowCount(len(holds))
        for i, h in enumerate(holds):
            _put(self.hold_tb, i, 0, str(h["rank"]), MUTED, True)
            _put(self.hold_tb, i, 1, h["code"], None, True)
            _put(self.hold_tb, i, 2, h["name"])
            _put(self.hold_tb, i, 3, f"{h['price']:.2f}", None, True)
            pm = h["premium"]
            _put(self.hold_tb, i, 4, "—" if pm is None else f"{pm:.1f}%", None, True)
            _put(self.hold_tb, i, 5, f"{h['score']:.2f}", None, True)
            _put(self.hold_tb, i, 6, h["rating"], MUTED, True)

    def _on_period_click(self, row, _col):
        per = getattr(self, "_periods", None)
        if not per or row < 0:
            return
        idx = self.period_tb.ref(row)
        if not isinstance(idx, int) or not (0 <= idx < len(per)):
            return
        p = per[idx]
        holds = p.get("holdings") or []
        if not holds:
            self.hold_lb.setText(f"{p['date']} 这一期没有留下持仓明细")
            self.hold_tb.setRowCount(0)
            return
        self._fill_holds(
            holds,
            f"{p['date']} 这一期持仓：{p['n']} 只，均价 {p['avg_price']:.2f}，"
            f"均溢价 {p['avg_premium']:.1f}%，当次换手 {p['turnover']:.0f}%")


# ================================================================== 多市场净值

class MultimktTab(QWidget):
    """把 A 股 / 港股 / 美股 / ETF / 可转债折算成人民币，拼成一条净值曲线。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self.result = None
        self.positions = None
        self._rows = []
        self._series = {}
        self._picked = set()
        self._base_lines = []
        self._chart_dates = []

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        bar = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["配置模拟", "当前持仓回溯"])
        self.mode.currentTextChanged.connect(lambda *_: self._on_mode())
        self.preset = QComboBox()
        self.preset.addItem("— 选个预设 —")
        for name, _mix in multimkt.presets():
            self.preset.addItem(name)
        self.preset.currentIndexChanged.connect(lambda *_: self._on_preset())
        self.start = QLineEdit(DEFAULT_START)
        self.start.setFixedWidth(90)
        self.end = QLineEdit(DEFAULT_END)
        self.end.setFixedWidth(90)
        self.pick_btn = QPushButton("读交割单持仓")
        self.pick_btn.clicked.connect(lambda *_: self.on_pick())
        self.pick_btn.setEnabled(False)
        self.btn = QPushButton("合成统一净值")
        self.btn.clicked.connect(lambda *_: self.run())

        bar.addWidget(QLabel("模式"))
        bar.addWidget(self.mode)
        bar.addWidget(self.preset)
        bar.addWidget(QLabel("区间"))
        bar.addWidget(self.start)
        bar.addWidget(QLabel("→"))
        bar.addWidget(self.end)
        bar.addWidget(self.pick_btn)
        bar.addWidget(self.btn)
        bar.addStretch(1)
        root.addLayout(bar)

        self.status = QLabel("默认按四等分配置（沪深300 / 恒生指数 / 标普500 / 中证转债），"
                             "点右上角按钮直接看结果")
        self.status.setStyleSheet("color:#888; font-size:12px;")
        root.addWidget(self.status)

        self.strip = _metric_strip([("总收益", "—", MUTED)])
        root.addWidget(self.strip)

        body = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        cfg = QHBoxLayout()
        cfg.addWidget(QLabel("标的与权重"))
        cfg.addStretch(1)
        add_btn = QPushButton("+ 加一行")
        add_btn.clicked.connect(lambda *_: self._add_row())
        del_btn = QPushButton("− 删末行")
        del_btn.clicked.connect(lambda *_: self._del_row())
        cfg.addWidget(add_btn)
        cfg.addWidget(del_btn)
        lv.addLayout(cfg)

        self.cfg_tb = QTableWidget(0, 2)
        self.cfg_tb.setHorizontalHeaderLabels(["标的", "权重%"])
        self.cfg_tb.verticalHeader().setVisible(False)
        self.cfg_tb.verticalHeader().setDefaultSectionSize(30)
        self.cfg_tb.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.cfg_tb.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Fixed)
        self.cfg_tb.setColumnWidth(1, 96)
        self.cfg_tb.setMaximumHeight(4 * 30 + 30)
        lv.addWidget(self.cfg_tb)

        self.hint = QLabel("权重会按合计归一，不必凑满 100%")
        self.hint.setStyleSheet("color:#888; font-size:11px;")
        lv.addWidget(self.hint)

        self.part_lb = QLabel("分市场拆解（点一行，把这条市场的折人民币曲线叠到图上）")
        self.part_lb.setStyleSheet("color:#444;font-weight:600;")
        lv.addWidget(self.part_lb)
        self.part_tb = _table(
            ["市场", "标的", "权重", "本币", "折人民币", "汇率", "贡献"],
            height=6 * 23 + 30,
            widths={0: 52, 2: 58, 3: 62, 4: 74, 5: 58, 6: 58}, stretch_col=1)
        self.part_tb.cellClicked.connect(self._on_part_click)
        lv.addWidget(self.part_tb, 0)

        lv.addWidget(QLabel("判读"))
        self.read = QTextBrowser()
        lv.addWidget(self.read, 1)
        body.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.chart = MultiLineChart()
        self.chart.show_placeholder("点右上角「合成统一净值」")
        rv.addWidget(self.chart, 1)
        body.addWidget(right)

        body.setSizes([620, 780])
        root.addWidget(body, 1)

        for m, code, w, name in multimkt.default_mix():
            self._add_row(m, code, w)

    # ---------------- 配置行 ----------------
    def _add_row(self, kind=None, code=None, w=None):
        if len(self._rows) >= 10:
            return
        r = self.cfg_tb.rowCount()
        self.cfg_tb.insertRow(r)
        combo = QComboBox()
        for mkt, name, k, c in multimkt.REPRESENTATIVES:
            combo.addItem(f"{mkt} · {name}", (k, c, name))
        if kind:
            for i in range(combo.count()):
                d = combo.itemData(i)
                if d and d[0] == kind and d[1] == code:
                    combo.setCurrentIndex(i)
                    break
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 100.0)
        spin.setDecimals(1)
        spin.setSingleStep(5.0)
        spin.setValue(float(w) if w is not None else 20.0)
        spin.setSuffix(" %")
        self.cfg_tb.setCellWidget(r, 0, combo)
        self.cfg_tb.setCellWidget(r, 1, spin)
        self._rows.append((combo, spin))

    def _del_row(self):
        if len(self._rows) <= 1:
            return
        self.cfg_tb.removeRow(self.cfg_tb.rowCount() - 1)
        self._rows.pop()

    def _collect_mix(self):
        out = []
        for combo, spin in self._rows:
            d = combo.currentData()
            w = spin.value()
            if d and w > 0:
                out.append((d[0], d[1], float(w), d[2]))
        return out

    def _on_preset(self):
        idx = self.preset.currentIndex()
        if idx <= 0:
            return
        _name, mix = multimkt.presets()[idx - 1]
        while self._rows:
            self._del_row()
            if len(self._rows) <= 1:
                break
        # 清空重建
        self.cfg_tb.setRowCount(0)
        self._rows.clear()
        for m, c, w, n in mix:
            self._add_row(m, c, w)
        self.status.setText(f"已载入预设「{_name}」，点「合成统一净值」")

    def _on_mode(self):
        pos = self.mode.currentText() == "当前持仓回溯"
        self.pick_btn.setEnabled(pos)
        self.cfg_tb.setEnabled(not pos)
        self.preset.setEnabled(not pos)
        if pos and not self.positions:
            self.status.setText("先点「读交割单持仓」，从成交记录里取当前持仓")
        elif pos:
            self.status.setText(f"已载入 {len(self.positions)} 只持仓，按成本市值权重回溯")

    def on_pick(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择券商交割单", os.path.expanduser("~"),
            "交割单 (*.xls *.xlsx *.csv *.txt)")
        if not path:
            return
        try:
            recs, _meta = trades_io.load_statement(path)
            res = trades_io.rebuild(recs)
            self.positions = res["open"]
            self.status.setText(f"已载入 {len(self.positions)} 只持仓，按成本市值权重回溯")
        except Exception as exc:  # noqa: BLE001
            self.status.setText("读取失败：" + str(exc).splitlines()[0])

    # ---------------- 运行 ----------------
    def run(self):
        if self._worker and self._worker.isRunning():
            return
        start = self.start.text().strip()
        end = self.end.text().strip()
        pos_mode = self.mode.currentText() == "当前持仓回溯"
        if pos_mode:
            if not self.positions:
                self.status.setText("先读交割单持仓")
                return
            args = ("pos", self.positions, start, end)
        else:
            mix = self._collect_mix()
            if not mix:
                self.status.setText("至少给一个标的分配权重")
                return
            args = ("config", mix, start, end)
        self.btn.setEnabled(True)
        self.status.setText("取行情与汇率…")
        self._worker = WorkerSig(self._job, *args)
        self._worker.tick.connect(lambda m: self.status.setText(m))
        self._worker.done.connect(self.on_done)
        self._worker.fail.connect(self.on_fail)
        self._worker.start()

    def _job(self, progress, mode, payload, start, end):
        progress("取汇率与各市场行情…")
        if mode == "pos":
            res = multimkt.from_positions(payload, start, end)
        else:
            res = multimkt.blend(payload, start, end, bench=("cn_index", "000300"))
        return res

    def on_done(self, res):
        if not res.get("ok"):
            self.status.setText("没算出来：" + res.get("msg", ""))
            return
        self.result = res
        self._fill(res)
        extra = res.get("window_note", "")
        self.status.setText(
            f"完成：{res['dates'][0]} → {res['dates'][-1]}，"
            f"{len(res['dates'])} 个交易日，{len(res['series'])} 个标的。{extra}")

    def on_fail(self, msg):
        self.status.setText("出错了：" + msg.splitlines()[0])
        print(msg)

    # ---------------- 渲染 ----------------
    def _fill(self, res):
        m, mb = res["metrics"], res["metrics_bh"]
        items = [
            ("总收益", _pct(m["total_return"]), _money_color(m["total_return"])),
            ("年化", _pct(m["annual_return"]), _money_color(m["annual_return"])),
            ("波动", f"{m['vol'] * 100:.1f}%", FG),
            ("夏普", f"{m['sharpe']:.2f}", _money_color(m["sharpe"])),
            ("最大回撤", f"{m['max_drawdown'] * 100:.1f}%", DOWN),
            ("买入不动年化", _pct(mb["annual_return"]), _money_color(mb["annual_return"])),
        ]
        fx = res.get("fx_total", 0.0)
        items.append(("汇率影响", f"{fx * 100:+.2f}pp", _money_color(fx)))
        self.strip.setText(_metric_strip(items).text())

        lines = [
            {"name": "统一净值（日频再平衡）", "values": res["nav"],
             "color": PALETTE[0], "lw": 1.9, "z": 5},
            {"name": "统一净值（买入不动）", "values": res["nav_bh"],
             "color": PALETTE[1], "lw": 1.2, "ls": "-", "z": 4},
        ]
        if res.get("bench"):
            lines.append({"name": f"参照 {res['bench_name']}", "values": res["bench"],
                          "color": "#8a8a8a", "lw": 1.0, "ls": "--", "z": 2})
        self._base_lines = lines
        self._chart_dates = list(res["dates"])
        self._picked = set()
        self.chart.set_data(res["dates"], lines)

        rows = sorted(res["series"], key=lambda r: -r["w"])
        self._series = {r["key"]: r for r in rows}
        self.part_lb.setText("分市场拆解（点一行，把这条市场的折人民币曲线叠到图上）")
        self.part_tb.setRowCount(len(rows))
        for i, r in enumerate(rows):
            _put(self.part_tb, i, 0, r["market"])
            _put(self.part_tb, i, 1, r["name"])
            _put(self.part_tb, i, 2, f"{r['w'] * 100:.1f}%", None, True)
            _put(self.part_tb, i, 3, _pct(r["ret_local"], 1),
                 _money_color(r["ret_local"]), True)
            rmb = _pct(r["ret_cny"], 1)
            _put(self.part_tb, i, 4, rmb, _money_color(r["ret_cny"]), True)
            fxv = "—" if r["ccy"] == "CNY" else _pct(r["fx"], 1)
            _put(self.part_tb, i, 5, fxv,
                 MUTED if r["ccy"] == "CNY" else _money_color(r["fx"]), True)
            _put(self.part_tb, i, 6, f"{r['contrib'] * 100:+.2f}",
                 _money_color(r["contrib"]), True)
            self.part_tb.bind(i, r["key"])

        html = []
        for v in res["verdict"]:
            html.append(f"<p style='margin:6px 0 2px 0;'><b>{v['title']}</b><br>"
                        f"<span style='color:#333;'>{v['text']}</span></p>")
        html.append("<p style='color:#888;font-size:11px;margin-top:8px;'>"
                    "港元按 USDCNH ÷ USDHKD 逐日折算，美元按 USDCNH。"
                    "各市场假期不同，曲线铺在日期并集上并对停市日向前填充，"
                    "所以停市当天的贡献记 0，波动率会略被低估。</p>")
        self.read.setHtml("".join(html))

    # ------------------------------------------------------- 分市场 → 曲线
    def _on_part_click(self, row, _col):
        if not self._series or row < 0:
            return
        key = self.part_tb.ref(row)
        if key is None:
            return
        if key in self._picked:
            self._picked.discard(key)
        else:
            self._picked.add(key)
        self._mark_picked()
        self._refresh_chart()

    def _mark_picked(self):
        for r in range(self.part_tb.rowCount()):
            key = self.part_tb.ref(r)
            on = key in self._picked
            for c in range(self.part_tb.columnCount()):
                it = self.part_tb.item(r, c)
                if it is None:
                    continue
                f = it.font()
                f.setBold(on)
                it.setFont(f)
        names = [self._series[k]["name"] for k in self._picked if k in self._series]
        self.part_lb.setText(
            "分市场拆解（图上叠加：" + "、".join(names) + "）" if names
            else "分市场拆解（点一行，把这条市场的折人民币曲线叠到图上）")

    def _refresh_chart(self):
        if not self._chart_dates or not self._base_lines:
            return
        lines = list(self._base_lines)
        for i, k in enumerate(sorted(self._picked)):
            r = self._series.get(k)
            nav = (r or {}).get("nav")
            if not nav:
                continue
            lines.append({
                "name": f"{r['name']}（折人民币）",
                "values": list(nav),
                "color": PALETTE[(i + 2) % len(PALETTE)],
                "lw": 1.5, "z": 3,
            })
        # 保持当前缩放，否则点一下就被拉回全区间
        self.chart.set_data(self._chart_dates, lines, keep_view=True)


# ================================================================== 机构持仓

class HolderTab(QWidget):
    """社保 / QFII / 险资 / 公募 / 券商 / 信托的季度重仓，外加 A+H 溢价。

    口径是季报，一年只有四个时点。看的是「上个季度末谁在买」，
    不是今天手里还拿着什么。这一点必须写在界面上，否则一定被当成实时持仓。
    """

    MAIN_COLS = ["代码", "名称", "机构数", "持股万股", "市值亿",
                 "变动万股", "变动%", "连续季", "动作"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workers = []
        self._main_loaded = False
        self._loaded = set()
        self._ps = []
        self._cur_code = None
        self._combo_token = None
        self.panel = None
        self.cross = None
        self.combos = None
        self.ah = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        bar = QHBoxLayout()
        self.kind = QComboBox()
        for k in holders.KINDS:
            self.kind.addItem(k)
        self.kind.currentTextChanged.connect(lambda *_: self.reload())
        self.period = QComboBox()
        self._periods = holders.recent_periods(5)
        for p in self._periods:
            self.period.addItem(holders.quarter_name(p), p)
        self.period.currentIndexChanged.connect(lambda *_: self.reload())
        self.btn = QPushButton("刷新")
        self.btn.clicked.connect(lambda *_: self.reload(force=True))
        bar.addWidget(QLabel("机构类型"))
        bar.addWidget(self.kind)
        bar.addWidget(QLabel("报告期"))
        bar.addWidget(self.period)
        bar.addWidget(self.btn)
        bar.addStretch(1)
        root.addLayout(bar)

        self.status = QLabel("季报口径，一年四个时点。切到这个页签自动加载最新一期")
        self.status.setStyleSheet("color:#888; font-size:12px;")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.strip = _metric_strip([("样本", "—", MUTED)])
        root.addWidget(self.strip)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab)
        root.addWidget(self.tabs, 1)

        self._build_main()
        self._build_cross()
        self._build_social()
        self._build_ah()

    # ------------------------------------------------------------ 通用执行
    def _run(self, fn, ok, fail=None):
        w = Worker(fn)
        w.done.connect(ok)
        w.fail.connect(fail or self._on_fail)
        self._workers.append(w)
        w.finished.connect(lambda *_: self._workers.remove(w)
                           if w in self._workers else None)
        w.start()

    def _on_fail(self, msg):
        self.status.setText("加载失败：" + str(msg).splitlines()[0])

    def showEvent(self, e):  # noqa: N802  (Qt 命名)
        super().showEvent(e)
        if not self._main_loaded:          # 首次切进来才拉，别拖慢启动
            self._main_loaded = True
            self.reload()

    def _on_tab(self, _i):
        name = self.tabs.tabText(self.tabs.currentIndex())
        if name == "交叉重仓" and self.cross is None:
            self._load_cross()
        elif name == "社保组合" and self.combos is None:
            self._load_social()
        elif name == "A+H 溢价" and self.ah is None:
            self._load_ah()

    # ------------------------------------------------------------ 重仓明细
    def _build_main(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(5)

        f = QHBoxLayout()
        self.f_action = QComboBox()
        for x in ("全部", "新进", "增持", "减持", "不变"):
            self.f_action.addItem(x)
        self.f_action.currentIndexChanged.connect(lambda *_: self._fill_main())
        self.f_streak = QSpinBox()
        self.f_streak.setRange(0, 5)
        self.f_streak.setSpecialValueText("不限")
        self.f_streak.setFixedWidth(58)
        self.f_streak.setToolTip("从最新一期往前数，连续增持了几个季度")
        self.f_streak.valueChanged.connect(lambda *_: self._fill_main())
        self.f_min = QDoubleSpinBox()
        self.f_min.setRange(0, 20000)
        self.f_min.setDecimals(0)
        self.f_min.setSingleStep(1)
        self.f_min.setSuffix(" 亿")
        self.f_min.setSpecialValueText("不限")
        self.f_min.setFixedWidth(86)
        self.f_min.valueChanged.connect(lambda *_: self._fill_main())
        self.f_kw = QLineEdit()
        self.f_kw.setPlaceholderText("代码 / 名称 / 关键词")
        self.f_kw.setFixedWidth(200)
        self.f_kw.textChanged.connect(lambda *_: self._fill_main())
        self.f_note = QLabel("")
        self.f_note.setStyleSheet("color:#8b96a3; font-size:11px;")

        f.addWidget(QLabel("动作"))
        f.addWidget(self.f_action)
        f.addWidget(QLabel("连续增持≥"))
        f.addWidget(self.f_streak)
        f.addWidget(QLabel("季"))
        f.addWidget(QLabel("市值≥"))
        f.addWidget(self.f_min)
        f.addWidget(self.f_kw)
        f.addWidget(self.f_note, 1)
        v.addLayout(f)

        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.main_tb = _table(
            self.MAIN_COLS, stretch_col=1,
            widths={0: 62, 2: 52, 3: 86, 4: 74, 5: 92, 6: 62, 7: 52, 8: 56})
        self.main_tb.setToolTip("点一行看右边逐季变化，双击看全部字段")
        self.main_tb.cellClicked.connect(self._on_pick)
        self.main_tb.cellDoubleClicked.connect(self._detail)
        lv.addWidget(self.main_tb, 1)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(4)
        self.hist_lb = QLabel("逐季持仓（点左边任一只票）")
        self.hist_lb.setStyleSheet("color:#444; font-weight:600;")
        self.hist_lb.setWordWrap(True)
        rv.addWidget(self.hist_lb)
        self.hist_tb = _table(
            ["季度", "机构数", "持股万股", "市值亿", "变动"],
            stretch=False, widths={0: 66, 1: 58, 2: 88, 3: 80, 4: 84}, row_h=21)
        self.hist_tb.setMaximumHeight(190)
        rv.addWidget(self.hist_tb, 1)

        self.combo_lb = QLabel("社保组合明细")
        self.combo_lb.setStyleSheet("color:#444; font-weight:600;")
        self.combo_lb.setWordWrap(True)
        rv.addWidget(self.combo_lb)
        self.combo_tb = _table(
            ["组合", "持股万股", "市值亿", "本季"],
            widths={1: 78, 2: 70}, stretch_col=0, row_h=21)
        self.combo_tb.setMaximumHeight(170)
        rv.addWidget(self.combo_tb, 1)
        rv.addStretch(1)
        split.addWidget(right)
        split.setSizes([790, 420])

        v.addWidget(split, 1)
        self.tabs.addTab(w, "重仓明细")

    def _periods_for(self, sel: str) -> list[str]:
        """选定报告期往前数，连续季数的含义才跟着对。"""
        ps = [p for p in self._periods if p <= sel]
        return ps or [sel]

    def reload(self, force=False):
        kind = self.kind.currentText()
        sel = self.period.currentData()
        if not sel:
            return
        self._ps = self._periods_for(sel)
        # 换机构/换报告期时必须先把旧表清掉。留着上一类机构的数字，
        # 等新数据回来的那几秒里，界面上是「QFII」标题配「社保」数据。
        self.panel = None
        self.main_tb.setRowCount(0)
        self.hist_tb.setRowCount(0)
        self.combo_tb.setRowCount(0)
        self.f_note.setText("")
        self.strip.setText(_metric_strip([("加载中", "…", MUTED)]).text())
        self.status.setText(
            f"正在拉 {kind} 最近 {len(self._ps)} 个季度的持仓…"
            "（首次要几秒，之后走缓存）")
        self.btn.setEnabled(False)
        self._run(lambda: holders.panel(kind, self._ps, force=force),
                  self._on_main, self._reload_fail)

    def _reload_fail(self, msg):
        self.btn.setEnabled(True)
        self._on_fail(msg)

    def _on_main(self, df):
        self.btn.setEnabled(True)
        self.panel = df
        kind = self.kind.currentText()
        q = holders.quarter_name(self._ps[0]) if self._ps else "—"
        if df is None or not len(df):
            self.status.setText(f"{kind} {q} 没有数据")
            self.strip.setText(_metric_strip([("样本", "0 只", MUTED)]).text())
            self.main_tb.setRowCount(0)
            return
        s = holders.summary(df)
        self.strip.setText(_metric_strip([
            ("样本", f"{s['n']} 只", FG),
            ("合计市值", f"{s['value'] / 1e8:,.0f} 亿", FG),
            ("新进", f"{s['new']}", UP),
            ("增持", f"{s['add']}", UP),
            ("减持", f"{s['cut']}", DOWN),
            ("连续2季增持", f"{s['streak2']} 只", UP),
            ("连续3季增持", f"{s['streak3']} 只", UP),
        ]).text())
        cov = int((df["n_period"] >= len(self._ps)).sum())
        self.status.setText(
            f"{kind} · {q}｜合计持股市值 {s['value'] / 1e8:,.0f} 亿元，"
            f"{s['n']} 只票有披露。市值按报告期末收盘价算，不是今天的价。"
            f"｜{len(self._ps)} 个季度里全程有数据的 {cov} 只，其余是中途进的前十大")
        self.tabs.setTabText(0, f"重仓明细 {s['n']}")
        self._fill_main()
        self.hist_tb.setRowCount(0)
        self.combo_tb.setRowCount(0)
        self.hist_lb.setText("逐季持仓（点左边任一只票）")
        self.combo_lb.setText("社保组合明细（只有社保能拆到具体组合）")

    def _view(self):
        d = self.panel
        if d is None or not len(d):
            return d
        act = self.f_action.currentText()
        if act != "全部":
            d = d[d["change"] == act]
        n = self.f_streak.value()
        if n > 0:
            d = d[d["streak"] >= n]
        mv = self.f_min.value()
        if mv > 0:
            d = d[d["value"] >= mv * 1e8]
        kw = self.f_kw.text().strip()
        if kw:
            hit = (d["code"].str.contains(kw, na=False) |
                   d["name"].str.contains(kw, na=False))
            d = d[hit]
        return d.sort_values("value", ascending=False)

    def _fill_main(self):
        d = self._view()
        if d is None:
            return
        if not len(d):
            self.main_tb.setRowCount(0)
            self.f_note.setText("当前筛选条件下没有标的")
            return
        k = self.kind.currentText()
        self.f_note.setText(
            f"筛出 {len(d)} 只 / 共 {len(self.panel)} 只"
            f"｜连续增持不含「这一季才第一次进」的")
        self.main_tb.setRowCount(len(d))
        for i, (_, r) in enumerate(d.iterrows()):
            chg = r["change"]
            col = UP if chg in ("新进", "增持") else (DOWN if chg == "减持" else MUTED)
            _put(self.main_tb, i, 0, r["code"], MUTED, True)
            _put(self.main_tb, i, 1, r["name"])
            _put(self.main_tb, i, 2, _n(r["n_inst"]), None, True)
            _put(self.main_tb, i, 3, _n(_fnum(r["shares"]) / 1e4, "{:,.1f}"), None, True)
            _put(self.main_tb, i, 4, _n(_fnum(r["value"]) / 1e8, "{:,.2f}"), None, True)
            chg_sh = _fnum(r["chg_shares"]) / 1e4
            _put(self.main_tb, i, 5, _n(chg_sh, "{:+,.1f}"), _col_of(chg_sh), True)
            _put(self.main_tb, i, 6, _n(r["chg_pct"], "{:+.1f}"), None, True)
            st = int(r["streak"]) if pd.notna(r["streak"]) else 0
            _put(self.main_tb, i, 7, "—" if st <= 0 else f"{st} 季",
                 UP if st >= 3 else (None if st >= 2 else MUTED), True)
            _put(self.main_tb, i, 8, chg, col, True)
            self.main_tb.bind(i, r["code"])

    def _on_pick(self, row, _col):
        code = self.main_tb.ref(row)
        if not code:
            self.hist_tb.setRowCount(0)
            return
        self._cur_code = code
        name = ""
        if self.panel is not None and len(self.panel):
            hit = self.panel[self.panel["code"] == code]
            if len(hit):
                name = hit.iloc[0]["name"]
        self.hist_lb.setText(f"{name}（{code}）逐季被 {self.kind.currentText()} 拿了多少")
        h = holders.history(code, self.kind.currentText(), self._ps)
        self.hist_tb.setRowCount(len(h))
        for i, (_, r) in enumerate(h.iterrows()):
            chg = r["change"]
            col = UP if chg in ("新进", "增持") else (DOWN if chg == "减持" else MUTED)
            _put(self.hist_tb, i, 0, r["季度"], None, True)
            _put(self.hist_tb, i, 1, str(int(r["n_inst"])), None, True)
            _put(self.hist_tb, i, 2, _n(r["shares"] / 1e4, "{:,.1f}"), None, True)
            _put(self.hist_tb, i, 3, _n(r["value"] / 1e8, "{:,.2f}"), None, True)
            _put(self.hist_tb, i, 4, chg, col, True)

        if self.kind.currentText() == "社保":
            self._fill_combos(code, name)
        else:
            self.combo_lb.setText(
                f"{self.kind.currentText()}只有合并数，没有组合级披露")
            self.combo_tb.setRowCount(0)

    def _fill_combos(self, code, name):
        self._combo_token = code
        if self.combos is not None:
            self._show_combos(code, name)
            return
        self.combo_lb.setText("社保组合明细（首次要拉全市场十大流通股东，约 15 秒）")
        self.combo_tb.setRowCount(0)
        self._load_social(then=lambda: self._show_combos(code, name))

    def _show_combos(self, code, name):
        if self._combo_token != code:
            return                      # 等的时候用户已经点别的票了
        d = self.combos
        if d is None or not len(d):
            self.combo_lb.setText("没有拿到社保组合数据")
            self.combo_tb.setRowCount(0)
            return
        sub = d[d["code"] == code].sort_values("value", ascending=False)
        if not len(sub):
            self.combo_lb.setText(f"{name}（{code}）不在十大流通股东的社保名单里")
            self.combo_tb.setRowCount(0)
            return
        total = float(sub["value"].fillna(0).sum())
        self.combo_lb.setText(
            f"{name}（{code}）有 {len(sub)} 只社保组合持有，"
            f"合计 {total / 1e8:,.2f} 亿元")
        self.combo_tb.setRowCount(len(sub))
        for i, (_, r) in enumerate(sub.iterrows()):
            chg = r["change"]
            col = UP if chg in ("新进", "增持") else (DOWN if chg == "减持" else MUTED)
            _put(self.combo_tb, i, 0, r["combo"], None, False, tooltip=str(r["manager"]))
            _put(self.combo_tb, i, 1, _n(_fnum(r["shares"]) / 1e4, "{:,.1f}"), None, True)
            _put(self.combo_tb, i, 2, _n(_fnum(r["value"]) / 1e8, "{:,.2f}"), None, True)
            _put(self.combo_tb, i, 3, chg, col, True)

    # ------------------------------------------------------------ 详情
    def _detail(self, row, _col):
        code = self.main_tb.ref(row)
        if not code:
            return
        d = self._view()
        hit = d[d["code"] == code] if d is not None and len(d) else d
        if hit is None or not len(hit):
            return
        r = hit.iloc[0]
        k = self.kind.currentText()
        q = holders.quarter_name(r["period"])
        net = _fnum(r["chg_shares"])
        fields = [
            ("代码", r["code"]), ("名称", r["name"]),
            ("机构类型", k), ("报告期", q),
            ("持有机构数", f"{int(_fnum(r['n_inst']))} 家"),
            ("持股总数", f"{_fnum(r['shares']) / 1e4:,.2f} 万股"),
            ("持股市值", f"{_fnum(r['value']) / 1e8:,.2f} 亿元"),
            ("本季动作", r["change"]),
            ("本季变动", f"{net / 1e4:+,.2f} 万股"),
            ("变动比例", _n(r["chg_pct"], "{:+.2f}%")),
            ("连续增持", f"{int(r['streak'])} 个季度"),
            ("有几季披露", f"{int(r['n_period'])} / {len(self._ps)}"),
        ]
        tables = []
        h = holders.history(code, k, self._ps)
        if len(h):
            rows = [[x["季度"], str(int(x["n_inst"])),
                     f"{x['shares'] / 1e4:,.2f}", f"{x['value'] / 1e8:,.2f}",
                     x["change"]] for _, x in h.iterrows()]
            tables.append({"caption": "逐季持仓（新到旧）",
                           "headers": ["季度", "机构数", "持股万股", "市值亿", "动作"],
                           "rows": rows, "right": [1, 2, 3, 4]})
        if k == "社保" and self.combos is not None and len(self.combos):
            sub = self.combos[self.combos["code"] == code]
            if len(sub):
                rows = [[x["combo"], x["manager"],
                         f"{_fnum(x['shares']) / 1e4:,.2f}",
                         f"{_fnum(x['value']) / 1e8:,.2f}", x["change"]]
                        for _, x in sub.sort_values("value", ascending=False).iterrows()]
                tables.append({"caption": f"谁在拿：{len(sub)} 只社保组合",
                               "headers": ["组合", "管理人", "持股万股", "市值亿", "动作"],
                               "rows": rows, "right": [2, 3, 4]})
        DetailDialog(
            self, title=f"{r['name']}（{r['code']}）· {k} {q}",
            subtitle=f"季报快照，市值按 {q} 期末收盘价算。"
                     f"这一季{net / 1e4:+,.1f} 万股，累计连续增持 {int(r['streak'])} 季",
            fields=fields, tables=tables,
            copy_text=f"{r['name']}({r['code']}) {k}{q} 持股"
                      f"{_fnum(r['shares']) / 1e4:,.2f}万股 "
                      f"市值{_fnum(r['value']) / 1e8:,.2f}亿 {r['change']}").exec()

    # ------------------------------------------------------------ 交叉重仓
    def _build_cross(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        note = QLabel(
            "同一个季度里被越多类型机构同时持有的票，逻辑越不依赖单一买方的偏好。"
            "双击看它被哪几类分别拿了多少。"
            "注意：社保有一部分委托公募管理，各类市值直接加总会重复计算，"
            "只能看量级，不能当真实持仓。信托样本少（一百多只），所以六类全覆盖几乎不存在。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(note)
        self.cross_note = QLabel("")
        self.cross_note.setStyleSheet("color:#444; font-weight:600;")
        v.addWidget(self.cross_note)
        self.cross_tb = _table(
            ["代码", "名称", "类别数", "机构类型", "合计市值亿", "合计万股", "机构家数"],
            stretch=False,
            widths={0: 82, 1: 290, 2: 72, 3: 250, 4: 110, 5: 100, 6: 90})
        self.cross_tb.setToolTip("点表头按类别数排；双击看各类分别拿了多少")
        self.cross_tb.cellDoubleClicked.connect(self._cross_detail)
        v.addWidget(self.cross_tb, 1)
        self.tabs.addTab(w, "交叉重仓")

    def _load_cross(self, force=False):
        sel = self.period.currentData()
        self.cross_note.setText("正在拉六类机构同一季度的持仓…")
        self._run(lambda: holders.cross_hold(sel, force=force), self._on_cross,
                  self._on_cross_fail)

    def _on_cross_fail(self, msg):
        self.cross_note.setText("加载失败：" + str(msg).splitlines()[0])

    def _on_cross(self, df):
        self.cross = df
        if df is None or not len(df):
            self.cross_note.setText("没有数据")
            self.cross_tb.setRowCount(0)
            return
        q = holders.quarter_name(self.period.currentData())
        multi = df[df["n_kind"] >= 3]
        self.cross_note.setText(
            f"{q}｜{len(df)} 只票有机构持仓，其中被 3 类以上同时持有的 {len(multi)} 只，"
            f"被 5 类持有的 {int((df['n_kind'] >= 5).sum())} 只")
        self.cross_tb.setRowCount(len(df))
        for i, (_, r) in enumerate(df.iterrows()):
            _put(self.cross_tb, i, 0, r["code"], MUTED, True)
            _put(self.cross_tb, i, 1, r["name"])
            k = int(r["n_kind"])
            _put(self.cross_tb, i, 2, str(k), UP if k >= 5 else (None if k >= 3 else MUTED), True)
            _put(self.cross_tb, i, 3, r["kinds"], None, False, tooltip=str(r["kinds"]))
            _put(self.cross_tb, i, 4, _n(_fnum(r["value"]) / 1e8, "{:,.1f}"), None, True)
            _put(self.cross_tb, i, 5, _n(_fnum(r["shares"]) / 1e4, "{:,.0f}"), None, True)
            _put(self.cross_tb, i, 6, _n(r["n_inst"]), None, True)
            self.cross_tb.bind(i, r["code"])
        # 默认按类别数排：先看被最多类机构同时持有的
        self.cross_tb.sort_by(2, True)

    def _cross_detail(self, row, _col):
        code = self.cross_tb.ref(row)
        if not code or self.cross is None:
            return
        hit = self.cross[self.cross["code"] == code]
        if not len(hit):
            return
        r = hit.iloc[0]
        q = holders.quarter_name(self.period.currentData())
        rows = []
        for k in holders.KINDS:
            f = holders.load(k, self.period.currentData())
            if not len(f):
                continue
            sub = f[f["code"] == code]
            if not len(sub):
                continue
            x = sub.iloc[0]
            rows.append([k, str(int(_fnum(x["n_inst"]))),
                         f"{_fnum(x['shares']) / 1e4:,.2f}",
                         f"{_fnum(x['value']) / 1e8:,.2f}", x["change"]])
        DetailDialog(
            self, title=f"{r['name']}（{code}）被 {int(r['n_kind'])} 类机构持有",
            subtitle=f"{q}｜{r['kinds']}｜各类市值加总 {_fnum(r['value']) / 1e8:,.1f} 亿元"
                     "（含重复计算，只看量级）",
            fields=[("代码", r["code"]), ("名称", r["name"]),
                    ("机构类别数", str(int(r["n_kind"]))),
                    ("类别", r["kinds"]),
                    ("合计市值", f"{_fnum(r['value']) / 1e8:,.1f} 亿元")],
            tables=[{"caption": "各类机构分别拿了多少",
                     "headers": ["机构类型", "机构数", "持股万股", "市值亿", "本季动作"],
                     "rows": rows, "right": [1, 2, 3, 4]}]).exec()

    # ------------------------------------------------------------ 社保组合
    def _build_social(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        note = QLabel(
            "机构持仓那张表只有合并数，拆不到具体是哪个组合。这里来自十大流通股东明细，"
            "能看到「全国社保基金一一四组合」这种颗粒度。"
            "同一只票常同时被两三只组合拿着，所以明细条数比股票只数多："
            "七百多条明细对应六百出头的票。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(note)
        self.soc_note = QLabel("")
        self.soc_note.setStyleSheet("color:#444; font-weight:600;")
        v.addWidget(self.soc_note)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("组合排行（点一行看它拿了哪些票）"))
        self.soc_tb = _table(
            ["组合", "持股票数", "合计市值亿", "净买万股", "增持笔", "减持笔"],
            stretch_col=0, widths={1: 66, 2: 88, 3: 84, 4: 62, 5: 62})
        self.soc_tb.setToolTip("双击看这个组合的全部持仓")
        self.soc_tb.cellClicked.connect(self._on_combo_pick)
        self.soc_tb.cellDoubleClicked.connect(self._combo_detail)
        lv.addWidget(self.soc_tb, 1)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.soc_sub_lb = QLabel("这个组合的持仓")
        self.soc_sub_lb.setStyleSheet("color:#444; font-weight:600;")
        self.soc_sub_lb.setWordWrap(True)
        rv.addWidget(self.soc_sub_lb)
        self.soc_sub_tb = _table(
            ["代码", "名称", "持股万股", "市值亿", "本季"],
            widths={0: 64, 2: 82, 3: 74}, stretch_col=1)
        rv.addWidget(self.soc_sub_tb, 1)
        split.addWidget(right)
        split.setSizes([560, 620])
        v.addWidget(split, 1)
        self.tabs.addTab(w, "社保组合")

    def _load_social(self, then=None, force=False):
        self.soc_note.setText("正在拉全市场十大流通股东，约 15 秒…")
        self._run(lambda: holders.social_combos(self.period.currentData(), force=force),
                  lambda d: self._on_social(d, then), self._on_cross_fail)

    def _on_social(self, df, then=None):
        self.combos = df
        if df is None or not len(df):
            self.soc_note.setText("没有拿到社保组合数据")
            self.soc_tb.setRowCount(0)
            return
        rank = holders.combo_rank(self.period.currentData())
        q = holders.quarter_name(self.period.currentData())
        self.soc_note.setText(
            f"{q}｜{df['combo'].nunique()} 只社保组合持有 {df['code'].nunique()} 只票，"
            f"合计 {df['value'].fillna(0).sum() / 1e8:,.0f} 亿元。"
            "同一个组合可以有多个管理人，名字里带基金公司的就是委托出去的那部分")
        self.soc_tb.setRowCount(len(rank))
        for i, (_, r) in enumerate(rank.iterrows()):
            _put(self.soc_tb, i, 0, r["combo"])
            _put(self.soc_tb, i, 1, str(int(r["n_stock"])), None, True)
            _put(self.soc_tb, i, 2, _n(_fnum(r["value"]) / 1e8, "{:,.2f}"), None, True)
            net = _fnum(r["chg_shares"]) / 1e4
            _put(self.soc_tb, i, 3, _n(net, "{:+,.0f}"), _col_of(net), True)
            _put(self.soc_tb, i, 4, str(int(r["add"])), UP, True)
            _put(self.soc_tb, i, 5, str(int(r["cut"])), DOWN, True)
            self.soc_tb.bind(i, r["combo"])
        if then is not None:
            then()

    def _combo_rows(self, combo):
        d = self.combos
        if d is None or not len(d):
            return d
        return d[d["combo"] == combo].sort_values("value", ascending=False)

    def _on_combo_pick(self, row, _col):
        combo = self.soc_tb.ref(row)
        if not combo:
            return
        sub = self._combo_rows(combo)
        self.soc_sub_lb.setText(f"{combo}：{len(sub)} 只票")
        self.soc_sub_tb.setRowCount(len(sub))
        for i, (_, r) in enumerate(sub.iterrows()):
            chg = r["change"]
            col = UP if chg in ("新进", "增持") else (DOWN if chg == "减持" else MUTED)
            _put(self.soc_sub_tb, i, 0, r["code"], MUTED, True)
            _put(self.soc_sub_tb, i, 1, r["name"])
            _put(self.soc_sub_tb, i, 2, _n(_fnum(r["shares"]) / 1e4, "{:,.1f}"), None, True)
            _put(self.soc_sub_tb, i, 3, _n(_fnum(r["value"]) / 1e8, "{:,.2f}"), None, True)
            _put(self.soc_sub_tb, i, 4, chg, col, True)

    def _combo_detail(self, row, _col):
        combo = self.soc_tb.ref(row)
        if not combo:
            return
        sub = self._combo_rows(combo)
        if sub is None or not len(sub):
            return
        rows = [[x["code"], x["name"], f"{_fnum(x['shares']) / 1e4:,.2f}",
                 f"{_fnum(x['value']) / 1e8:,.2f}", x["change"],
                 _n(x["chg_pct"], "{:+.2f}%")]
                for _, x in sub.iterrows()]
        net = float(sub["chg_shares"].fillna(0).sum()) / 1e4
        DetailDialog(
            self, title=combo,
            subtitle=f"{len(sub)} 只票｜合计 {_fnum(sub['value'].sum()) / 1e8:,.2f} 亿元"
                     f"｜本季净买 {net:+,.0f} 万股",
            tables=[{"caption": "这个组合的全部持仓",
                     "headers": ["代码", "名称", "持股万股", "市值亿", "本季", "变动比例"],
                     "rows": rows, "right": [2, 3, 4, 5]}]).exec()

    # ------------------------------------------------------------ A+H 溢价
    def _build_ah(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        note = QLabel(
            "北向（陆股通）的个股持股明细从 2024-08-18 起交易所停止披露，"
            "港资每天买了什么已经查不到，任何软件都给不出日频数据。"
            "还能看的是两地价差：同一家公司 A 股比 H 股贵多少。"
            "正溢价是 A 股贵，负溢价是 A 股比 H 股还便宜。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(note)
        self.ah_note = QLabel("")
        self.ah_note.setStyleSheet("color:#444; font-weight:600;")
        v.addWidget(self.ah_note)
        self.ah_tb = _table(
            ["A股代码", "名称", "H股代码", "H价(HKD)", "A价(RMB)",
             "比价", "溢价%", "H涨跌%", "A涨跌%"],
            stretch=False,
            widths={0: 92, 1: 268, 2: 92, 3: 100, 4: 100, 5: 70,
                    6: 92, 7: 92, 8: 92})
        self.ah_tb.setToolTip("点表头按溢价率排；溢价 = A 股相对 H 股贵多少")
        self.ah_tb.cellDoubleClicked.connect(self._ah_detail)
        v.addWidget(self.ah_tb, 1)
        self.tabs.addTab(w, "A+H 溢价")

    def _load_ah(self, force=False):
        self.ah_note.setText("正在拉 A+H 两地报价…")
        self._run(lambda: holders.ah_premium(force=force), self._on_ah,
                  self._on_cross_fail)

    def _on_ah(self, df):
        self.ah = df
        if df is None or not len(df):
            self.ah_note.setText("没有数据")
            self.ah_tb.setRowCount(0)
            return
        med = float(df["premium"].median())
        neg = int((df["premium"] < 0).sum())
        self.ah_note.setText(
            f"{len(df)} 只 A+H｜溢价中位数 {med:.1f}%｜A 股比 H 股便宜的 {neg} 只。"
            "比价是 A 股价格除以 H 股价格（已折成同一货币）")
        self.ah_tb.setRowCount(len(df))
        for i, (_, r) in enumerate(df.iterrows()):
            p = _fnum(r["premium"])
            col = UP if p > 0 else (DOWN if p < 0 else MUTED)
            _put(self.ah_tb, i, 0, r["a_code"], MUTED, True)
            _put(self.ah_tb, i, 1, r["name"])
            _put(self.ah_tb, i, 2, r["h_code"], MUTED, True)
            _put(self.ah_tb, i, 3, _n(r["h_px"], "{:,.2f}"), None, True)
            _put(self.ah_tb, i, 4, _n(r["a_px"], "{:,.2f}"), None, True)
            _put(self.ah_tb, i, 5, _n(r["ratio"], "{:.2f}"), None, True)
            _put(self.ah_tb, i, 6, _n(p, "{:+.1f}"), col, True)
            hc, ac = _fnum(r["h_chg"]), _fnum(r["a_chg"])
            _put(self.ah_tb, i, 7, _n(hc, "{:+.2f}"), _col_of(hc), True)
            _put(self.ah_tb, i, 8, _n(ac, "{:+.2f}"), _col_of(ac), True)
            self.ah_tb.bind(i, r["a_code"])

    def _ah_detail(self, row, _col):
        code = self.ah_tb.ref(row)
        if not code or self.ah is None:
            return
        hit = self.ah[self.ah["a_code"] == code]
        if not len(hit):
            return
        r = hit.iloc[0]
        p = _fnum(r["premium"])
        tables = []
        if self.panel is not None and len(self.panel):
            sub = self.panel[self.panel["code"] == code]
            if len(sub):
                x = sub.iloc[0]
                tables.append({
                    "caption": f"同期 {self.kind.currentText()}持仓",
                    "headers": ["机构数", "持股万股", "市值亿", "本季动作", "连续季"],
                    "rows": [[str(int(_fnum(x["n_inst"]))),
                              f"{_fnum(x['shares']) / 1e4:,.2f}",
                              f"{_fnum(x['value']) / 1e8:,.2f}",
                              x["change"], f"{int(x['streak'])}"]],
                    "right": [0, 1, 2, 4]})
        DetailDialog(
            self, title=f"{r['name']}　A {r['a_code']} / H {r['h_code']}",
            subtitle=(f"A 股比 H 股贵 {p:.1f}%" if p >= 0
                      else f"A 股比 H 股便宜 {abs(p):.1f}%"),
            fields=[("A 股代码", r["a_code"]), ("A 股价格", f"{_fnum(r['a_px']):,.2f} 元"),
                    ("H 股代码", r["h_code"]), ("H 股价格", f"{_fnum(r['h_px']):,.2f} 港元"),
                    ("比价", f"{_fnum(r['ratio']):.2f}"),
                    ("溢价", f"{p:+.1f}%"),
                    ("A 股当日", _n(r["a_chg"], "{:+.2f}%")),
                    ("H 股当日", _n(r["h_chg"], "{:+.2f}%"))],
            tables=tables,
            copy_text=f"{r['name']} A{r['a_code']}/H{r['h_code']} 溢价{p:+.1f}%").exec()
