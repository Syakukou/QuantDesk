"""QuantDesk —— 自研 A 股 看盘 + 回测 桌面程序（PyQt6 外壳）。

GUI 用 PyQt6（独立窗口）；图表用 matplotlib 的 QtAgg 后端，自绘蜡烛图。
数据全部来自 akshare / 东方财富公开接口（免费、免 token），本地运行。

看盘能力：
    · 周期：分时 / 5分钟 / 15分钟 / 30分钟 / 60分钟 / 日K / 周K / 月K
    · 复权：前复权 / 不复权
    · 交互：滚轮缩放、按住拖动平移、双击复位、鼠标悬停显示十字光标与 OHLC 数值
    · 股票池：内置 29 只，支持按代码或名称搜索任意 A 股添加，自动持久化

运行（开发态）：
    cd QuantDesk && source .venv/Scripts/activate && python main.py

无界面自检（验证数据/回测链路，打包后也能用）：
    python main.py --selftest
"""
from __future__ import annotations

import sys
import os
import logging
import argparse
import traceback
import tempfile
import datetime

# ---------------------------------------------------------------------------
# 启动痕迹日志：Qt 找不到平台插件时会在 C++ 层直接 abort，绕过所有 try/except，
# 所以必须在每一步之前先把痕迹落盘，才能知道到底死在哪。
# ---------------------------------------------------------------------------
_TRACE_ON = "--selftest" not in sys.argv


def _trace(msg):
    if not _TRACE_ON:
        return
    try:
        base = os.path.join(os.path.expanduser("~"), "Desktop")
        if not os.path.isdir(base):
            base = tempfile.gettempdir()
        with open(os.path.join(base, "QuantDesk_trace.log"), "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg))
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass


_trace("=== 进程启动 argv=%r frozen=%r exe=%r cwd=%r ==="
       % (sys.argv, getattr(sys, "frozen", False), sys.executable, os.getcwd()))


# ---------------------------------------------------------------------------
# 黑匣子 1：faulthandler 抓 C 层致命信号（abort / 访问违例 / 段错误）
# ---------------------------------------------------------------------------
def _diag_dir():
    base = os.path.join(os.path.expanduser("~"), "Desktop")
    return base if os.path.isdir(base) else tempfile.gettempdir()


def _open_diag(name):
    try:
        return open(os.path.join(_diag_dir(), name), "a", buffering=1,
                    encoding="utf-8", errors="replace")
    except Exception:
        return None


_FAULT_FH = _open_diag("QuantDesk_fault.log")
if _TRACE_ON and _FAULT_FH is not None:
    try:
        _trace("fault log -> %s" % _FAULT_FH.name)
        _FAULT_FH.write("\n===== 新进程 %s =====\n" % datetime.datetime.now())
        import faulthandler
        faulthandler.enable(_FAULT_FH, all_threads=True)
        _trace("faulthandler 已启用")
    except Exception:
        _trace("faulthandler 启用失败:\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# 黑匣子 2：主线程 / 子线程未捕获异常钩子
# ---------------------------------------------------------------------------
import threading


def _excepthook(exc_type, exc, tb):
    _trace("!!! 未捕获异常(主线程):\n" + "".join(
        traceback.format_exception(exc_type, exc, tb)))


def _thread_excepthook(args):
    _trace("!!! 未捕获异常(线程 %s):\n%s" % (
        getattr(args.thread, "name", "?"),
        "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))))


sys.excepthook = _excepthook
threading.excepthook = _thread_excepthook

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
# 中文字体：否则图表标题/图例会显示成方块（DejaVu Sans 无 CJK 字形）
# 三个平台的中文字体名都不一样，按平台给候选，后面的兜底放通用名。
_CJK_FONTS = {
    "win32": ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"],
    "darwin": ["PingFang SC", "Hiragino Sans GB", "Heiti SC",
               "Songti SC", "Arial Unicode MS", "DejaVu Sans"],
}
try:
    matplotlib.rcParams["font.sans-serif"] = _CJK_FONTS.get(
        sys.platform, ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"])
    matplotlib.rcParams["axes.unicode_minus"] = False
    _trace("中文字体已配置: %r" % (matplotlib.rcParams["font.sans-serif"][:2],))
except Exception:
    _trace("中文字体配置失败:\n" + traceback.format_exc())
_trace("matplotlib OK %s" % matplotlib.__version__)

import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
_trace("matplotlib QtAgg 后端 OK")

from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTableWidget, QTableWidgetItem, QPushButton, QLabel, QComboBox,
    QLineEdit, QSpinBox, QHeaderView, QMessageBox, QSplitter, QInputDialog,
    QDialog, QDialogButtonBox, QFileDialog,
)
_trace("PyQt6 导入 OK")

import tablekit
from tablekit import SortTable, CellItem

# ---------------------------------------------------------------------------
# 黑匣子 3：Qt 消息钩子。Qt 在 abort 之前通常会先打一条致命警告，先记下来。
# ---------------------------------------------------------------------------
try:
    from PyQt6.QtCore import qInstallMessageHandler

    def _qt_msg_handler(mode, ctx, msg):
        _trace("Qt[%s] %s" % (mode, msg))

    qInstallMessageHandler(_qt_msg_handler)
    _trace("Qt 消息钩子已安装")
except Exception:
    _trace("Qt 消息钩子安装失败:\n" + traceback.format_exc())

import pandas as pd

from config import (
    DEFAULT_START, DEFAULT_END, INIT_CASH, WATCHLIST,
    load_watchlist, save_watchlist, reset_watchlist,
)
import data_feed as feed
from data_feed import get_bars, get_quotes, search_stocks, MIN_PERIODS, ALL_PERIODS
from backtest import ma_cross_signal, momentum_signal, run_backtest
import trades_io
_trace("本地模块(config/data_feed/backtest/trades_io) OK")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("quant_desk")

UP = "#d13b3b"        # 涨：红（A股习惯）
DOWN = "#1e9e64"      # 跌：绿
GREY = "#7f8c8d"
FG = "#1f2d3d"
GRID = "#eef1f5"


def _num(text):
    """把 '1.23' / '-0.45%' 这类文本解析成颜色：涨红、跌绿、零/无效返回 None。"""
    try:
        s = str(text).strip().replace("%", "").replace(",", "")
        if not s:
            return None
        v = float(s)
    except (TypeError, ValueError):
        return None
    if v != v:          # NaN
        return None
    if v > 0:
        return UP
    if v < 0:
        return DOWN
    return None


def _guard(fn):
    """槽函数护栏：PyQt6 里槽函数抛异常会直接 abort 整个进程，这里统一兜住。"""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            tb = traceback.format_exc()
            _trace("槽函数 %s 异常:\n%s" % (getattr(fn, "__name__", fn), tb))
            print("槽函数异常:\n" + tb, file=sys.stderr)
            return None
    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    return wrapper


def _fmt_amount(v):
    """成交额格式化：亿/万。"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if v != v:
        return ""
    if abs(v) >= 1e8:
        return f"{v / 1e8:.2f}亿"
    if abs(v) >= 1e4:
        return f"{v / 1e4:.0f}万"
    return f"{v:.0f}"


def _fmt_vol(v):
    """成交量（手）格式化。"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if v != v:
        return ""
    if abs(v) >= 1e8:
        return f"{v / 1e8:.2f}亿手"
    if abs(v) >= 1e4:
        return f"{v / 1e4:.1f}万手"
    return f"{v:.0f}手"


# --------------------------------------------------------------------------- #
# 后台任务：QThread 跑网络/计算，信号回主线程更新 UI
# --------------------------------------------------------------------------- #
class Worker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn = fn
        self.args = args

    def run(self):
        try:
            self.finished.emit(self.fn(*self.args))
        except Exception as exc:  # noqa: BLE001
            logger.exception("worker error")
            self.error.emit(str(exc))


# --------------------------------------------------------------------------- #
# 蜡烛图控件：OHLC 蜡烛 + MA + 成交量 + 十字光标取值 + 滚轮缩放 + 拖动平移
# --------------------------------------------------------------------------- #
class KLineChart(QWidget):
    MIN_SPAN = 12.0        # 最多放大到可见 12 根

    def __init__(self, parent=None):
        super().__init__(parent)
        self.df: pd.DataFrame | None = None
        self.dates = None
        self.x = None
        self.is_minute = False
        self.period = "日K"
        self._bg = None
        self._drag = None

        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor="#ffffff")
        self.canvas = FigureCanvas(self.fig)
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2, sharex=self.ax1)
        self.fig.subplots_adjust(left=0.062, right=0.995, top=0.945,
                                 bottom=0.085, hspace=0.055)
        self._style_axes()

        # 十字光标对象（每次重绘后重建）
        self.v1 = self.v2 = self.h1 = self.info = None

        hint = QLabel("滚轮缩放 · 按住拖动平移 · 双击复位 · 鼠标悬停看数值")
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

    # ---------------- 基础样式 ----------------
    def _style_axes(self):
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor("#ffffff")
            ax.grid(True, color=GRID, lw=0.6)
            ax.tick_params(labelsize=8, colors="#55606b", length=3)
            for sp in ax.spines.values():
                sp.set_color("#d7dce2")

    # ---------------- 占位 ----------------
    def show_placeholder(self, text):
        self.df = None
        self._bg = None
        self.ax1.cla()
        self.ax2.cla()
        self._style_axes()
        for ax in (self.ax1, self.ax2):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        self.ax1.text(0.5, 0.5, text, ha="center", va="center",
                      transform=self.ax1.transAxes, fontsize=12, color="#98a2ae")
        if self.isVisible():
            self._draw_now()

    # ---------------- 数据 ----------------
    def set_data(self, df: pd.DataFrame, title: str, period: str):
        if df is None or df.empty:
            self.show_placeholder("无数据")
            return
        self.df = df.reset_index(drop=True)
        self.dates = self.df["date"]
        self.x = np.arange(len(self.df), dtype=float)
        self.is_minute = period in MIN_PERIODS
        self.period = period
        self._render(title)
        self._reset_view()
        self._autoscale_y()
        self._draw_now()
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)

    def _render(self, title):
        df = self.df
        x = self.x
        ax1, ax2 = self.ax1, self.ax2
        ax1.cla()
        ax2.cla()
        self._style_axes()

        o = df["open"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        up = c >= o

        # 影线（上下影）
        ax1.vlines(x[up], l[up], h[up], color=UP, lw=0.85, zorder=2)
        ax1.vlines(x[~up], l[~up], h[~up], color=DOWN, lw=0.85, zorder=2)

        # 实体。十字星（开=收）给一个极小高度，否则完全看不见
        span = float(np.nanmax(h) - np.nanmin(l)) or max(float(np.nanmax(h)), 1.0) * 0.02
        floor = span * 0.0016
        body = np.maximum(np.abs(c - o), floor)
        bottom = np.minimum(o, c)
        ax1.bar(x[up], body[up], bottom=bottom[up], width=0.66,
                color=UP, edgecolor=UP, linewidth=0.3, zorder=3)
        ax1.bar(x[~up], body[~up], bottom=bottom[~up], width=0.66,
                color=DOWN, edgecolor=DOWN, linewidth=0.3, zorder=3)

        # 均线
        for n, col in ((5, "#2980b9"), (10, "#e67e22"), (20, "#8e44ad"), (60, "#16a085")):
            if len(df) >= n:
                ma = df["close"].rolling(n).mean().to_numpy(dtype=float)
                ax1.plot(x, ma, color=col, lw=0.9, label=f"MA{n}", zorder=4)
        ax1.legend(loc="upper left", fontsize=8, ncol=4, framealpha=0.92,
                   borderpad=0.3, handlelength=1.3, columnspacing=1.1)
        ax1.set_title(title, fontsize=10, loc="left", color=FG, pad=6)

        # 成交量
        vol = df["volume"].to_numpy(dtype=float)
        ax2.bar(x, vol, width=0.66, color=np.where(up, UP, DOWN),
                alpha=0.72, linewidth=0)
        ax2.set_ylabel("成交量", fontsize=8, color="#55606b")
        ax2.ticklabel_format(axis="y", style="plain", useOffset=False)
        vmax = float(np.nanmax(vol)) if len(vol) else 0.0
        ax2.yaxis.set_major_formatter(self._vol_formatter(vmax))

        # X 轴：整数索引 + 日期文本（非交易时段不留空白，蜡烛均匀）
        ax1.tick_params(labelbottom=False)
        if self.period == "分时":
            # 分时用固定时间点，否则午休前后（11:30 与 13:00 索引相邻）刻度会重叠
            ax2.xaxis.set_major_locator(mticker.FixedLocator(self._intraday_ticks()))
        else:
            ax2.xaxis.set_major_locator(
                mticker.MaxNLocator(nbins=9, integer=True, min_n_ticks=3))
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(self._fmt_x))

        # 十字光标对象重建（cla 会把旧对象清掉）
        self.v1 = ax1.axvline(0, color="#5b6b7c", lw=0.8, ls="--", visible=False, zorder=6)
        self.v2 = ax2.axvline(0, color="#5b6b7c", lw=0.8, ls="--", visible=False, zorder=6)
        self.h1 = ax1.axhline(0, color="#5b6b7c", lw=0.8, ls="--", visible=False, zorder=6)
        self.info = ax1.text(
            0.012, 0.962, "", transform=ax1.transAxes, va="top", ha="left",
            fontsize=8.5, color=FG, visible=False, zorder=8, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.42", fc="#ffffff", ec="#c8cdd4",
                      lw=0.7, alpha=0.96))

    def _fmt_x(self, val, _pos=None):
        if self.dates is None:
            return ""
        i = int(round(val))
        if i < 0 or i >= len(self.dates):
            return ""
        d = self.dates.iloc[i]
        if self.is_minute:
            return d.strftime("%m-%d %H:%M") if self.period != "分时" else d.strftime("%H:%M")
        if self.period == "月K":
            return d.strftime("%Y-%m")
        if self.period == "周K":
            return d.strftime("%y-%m-%d")
        return d.strftime("%y-%m-%d")

    @staticmethod
    def _fmt_vol_axis(v, _pos=None):
        """成交量轴：默认会显示 1e6 这种科学计数法，改成 万/亿 更好读。"""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
        if v >= 1e8:
            return f"{v / 1e8:.1f}亿"
        if v >= 1e4:
            return f"{v / 1e4:.0f}万"
        return f"{v:.0f}"

    @staticmethod
    def _vol_formatter(vmax):
        """按该图成交量量级选单位，避免「1万」和「8000」混排。"""
        def f(v, _pos=None):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return ""
            if vmax >= 1e8:
                return f"{v / 1e8:.1f}亿"
            if vmax >= 1e5:
                return f"{v / 1e4:.0f}万"
            return f"{v:,.0f}"
        return mticker.FuncFormatter(f)

    def _intraday_ticks(self):
        """分时图固定 5 个时间点做刻度（09:30 / 10:30 / 11:30 / 13:30 / 14:30）。"""
        out = []
        try:
            arr = pd.to_datetime(self.dates).to_numpy()
            if len(arr) == 0:
                return [0]
            day = pd.Timestamp(arr[0]).normalize()
            for hh, mm in ((9, 30), (10, 30), (11, 30), (13, 30), (14, 30)):
                key = np.datetime64(day + pd.Timedelta(hours=hh, minutes=mm))
                j = int(np.searchsorted(arr, key))
                if 0 <= j < len(arr):
                    out.append(j)
        except Exception:
            out = [0]
        return out or [0]

    # ---------------- 视图 ----------------
    def _reset_view(self, n_last=120):
        if self.df is None:
            return
        n = len(self.df)
        # 分时只有当日两三百根，默认全览（看完整一天节奏）；其余周期默认看最近 120 根
        if self.period == "分时" or n <= n_last:
            self.ax1.set_xlim(-0.5, n - 0.5)
            return
        self.ax1.set_xlim(max(-0.5, n - n_last - 0.5), n - 0.5)

    def reset_view(self):
        """全览。"""
        if self.df is None:
            return
        n = len(self.df)
        self.ax1.set_xlim(-0.5, n - 0.5)
        self._autoscale_y()
        self._draw_now()

    def _autoscale_y(self):
        if self.df is None:
            return
        x0, x1 = self.ax1.get_xlim()
        i0 = max(0, int(np.floor(x0)))
        i1 = min(len(self.df) - 1, int(np.ceil(x1)))
        if i1 < i0:
            return
        seg = self.df.iloc[i0:i1 + 1]
        lo = float(seg["low"].min())
        hi = float(seg["high"].max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return
        pad = (hi - lo) * 0.08
        self.ax1.set_ylim(lo - pad, hi + pad)
        vmax = float(seg["volume"].max()) if len(seg) else 1.0
        self.ax2.set_ylim(0, (vmax if np.isfinite(vmax) and vmax > 0 else 1.0) * 1.18)

    # ---------------- 绘制 / blit ----------------
    def _set_overlay_visible(self, vis):
        for a in (self.v1, self.v2, self.h1, self.info):
            if a is not None:
                a.set_visible(vis)

    def _draw_now(self):
        # 画背景前必须先藏掉十字光标：否则光标会被一起画进背景图，
        # 之后 blit 只叠加新光标，旧位置就留下一条残影。
        self._set_overlay_visible(False)
        try:
            self.canvas.draw()
            self._bg = (self.canvas.copy_from_bbox(self.ax1.bbox),
                        self.canvas.copy_from_bbox(self.ax2.bbox))
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
            cv.restore_region(self._bg[0])
            cv.restore_region(self._bg[1])
            for a in (self.v1, self.h1, self.info):
                if a is not None and a.get_visible():
                    self.ax1.draw_artist(a)
            cv.blit(self.ax1.bbox)
            if self.v2 is not None and self.v2.get_visible():
                self.ax2.draw_artist(self.v2)
            cv.blit(self.ax2.bbox)
        except Exception:
            self._bg = None
            try:
                self.canvas.draw_idle()
            except Exception:
                pass

    def _on_resize(self, _event):
        if self.df is None:
            return
        self._draw_now()

    # ---------------- 十字光标 ----------------
    def _hide_cross(self):
        if self.v1 is None or not self.v1.get_visible():
            return
        for a in (self.v1, self.v2, self.h1, self.info):
            if a is not None:
                a.set_visible(False)
        self._blit()

    def _on_leave(self, _event):
        self._hide_cross()

    def _update_cross(self, i, y=None):
        row = self.df.iloc[i]
        xv = float(self.x[i])
        for ln in (self.v1, self.v2):
            ln.set_xdata([xv, xv])
            ln.set_visible(True)
        if y is not None and np.isfinite(y):
            self.h1.set_ydata([y, y])
            self.h1.set_visible(True)
        else:
            self.h1.set_visible(False)

        o = float(row["open"])
        c = float(row["close"])
        h = float(row["high"])
        l = float(row["low"])
        prev = float(self.df["close"].iloc[i - 1]) if i > 0 else o
        pct = (c / prev - 1) * 100 if prev else 0.0
        d = row["date"]
        when = d.strftime("%Y-%m-%d %H:%M") if self.is_minute else d.strftime("%Y-%m-%d")

        lines = [
            f"{when}",
            f"开 {o:>8.2f}   高 {h:>8.2f}",
            f"低 {l:>8.2f}   收 {c:>8.2f}",
            f"涨跌 {pct:>+7.2f}%   量 {_fmt_vol(row['volume'])}",
        ]
        amt = row.get("amount")
        if amt is not None and np.isfinite(amt):
            lines.append(f"成交额 {_fmt_amount(amt)}")

        # 数值框自动躲到鼠标另一侧，避免挡住正在看的 K 线
        x0, x1 = self.ax1.get_xlim()
        on_left = i < (x0 + x1) / 2
        self.info.set_x(0.988 if on_left else 0.012)
        self.info.set_ha("right" if on_left else "left")
        self.info.set_text("\n".join(lines))
        self.info.set_color(UP if pct > 0 else (DOWN if pct < 0 else GREY))
        self.info.set_visible(True)
        self._blit()

    # ---------------- 鼠标事件 ----------------
    def _on_move(self, event):
        if self.df is None:
            return

        # 拖动平移
        if self._drag is not None:
            if event.xdata is None:
                return
            x_press, old0, old1 = self._drag
            dx = event.xdata - x_press
            span = old1 - old0
            n = len(self.df)
            new0 = old0 - dx
            if new0 < -0.5:
                new0 = -0.5
            if new0 + span > n - 0.5:
                new0 = max(-0.5, n - 0.5 - span)
            self.ax1.set_xlim(new0, new0 + span)
            self._autoscale_y()
            self._draw_now()
            return

        if event.inaxes not in (self.ax1, self.ax2) or event.xdata is None:
            self._hide_cross()
            return
        i = int(round(event.xdata))
        if i < 0 or i >= len(self.df):
            self._hide_cross()
            return
        self._update_cross(i, event.ydata if event.inaxes is self.ax1 else None)

    def _on_press(self, event):
        if self.df is None:
            return
        if event.dblclick:
            self.reset_view()
            return
        if event.button != 1 or event.inaxes not in (self.ax1, self.ax2):
            return
        if event.xdata is None:
            return
        x0, x1 = self.ax1.get_xlim()
        self._drag = (event.xdata, x0, x1)
        self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _on_release(self, _event):
        if self._drag is not None:
            self._drag = None
            self.canvas.setCursor(Qt.CursorShape.CrossCursor)

    def _on_scroll(self, event):
        if self.df is None or event.xdata is None:
            return
        n = len(self.df)
        x0, x1 = self.ax1.get_xlim()
        span = x1 - x0
        if span <= 0:
            return
        factor = 0.82 if event.button == "up" else 1.22
        new_span = max(min(self.MIN_SPAN, float(n)), min(float(n) + 1.0, span * factor))
        ratio = (event.xdata - x0) / span
        new0 = event.xdata - new_span * ratio
        if new0 < -0.5:
            new0 = -0.5
        if new0 + new_span > n - 0.5:
            new0 = max(-0.5, n - 0.5 - new_span)
        self.ax1.set_xlim(new0, new0 + new_span)
        self._autoscale_y()
        self._draw_now()


# --------------------------------------------------------------------------- #
# 净值曲线控件：滚轮缩放 + 拖动平移 + 双击复位 + 十字光标取值 + 买卖点标记
# 交互逻辑与 KLineChart 保持一致，只是把蜡烛换成一条净值线。
# --------------------------------------------------------------------------- #
class EquityChart(QWidget):
    MIN_SPAN = 8.0         # 最多放大到可见 8 个交易日

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dates = None
        self.eq = None
        self.cum = None        # 累计收益率（%）
        self.dd = None         # 距前期高点的回撤（%）
        self.sig = None        # 每日持仓状态 0/1
        self.init_cash = None
        self.entries: list[int] = []
        self.exits: list[int] = []
        self._pos: dict[str, int] = {}     # 日期 -> 索引，供表格点击定位
        self._hl: list = []                # 当前高亮的交易标记（下次定位时清掉）
        self._bg = None
        self._drag = None

        self.fig = Figure(figsize=(6, 2.6), dpi=100, facecolor="#ffffff")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.subplots_adjust(left=0.082, right=0.995, top=0.905, bottom=0.175)
        self._style()

        self.v = self.h = self.info = self.trade_info = None

        hint = QLabel("滚轮缩放 · 按住拖动平移 · 双击复位 · 悬停看数值 · 点下方交易行可定位")
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
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(labelsize=8, colors="#55606b", length=3)
        for sp in ax.spines.values():
            sp.set_color("#d7dce2")

    # ---------------- 占位 ----------------
    def show_placeholder(self, text):
        self.dates = None
        self._bg = None
        self.ax.cla()
        self._style()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(False)
        self.ax.text(0.5, 0.5, text, ha="center", va="center",
                     transform=self.ax.transAxes, fontsize=12, color="#98a2ae")
        self.v = self.h = self.info = self.trade_info = None
        self._hl = []
        self._pos = {}
        if self.isVisible():
            self._draw_now()

    # ---------------- 数据 ----------------
    def set_result(self, res):
        df = res["df"]
        if df is None or len(df) == 0:
            self.show_placeholder("无数据")
            return
        self.dates = df["date"].reset_index(drop=True)
        eq = df["equity"].to_numpy(dtype=float)
        self.eq = eq
        init = float(res["metrics"]["init_cash"])
        self.init_cash = init
        self.cum = (eq / init - 1.0) * 100.0
        run_max = np.maximum.accumulate(eq)
        self.dd = (eq / np.where(run_max > 0, run_max, 1.0) - 1.0) * 100.0
        self.sig = df["signal"].to_numpy(dtype=int)

        # 买卖点：用交易日期反查索引，画成三角形标记
        pos = {}
        for i, d in enumerate(pd.to_datetime(self.dates)):
            pos[pd.Timestamp(d).strftime("%Y-%m-%d")] = i
        self._pos = pos
        self.entries, self.exits = [], []
        for t in res.get("trades", []):
            i = pos.get(str(t.get("entry_date")))
            if i is not None:
                self.entries.append(i)
            j = pos.get(str(t.get("exit_date")))
            if j is not None:
                self.exits.append(j)

        self._render()
        self._reset_view()
        self._autoscale_y()
        self._draw_now()
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)

    def _render(self):
        ax = self.ax
        ax.cla()
        self._style()
        x = np.arange(len(self.eq), dtype=float)
        eq = self.eq
        init = self.init_cash
        up = bool(eq[-1] >= init)
        line_col = UP if up else DOWN

        # 曲线下方淡色铺底（基底取全序列最低点，缩放时自然被裁到画面底部，
        # 不会像"峰值落差"那样在放大后糊成一大块）
        base = float(np.nanmin(eq))
        ax.fill_between(x, base, eq, color=line_col, alpha=0.09,
                        linewidth=0, interpolate=True)
        ax.plot(x, eq, color=line_col, lw=1.2, zorder=4)
        self.base_line = ax.axhline(init, color=GREY, ls="--", lw=0.8, zorder=3)
        # 期初线标注：x 用轴坐标、y 用数据坐标，钉在左侧不会被挤出画布
        self.base_text = ax.text(
            0.006, init, f"期初 {init / 1e4:.1f}万", color=GREY, fontsize=7.5,
            va="bottom", ha="left", transform=ax.get_yaxis_transform(), zorder=5)

        # 买入红三角、卖出绿三角
        if self.entries:
            ax.scatter(np.asarray(self.entries, dtype=float), eq[self.entries],
                       marker="^", s=26, color=UP, zorder=6, linewidths=0)
        if self.exits:
            ax.scatter(np.asarray(self.exits, dtype=float), eq[self.exits],
                       marker="v", s=26, color=DOWN, zorder=6, linewidths=0)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _p=None: f"{v / 1e4:.1f}万"))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, integer=True, min_n_ticks=3))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(self._fmt_x))

        self.v = ax.axvline(0, color="#5b6b7c", lw=0.8, ls="--", visible=False, zorder=7)
        self.h = ax.axhline(0, color="#5b6b7c", lw=0.8, ls="--", visible=False, zorder=7)
        self.info = ax.text(
            0.012, 0.965, "", transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, color=FG, visible=False, zorder=9, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.42", fc="#ffffff", ec="#c8cdd4",
                      lw=0.7, alpha=0.96))
        # 选中某笔交易时的摘要框（固定在左下角，不跟鼠标跑）
        self.trade_info = ax.text(
            0.012, 0.035, "", transform=ax.transAxes, va="bottom", ha="left",
            fontsize=8.5, color=FG, visible=False, zorder=9, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.42", fc="#fffdf5", ec="#e0b862",
                      lw=0.9, alpha=0.97))
        self._hl = []

    def _fmt_x(self, val, _pos=None):
        if self.dates is None:
            return ""
        i = int(round(val))
        if i < 0 or i >= len(self.dates):
            return ""
        return pd.Timestamp(self.dates.iloc[i]).strftime("%y-%m-%d")

    # ---------------- 视图 ----------------
    def _reset_view(self, n_last=180):
        n = len(self.eq)
        if n <= n_last:
            self.ax.set_xlim(-0.5, n - 0.5)
            return
        self.ax.set_xlim(max(-0.5, n - n_last - 0.5), n - 0.5)

    def reset_view(self):
        if self.eq is None:
            return
        self._clear_highlight()
        self.ax.set_xlim(-0.5, len(self.eq) - 0.5)
        self._autoscale_y()
        self._draw_now()

    # ---------------- 交易定位（表格点行 -> 图上两个点） ----------------
    def index_of(self, date_str):
        return self._pos.get(str(date_str))

    def _clear_highlight(self):
        for a in self._hl:
            try:
                a.remove()
            except Exception:
                pass
        self._hl = []
        if self.trade_info is not None:
            self.trade_info.set_visible(False)

    def focus_trade(self, i_in, i_out=None, trade=None, pad=18):
        """把视图挪到这笔交易上：入场/退出各打一根竖线 + 空心圆点，
        中间涂一道淡蓝区间，左下角给出这笔交易的摘要。"""
        if self.eq is None or i_in is None:
            return False
        i_in = int(i_in)
        i_out = int(i_out) if i_out is not None else None
        n = len(self.eq)
        lo_i = min(i_in, i_out) if i_out is not None else i_in
        hi_i = max(i_in, i_out) if i_out is not None else i_in
        lo = max(-0.5, lo_i - pad)
        hi = min(n - 0.5, hi_i + pad)
        if hi - lo < self.MIN_SPAN:            # 太短的持仓补到最小跨度，别缩成一根线
            grow = (self.MIN_SPAN - (hi - lo)) / 2.0
            lo = max(-0.5, lo - grow)
            hi = min(n - 0.5, hi + grow)
        self.ax.set_xlim(lo, hi)
        self._autoscale_y()

        self._clear_highlight()
        ax = self.ax
        if i_out is not None and i_out != i_in:
            self._hl.append(ax.axvspan(lo_i - 0.5, hi_i + 0.5, color="#0b63b6",
                                       alpha=0.07, zorder=1))
        self._hl.append(ax.axvline(i_in, color=UP, ls=":", lw=1.2, zorder=6))
        self._hl.append(ax.plot([i_in], [self.eq[i_in]], marker="o", ms=7.5,
                                mfc="none", mec=UP, mew=1.7, zorder=8)[0])
        if i_out is not None:
            self._hl.append(ax.axvline(i_out, color=DOWN, ls=":", lw=1.2, zorder=6))
            self._hl.append(ax.plot([i_out], [self.eq[i_out]], marker="o", ms=7.5,
                                    mfc="none", mec=DOWN, mew=1.7, zorder=8)[0])

        if trade is not None and self.trade_info is not None:
            ret = trade.get("return")
            entry_p = trade.get("entry_price")
            last_p = trade.get("exit_price")
            if ret is None:                      # 到区间末还没平仓
                try:
                    flt = (float(last_p) / float(entry_p) - 1.0) * 100.0
                    tail = f"浮动 {flt:+.2f}%"
                except (TypeError, ValueError, ZeroDivisionError):
                    flt, tail = 0.0, "浮动 —"
                lines = [
                    f"{trade.get('entry_date')}  买 {entry_p}",
                    f"持有中  现价 {last_p}",
                    tail,
                ]
            else:
                flt = ret
                lines = [
                    f"{trade.get('entry_date')}  买 {entry_p}",
                    f"{trade.get('exit_date')}  卖 {last_p}",
                    f"这笔 {ret:+.2f}%",
                ]
                # 持仓期内资金曲线到过多高、又回吐多少，写明白就不会被误读成回撤
                if i_out is not None and i_out > i_in:
                    seg = self.eq[i_in:i_out + 1]
                    k = int(np.argmax(seg))
                    peak, base = float(seg[k]), float(seg[0])
                    if peak > 0 and base > 0:
                        up = (peak / base - 1.0) * 100.0
                        give = (float(seg[-1]) / peak - 1.0) * 100.0
                        if up > abs(float(ret or 0)) + 0.5:
                            try:
                                ds = pd.Timestamp(self.dates.iloc[i_in + k]).strftime("%m-%d")
                            except Exception:
                                ds = str(self.dates.iloc[i_in + k])[5:10]
                            lines.append(f"期间最高 {up:+.2f}%  {ds} 后回吐 {give:+.2f}%")
            self.trade_info.set_text("\n".join(lines))
            self.trade_info.set_color(UP if flt > 0 else (DOWN if flt < 0 else FG))
            self.trade_info.set_visible(True)
        self._draw_now()
        return True

    def _autoscale_y(self):
        if self.eq is None:
            return
        x0, x1 = self.ax.get_xlim()
        i0 = max(0, int(np.floor(x0)))
        i1 = min(len(self.eq) - 1, int(np.ceil(x1)))
        if i1 < i0:
            return
        seg = self.eq[i0:i1 + 1]
        lo, hi = float(np.min(seg)), float(np.max(seg))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if hi <= lo:
            hi = lo + max(abs(lo) * 0.01, 1.0)
        pad = (hi - lo) * 0.12
        self.ax.set_ylim(lo - pad, hi + pad)
        # 期初基准线只在可视范围内才显示，否则会被裁成一条没头没尾的虚影
        show = (lo - pad) <= self.init_cash <= (hi + pad)
        for a in (getattr(self, "base_line", None), getattr(self, "base_text", None)):
            if a is not None:
                a.set_visible(show)

    # ---------------- 绘制 / blit ----------------
    def _set_overlay_visible(self, vis):
        for a in (self.v, self.h, self.info):
            if a is not None:
                a.set_visible(vis)

    def _draw_now(self):
        # 同 K 线图：画背景前先藏光标，否则光标被烤进底图留下残影
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
        if self.eq is not None:
            self._draw_now()

    # ---------------- 十字光标 ----------------
    def _hide_cross(self):
        if self.v is None or not self.v.get_visible():
            return
        for a in (self.v, self.h, self.info):
            if a is not None:
                a.set_visible(False)
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

        eq = float(self.eq[i])
        cum = float(self.cum[i])
        dd = float(self.dd[i])
        prev = float(self.eq[i - 1]) if i > 0 else eq
        day = (eq / prev - 1) * 100 if prev else 0.0
        hold = "持仓" if int(self.sig[i]) == 1 else "空仓"
        when = pd.Timestamp(self.dates.iloc[i]).strftime("%Y-%m-%d")

        lines = [
            f"{when}   {hold}",
            f"净值 {eq:>11,.2f}",
            f"累计 {cum:>+10.2f}%",
            f"当日 {day:>+10.2f}%",
            f"回撤 {dd:>10.2f}%",
        ]
        x0, x1 = self.ax.get_xlim()
        on_left = i < (x0 + x1) / 2
        self.info.set_x(0.988 if on_left else 0.012)
        self.info.set_ha("right" if on_left else "left")
        self.info.set_text("\n".join(lines))
        self.info.set_color(UP if cum > 0 else (DOWN if cum < 0 else GREY))
        self.info.set_visible(True)
        self._blit()

    # ---------------- 鼠标事件 ----------------
    def _on_move(self, event):
        if self.eq is None:
            return
        if self._drag is not None:
            if event.xdata is None:
                return
            x_press, old0, old1 = self._drag
            dx = event.xdata - x_press
            span = old1 - old0
            n = len(self.eq)
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
        if i < 0 or i >= len(self.eq):
            self._hide_cross()
            return
        self._update_cross(i, event.ydata)

    def _on_press(self, event):
        if self.eq is None:
            return
        if event.dblclick:
            self.reset_view()
            return
        if event.button != 1 or event.inaxes is not self.ax or event.xdata is None:
            return
        x0, x1 = self.ax.get_xlim()
        self._drag = (event.xdata, x0, x1)
        self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _on_release(self, _event):
        if self._drag is not None:
            self._drag = None
            self.canvas.setCursor(Qt.CursorShape.CrossCursor)

    def _on_scroll(self, event):
        if self.eq is None or event.xdata is None:
            return
        n = len(self.eq)
        x0, x1 = self.ax.get_xlim()
        span = x1 - x0
        if span <= 0:
            return
        factor = 0.82 if event.button == "up" else 1.22
        new_span = max(min(self.MIN_SPAN, float(n)), min(float(n) + 1.0, span * factor))
        ratio = (event.xdata - x0) / span
        new0 = event.xdata - new_span * ratio
        if new0 < -0.5:
            new0 = -0.5
        if new0 + new_span > n - 0.5:
            new0 = max(-0.5, n - 0.5 - new_span)
        self.ax.set_xlim(new0, new0 + new_span)
        self._autoscale_y()
        self._draw_now()


# --------------------------------------------------------------------------- #
# 监控 Tab
# --------------------------------------------------------------------------- #
class LegacyMonitorTab(QWidget):
    """已停用：行情监控由券商客户端承担，本工具不再重复实现。

    代码暂时保留，防止外部脚本引用报错；主窗口不再挂载这个页签。
    """
    COLS = ["名称", "代码", "最新价", "涨跌%", "换手%", "成交额"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = load_watchlist()
        self.quotes: dict[str, dict] = {}
        self.period = "日K"
        self.adjust = "qfq"
        self.cur_code = None
        self.seq = 0
        self.cache: dict[tuple, pd.DataFrame] = {}
        self._workers = []          # 持有线程引用，防止提前回收

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ---------- 工具条 ----------
        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(QLabel("周期"))
        self.period_box = QComboBox()
        self.period_box.addItems(ALL_PERIODS)
        self.period_box.setCurrentText("日K")
        self.period_box.setFixedWidth(92)
        self.period_box.currentTextChanged.connect(self.on_period)
        bar.addWidget(self.period_box)

        bar.addWidget(QLabel("复权"))
        self.adjust_box = QComboBox()
        self.adjust_box.addItems(["前复权", "不复权"])
        self.adjust_box.setFixedWidth(86)
        self.adjust_box.currentTextChanged.connect(self.on_adjust)
        bar.addWidget(self.adjust_box)

        bar.addSpacing(8)
        self.search = QLineEdit()
        self.search.setPlaceholderText("输入代码或名称添加标的（如 603011 / 合锻 / 锂业）")
        self.search.setMinimumWidth(180)
        self.search.returnPressed.connect(self.add_symbol)
        bar.addWidget(self.search, 1)

        # clicked 信号会带一个 checked:bool 参数，槽函数若只收 self 就必须用 lambda 兜住，
        # 否则 TypeError 会被 _guard 吞掉，表现成"点了没反应"。
        self.add_btn = QPushButton("添加标的")
        self.add_btn.clicked.connect(lambda *_: self.add_symbol())
        bar.addWidget(self.add_btn)

        self.del_btn = QPushButton("删除选中")
        self.del_btn.clicked.connect(lambda *_: self.del_symbol())
        bar.addWidget(self.del_btn)

        self.reset_btn = QPushButton("恢复默认池")
        self.reset_btn.clicked.connect(lambda *_: self.reset_pool())
        bar.addWidget(self.reset_btn)

        self.refresh_btn = QPushButton("刷新行情")
        self.refresh_btn.clicked.connect(lambda: self.refresh())
        bar.addWidget(self.refresh_btn)
        root.addLayout(bar)

        # ---------- 主体：左侧列表 + 右侧图表 ----------
        left_w = QWidget()
        left = QVBoxLayout(left_w)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(4)

        # 可排序：点表头按名称/价格/涨跌/换手/成交额排，列宽可拖
        self.table = SortTable(self.COLS, row_h=25, last_stretch=True)
        for i, w in enumerate((92, 68, 70, 64, 60)):
            self.table.setColumnWidth(i, w)
        self.table.setToolTip("点表头排序，再点一次反序；列宽可拖动")
        self.table.cellClicked.connect(self.on_select)
        left.addWidget(self.table, 1)

        self.status = QLabel("启动中…")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        left.addWidget(self.status)

        self.chart = KLineChart()

        left_w.setMinimumWidth(452)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left_w)
        split.addWidget(self.chart)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([466, 880])
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        QTimer.singleShot(0, self._boot)

    # ---------------- 启动 ----------------
    def _boot(self):
        try:
            self.chart.show_placeholder("正在加载行情…")
        except Exception:
            pass
        self.refresh()

    # ---------------- 线程管理 ----------------
    def _run(self, fn, on_done, on_error=None, *args):
        w = Worker(fn, *args)
        w.finished.connect(on_done)
        if on_error is not None:
            w.error.connect(on_error)
        else:
            w.error.connect(lambda e: self.status.setText(f"失败：{e}"))
        # 队列里最多留 8 个线程引用
        self._workers.append(w)
        if len(self._workers) > 8:
            self._workers = self._workers[-8:]
        w.start()
        return w

    # ---------------- 行情刷新 ----------------
    def refresh(self):
        if not self.items:
            self.table.setRowCount(0)
            self.status.setText("股票池为空，输入代码或名称添加标的")
            self.chart.show_placeholder("股票池为空")
            return
        self.refresh_btn.setEnabled(False)
        self.status.setText("拉取行情中…")
        codes = [it["code"] for it in self.items]
        self._run(get_quotes, self.on_quotes,
                  lambda e: (self.status.setText(f"行情失败：{e}"),
                             self.refresh_btn.setEnabled(True)), codes)

    @_guard
    def on_quotes(self, quotes):
        self.refresh_btn.setEnabled(True)
        self.quotes = quotes or {}
        self.fill_table()
        self.status.setText(
            f"{len(self.items)} 只 · 更新 {datetime.datetime.now():%H:%M:%S} · "
            f"当前 {self.period}")
        if self.cur_code is None or self.cur_code not in [it["code"] for it in self.items]:
            if self.items:
                self._select_code(self.items[0]["code"])

    def fill_table(self):
        self.table.setRowCount(len(self.items))
        for r, it in enumerate(self.items):
            q = self.quotes.get(it["code"], {})
            price = q.get("price")
            pct = q.get("pct")
            turn = q.get("turnover")
            amt = q.get("amount")
            name = it["name"] or q.get("name") or it["code"]
            self._set(r, 0, name, None, ref=it["code"])
            self._set(r, 1, it["code"], "#55606b")
            self._set(r, 2, f"{price:.2f}" if price is not None else "—",
                      UP if (pct or 0) > 0 else (DOWN if (pct or 0) < 0 else None))
            self._set(r, 3, f"{pct:+.2f}" if pct is not None else "—", _num(pct))
            self._set(r, 4, f"{turn:.2f}" if turn is not None else "—", None)
            self._set(r, 5, _fmt_amount(amt), "#55606b")

    def _set(self, r, c, text, color, ref=None):
        item = CellItem(text, color=QColor(color) if color else None)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(r, c, item)
        if ref is not None:
            item.setData(Qt.ItemDataRole.UserRole, ref)

    # ---------------- 选择 & 画图 ----------------
    @_guard
    def on_select(self, row, _col=0):
        code = self.table.ref(row)          # 排序后行号不可信，认绑定
        if not code:
            if row < 0 or row >= len(self.items):
                return
            code = self.items[row]["code"]
        if code != self.cur_code:
            self.load_chart(code)

    def _name_of(self, code):
        for it in self.items:
            if it["code"] == code:
                return it["name"]
        q = self.quotes.get(code) or {}
        return q.get("name", code)

    def load_chart(self, code, force=False):
        self.cur_code = code
        self.seq += 1
        seq = self.seq
        key = (code, self.period, self.adjust)
        if not force and key in self.cache:
            self.render_chart(code, self.cache[key])
            return
        self.status.setText(f"加载 {self._name_of(code)} {self.period}…")
        self.chart.show_placeholder(f"加载 {self._name_of(code)} {self.period} 数据…")
        self._run(get_bars,
                  lambda df, c=code, s=seq: self.on_bars(c, df, s),
                  lambda e: self.chart.show_placeholder(f"加载失败：{e}"),
                  code, self.period, self.adjust)

    @_guard
    def on_bars(self, code, df, seq):
        if seq != self.seq:          # 用户已切到别的标的/周期，丢弃过期结果
            return
        if df is None or df.empty:
            self.chart.show_placeholder(f"{self._name_of(code)} {self.period} 无数据")
            self.status.setText("无数据（该周期可能尚未产生数据）")
            return
        if len(self.cache) > 40:
            self.cache.clear()
        self.cache[(code, self.period, self.adjust)] = df
        self.render_chart(code, df)

    def render_chart(self, code, df):
        q = self.quotes.get(code, {})
        name = self._name_of(code)
        title = f"{name} {code} · {self.period}"
        if q.get("price") is not None:
            title += f"   {q['price']:.2f}   {q.get('pct', 0):+.2f}%"
        self.chart.set_data(df, title, self.period)
        d0, d1 = df["date"].iloc[0], df["date"].iloc[-1]
        fmt = "%Y-%m-%d %H:%M" if self.period in MIN_PERIODS else "%Y-%m-%d"
        self.status.setText(
            f"{name} {code} · {self.period} · {len(df)} 根 · "
            f"{d0.strftime(fmt)} ~ {d1.strftime(fmt)}")

    # ---------------- 周期 / 复权 ----------------
    @_guard
    def on_period(self, text):
        self.period = text
        if self.cur_code:
            self.load_chart(self.cur_code)

    @_guard
    def on_adjust(self, text):
        self.adjust = "qfq" if text == "前复权" else ""
        if self.cur_code:
            self.load_chart(self.cur_code, force=True)

    # ---------------- 股票池管理 ----------------
    @_guard
    def add_symbol(self):
        kw = self.search.text().strip()
        if not kw:
            self.status.setText("请输入代码或名称")
            return
        self.add_btn.setEnabled(False)
        self.status.setText("搜索中…（首次需下载全市场代码表，约 15 秒，之后走本地缓存）")
        self._run(lambda: search_stocks(kw, 10), self.on_search_hits,
                  lambda e: (self.status.setText(f"搜索失败：{e}"),
                             self.add_btn.setEnabled(True)))

    @_guard
    def on_search_hits(self, hits):
        self.add_btn.setEnabled(True)
        if not hits:
            self.status.setText(f"未找到与「{self.search.text().strip()}」匹配的标的")
            return
        if len(hits) == 1:
            pick = hits[0]
        else:
            opts = [f"{h['name']}  ({h['code']})" for h in hits]
            pick = self._pick_dialog(opts, hits)
            if pick is None:
                self.status.setText("已取消")
                return

        code, name = pick["code"], pick["name"]
        if any(it["code"] == code for it in self.items):
            self.status.setText(f"{name}({code}) 已在股票池中")
            self._select_code(code)
            return
        self.items.append({"code": code, "name": name})
        save_watchlist(self.items)
        self.search.clear()
        self.fill_table()
        self.status.setText(f"已添加 {name}({code})，股票池共 {len(self.items)} 只")
        self.refresh()
        self._select_code(code)

    def _confirm(self, title, text, ok_text="确定"):
        """中文按钮的确认框（不依赖 Qt 翻译文件，打包后也不会变回 OK/Cancel）。"""
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        box.setIcon(QMessageBox.Icon.Question)
        yes = box.addButton(ok_text, QMessageBox.ButtonRole.AcceptRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return box.clickedButton() is yes

    def _pick_dialog(self, opts, hits):
        """多个匹配时弹选择框。用中文按钮，避免 PyQt 默认的 OK/Cancel。"""
        dlg = QInputDialog(self)
        dlg.setWindowTitle("选择标的")
        dlg.setLabelText(f"匹配到 {len(opts)} 个，请选择要添加的标的：")
        dlg.setComboBoxItems(opts)
        dlg.setComboBoxEditable(False)
        dlg.setFixedWidth(420)
        box = dlg.findChild(QDialogButtonBox)
        if box is not None:
            for std, text in ((QDialogButtonBox.StandardButton.Ok, "确定"),
                              (QDialogButtonBox.StandardButton.Cancel, "取消")):
                b = box.button(std)
                if b is not None:
                    b.setText(text)
        ok = dlg.exec() == QDialog.DialogCode.Accepted
        choice = dlg.textValue()
        if not ok or not choice or choice not in opts:
            return None
        return hits[opts.index(choice)]

    def _select_code(self, code):
        r = self.table.row_of(code)
        if r < 0:
            for i, it in enumerate(self.items):
                if it["code"] == code:
                    r = i
                    break
        if r < 0:
            return
        self.table.selectRow(r)
        self.cur_code = None              # 强制重画
        self.load_chart(code)

    @_guard
    def del_symbol(self):
        row = self.table.currentRow()
        code = self.table.ref(row) if row >= 0 else None
        idx = next((i for i, x in enumerate(self.items) if x["code"] == code), -1)
        if idx < 0:
            self.status.setText("请先在左侧列表选中一行")
            return
        it = self.items[idx]
        if not self._confirm(
                "删除标的",
                f"从股票池移除 {it['name']}({it['code']})？\n（只影响看盘列表，不删任何数据）",
                ok_text="删除"):
            return
        self.items.pop(idx)
        save_watchlist(self.items)
        self.fill_table()
        if self.cur_code == it["code"]:
            self.cur_code = None
            if self.items:
                self._select_code(self.items[0]["code"])
            else:
                self.chart.show_placeholder("股票池为空")
        self.status.setText(f"已移除 {it['name']}({it['code']})，剩余 {len(self.items)} 只")

    @_guard
    def reset_pool(self):
        ret = self._confirm(
            "恢复默认池",
            f"把股票池恢复成内置的 {len(WATCHLIST)} 只关注标的？\n（你自己添加的会被清掉）",
            ok_text="恢复")
        if not ret:
            return
        self.items = reset_watchlist()
        self.cur_code = None
        self.cache.clear()
        self.fill_table()
        self.refresh()
        self.status.setText("已恢复默认股票池")


# --------------------------------------------------------------------------- #
# 回测 Tab
# --------------------------------------------------------------------------- #
class LegacyBacktestTab(QWidget):
    """已停用：被 ui_tabs.BacktestTab 取代（新引擎 + 完整指标 + 稳健性检验）。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = load_watchlist()

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        self.symbol = QComboBox()
        self.symbol.addItems([f"{it['name']}({it['code']})" for it in self.items])
        self.symbol.setCurrentIndex(0)
        self.symbol.setMinimumWidth(150)

        self.strategy = QComboBox()
        self.strategy.addItems(["均线交叉", "动量"])
        self.strategy.setCurrentIndex(0)
        self.strategy.setMinimumWidth(100)

        self.fast = QSpinBox(); self.fast.setRange(2, 250); self.fast.setValue(5)
        self.slow = QSpinBox(); self.slow.setRange(3, 250); self.slow.setValue(20)
        self.mom_n = QSpinBox(); self.mom_n.setRange(2, 250); self.mom_n.setValue(20)
        for w in (self.fast, self.slow, self.mom_n):
            w.setMinimumWidth(70)

        self.start = QLineEdit(DEFAULT_START); self.start.setFixedWidth(100)
        self.end = QLineEdit(DEFAULT_END); self.end.setFixedWidth(100)

        grid.addWidget(QLabel("标的"), 0, 0)
        grid.addWidget(self.symbol, 0, 1)
        grid.addWidget(QLabel("策略"), 0, 2)
        grid.addWidget(self.strategy, 0, 3)
        grid.addWidget(QLabel("快线"), 0, 4)
        grid.addWidget(self.fast, 0, 5)
        grid.addWidget(QLabel("慢线"), 0, 6)
        grid.addWidget(self.slow, 0, 7)
        grid.addWidget(QLabel("动量N"), 0, 8)
        grid.addWidget(self.mom_n, 0, 9)

        grid.addWidget(QLabel("起"), 1, 0)
        grid.addWidget(self.start, 1, 1)
        grid.addWidget(QLabel("止"), 1, 2)
        grid.addWidget(self.end, 1, 3)

        self.run_btn = QPushButton("运行回测")
        self.run_btn.clicked.connect(lambda *_: self.run())
        self.run_btn.setMinimumWidth(110)
        grid.addWidget(self.run_btn, 1, 4, 1, 2)

        self.status = QLabel("选择标的后点击「运行回测」")
        self.status.setObjectName("status")
        grid.addWidget(self.status, 1, 6, 1, 4)

        grid.setColumnStretch(10, 1)
        root.addLayout(grid)

        self.chart = EquityChart()
        root.addWidget(self.chart, 3)

        self.metrics = QTableWidget(0, 4)
        self.metrics.setHorizontalHeaderLabels(["指标", "数值", "指标", "数值"])
        self.metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics.setAlternatingRowColors(True)
        self.metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics.verticalHeader().setVisible(False)
        # 九项指标分左右两栏排，5 行就够，省下的高度留给净值曲线
        self.metrics.verticalHeader().setDefaultSectionSize(22)
        self.metrics.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(self.metrics, 0)

        self.trades = SortTable(["入场日", "入场价", "退出日", "退出价", "收益%"])
        self.trades.setToolTip("点表头按日期或收益排序，点某一行把图上区间压到这笔交易")
        # 点一行就定位到图上这笔交易
        self.trades.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.trades.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.trades.setCursor(Qt.CursorShape.PointingHandCursor)
        self.trades.itemSelectionChanged.connect(lambda *_: self.on_trade_pick())
        # 同一行再点一次也要能重新定位（键盘上下选行走上面的 selectionChanged）
        self.trades.cellClicked.connect(lambda *_: self.on_trade_pick())
        self._trade_rows = []
        self._filling = False
        root.addWidget(self.trades, 2)

        # 不能在建构造里直接 draw()，控件未显示时绘制会硬崩，延后到事件循环。
        QTimer.singleShot(0, self._draw_placeholder)

    def _draw_placeholder(self):
        try:
            self.chart.show_placeholder("点击「运行回测」生成净值曲线")
        except Exception:
            _trace("回测占位绘制失败:\n" + traceback.format_exc())

    @_guard
    def on_trade_pick(self):
        """交易明细点行 -> 图上放大到这笔交易的入场/退出两点。"""
        if self._filling:
            return
        rows = self.trades.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        idx = self.trades.ref(r)      # 排序后行号不可信，认绑定的原始序号
        if not isinstance(idx, int) or not (0 <= idx < len(self._trade_rows)):
            idx = r
        if idx < 0 or idx >= len(self._trade_rows):
            return
        t = self._trade_rows[idx]
        seq = idx + 1
        i_in = self.chart.index_of(t["entry_date"])
        i_out = self.chart.index_of(t["exit_date"]) if t["exit_date"] != "持有中" else None
        if i_in is None:
            self.status.setText("这笔交易不在当前数据区间内")
            return
        if self.chart.focus_trade(i_in, i_out, t):
            if t["return"] is None:
                try:
                    flt = (float(t["exit_price"]) / float(t["entry_price"]) - 1) * 100
                    tail = f"浮动 {flt:+.2f}%"
                except (TypeError, ValueError, ZeroDivisionError):
                    tail = "浮动 —"
                self.status.setText(
                    f"第 {seq} 笔：{t['entry_date']} 买 {t['entry_price']} → 持有中 "
                    f"现价 {t['exit_price']}　{tail}　（双击图面回到全览）")
            else:
                self.status.setText(
                    f"第 {seq} 笔：{t['entry_date']} 买 {t['entry_price']} → "
                    f"{t['exit_date']} 卖 {t['exit_price']}　收益 {t['return']:+.2f}%　"
                    f"（双击图面回到全览）")

    def _code(self):
        text = self.symbol.currentText()
        return text.split("(")[-1].rstrip(")") if "(" in text else text.strip()

    def showEvent(self, event):
        """每次切到回测页时同步股票池（在监控页新加的标的要能选到）。"""
        super().showEvent(event)
        try:
            items = load_watchlist()
            if [i["code"] for i in items] == [i["code"] for i in self.items]:
                return
            keep = self._code()
            self.items = items
            self.symbol.blockSignals(True)
            self.symbol.clear()
            self.symbol.addItems([f"{it['name']}({it['code']})" for it in items])
            self.symbol.blockSignals(False)
            for i, it in enumerate(items):
                if it["code"] == keep:
                    self.symbol.setCurrentIndex(i)
                    break
        except Exception:
            _trace("回测页同步股票池失败:\n" + traceback.format_exc())

    @_guard
    def run(self):
        code = self._code()
        start = self.start.text().strip()
        end = self.end.text().strip()
        if not code:
            self.status.setText("请先选择标的")
            return
        for label, val in (("起始日期", start), ("结束日期", end)):
            if len(val) != 8 or not val.isdigit():
                self.status.setText(f"{label}格式应为 YYYYMMDD，当前：{val or '空'}")
                return
        if start > end:
            self.status.setText("起始日期晚于结束日期")
            return

        strat = self.strategy.currentText()
        fast, slow, mom_n = self.fast.value(), self.slow.value(), self.mom_n.value()
        if strat == "均线交叉" and fast >= slow:
            self.status.setText(f"快线({fast})必须小于慢线({slow})")
            return

        self.run_btn.setEnabled(False)
        self.status.setText(f"回测中…（{code} {start}-{end} {strat}）")

        w = Worker(self._backtest, code, start, end, strat, fast, slow, mom_n)
        # 必须持有引用：run() 返回后局部变量被回收，QThread 析构会导致进程直接 abort。
        self._worker = w
        w.finished.connect(self.on_done)
        w.error.connect(self.on_error)
        w.finished.connect(lambda _: self.run_btn.setEnabled(True))
        w.start()

    @_guard
    def on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.status.setText(f"回测失败：{msg}")
        self._draw_placeholder()
        _trace("回测失败: %s" % msg)

    @staticmethod
    def _backtest(code, start, end, strat, fast, slow, mom_n):
        df = feed.get_history(code, start, end)
        if df.empty:
            raise RuntimeError(f"无数据：{code}（区间 {start}-{end}）")
        close = df["close"]
        sig = (ma_cross_signal(close, fast, slow) if strat == "均线交叉"
               else momentum_signal(close, mom_n))
        return run_backtest(df, sig, INIT_CASH)

    @_guard
    def on_done(self, res):
        # 可缩放、可悬停取值的净值曲线（滚轮/拖动/双击，与 K 线图同一套手感）
        self.chart.set_result(res)

        m = res["metrics"]
        rows = [
            ("期初资金", f"{m['init_cash']:.2f}"),
            ("期末净值", f"{m['final_equity']:.2f}"),
            ("总收益率%", f"{m['total_return']:.2f}"),
            ("年化%", f"{m['annual_return']:.2f}"),
            ("最大回撤%", f"{m['max_drawdown']:.2f}"),
            ("胜率%", f"{m['win_rate']:.2f}"),
            ("交易次数", f"{m['trade_count']}"),
            ("夏普(近似)", f"{m['sharpe']:.2f}"),
            ("持仓未平", f"{m['open_trades']}"),
        ]
        # 九项指标分两栏：左 5 项 + 右 4 项，5 行排完
        left, right = rows[:5], rows[5:]
        n_row = max(len(left), len(right))
        self.metrics.setRowCount(n_row)
        tinted = ("总收益率%", "年化%", "最大回撤%")
        for r in range(n_row):
            if r < len(left):
                self._set(self.metrics, r, 0, left[r][0], None)
                self._set(self.metrics, r, 1, left[r][1],
                          _num(left[r][1]) if left[r][0] in tinted else None)
            if r < len(right):
                self._set(self.metrics, r, 2, right[r][0], None)
                self._set(self.metrics, r, 3, right[r][1],
                          _num(right[r][1]) if right[r][0] in tinted else None)

        # 重填表格期间别触发选中回调（setRowCount 会顺带清空选中）
        self._filling = True
        try:
            self._trade_rows = list(res["trades"])
            self.trades.clearSelection()
            self.trades.setRowCount(len(self._trade_rows))
            for r, t in enumerate(self._trade_rows):
                rv = "" if t["return"] is None else f"{t['return']:.2f}"
                self._set(self.trades, r, 0, t["entry_date"], None, ref=r)
                self._set(self.trades, r, 1, f"{t['entry_price']:.3f}", None)
                self._set(self.trades, r, 2, t["exit_date"], None)
                self._set(self.trades, r, 3, f"{t['exit_price']:.3f}", None)
                self._set(self.trades, r, 4, rv, _num(rv))
        finally:
            self._filling = False
        # 指标表按实际行数定高，两栏五行一屏看全，不留半截滚动
        self.metrics.resizeRowsToContents()
        body = sum(self.metrics.rowHeight(r) for r in range(n_row))
        self.metrics.setFixedHeight(self.metrics.horizontalHeader().height() + body + 2)
        self.status.setText(f"回测完成 · {len(res['trades'])} 笔交易　"
                            f"（点交易明细任一行，图上会定位到这笔买卖）")

    @staticmethod
    def _set(table, r, c, text, color):
        item = QTableWidgetItem(str(text))
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if color:
            item.setForeground(QColor(color))
        table.setItem(r, c, item)


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #
CSS = """
/* 全部显式指定前景色：Windows 深色模式下 Qt 会继承白色文字，
   若只把背景刷浅就会变成白字白底（控件内容全看不见）。 */
QMainWindow, QWidget { background: #f5f6f8; color: #1f2d3d; }
QLabel { color: #1f2d3d; background: transparent; }
QTabWidget::pane { border: 1px solid #d0d4da; background: #ffffff; top: -1px; }
QTabBar::tab { background: #e3e6ea; color: #1f2d3d; padding: 8px 20px;
               border: 1px solid #d0d4da; border-bottom: none;
               border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background: #ffffff; color: #0b63b6; font-weight: bold; }
QTabBar::tab:hover { background: #eef1f5; }
QPushButton { background: #2c3e50; color: #ffffff; border-radius: 5px;
              padding: 6px 14px; font-size: 12.5px; font-weight: bold; }
QPushButton:hover { background: #34495e; }
QPushButton:pressed { background: #1b2836; }
QPushButton:disabled { background: #b3bcc6; color: #f0f2f5; }
QComboBox, QLineEdit, QSpinBox {
    background: #ffffff; color: #1f2d3d; border: 1px solid #c8cdd4;
    border-radius: 4px; padding: 4px 8px; min-height: 20px;
    selection-background-color: #0b63b6; selection-color: #ffffff;
}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus { border: 1px solid #0b63b6; }
QComboBox QAbstractItemView {
    background: #ffffff; color: #1f2d3d; outline: none;
    selection-background-color: #0b63b6; selection-color: #ffffff;
}
QTableWidget {
    background: #ffffff; color: #1f2d3d; gridline-color: #e6e9ed;
    alternate-background-color: #fafbfc; selection-background-color: #d6e4f0;
    selection-color: #1f2d3d; font-size: 12.5px;
}
QHeaderView::section { background: #2c3e50; color: #ffffff; padding: 5px;
                       border: none; font-weight: bold; font-size: 12px; }
QTableCornerButton::section { background: #2c3e50; border: none; }
QLabel#status { color: #55606b; font-size: 11.5px; }
QSplitter::handle { background: #dfe3e8; width: 3px; }
"""


# --------------------------------------------------------------------------- #
# 实盘复盘 Tab：导入券商交割单，重建真实持仓与平仓回合
# --------------------------------------------------------------------------- #
class PnlChart(QWidget):
    """累计已实现盈亏曲线（按平仓日累加）。"""

    def __init__(self):
        super().__init__()
        self.canvas = FigureCanvas(Figure(figsize=(6, 2.1), dpi=100))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.show_placeholder("导入交割单后显示累计已实现盈亏")

    def show_placeholder(self, text):
        try:
            fig = self.canvas.figure
            fig.clear()
            ax = fig.add_subplot(1, 1, 1)
            ax.axis("off")
            ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12,
                    color="#98a2ae", transform=ax.transAxes)
            self.canvas.draw_idle()
        except Exception:
            _trace("盈亏曲线占位失败:\n" + traceback.format_exc())

    def set_curve(self, points):
        if not points:
            self.show_placeholder("还没有已平仓的交易")
            return
        dates = [pd.Timestamp(d) for d, _ in points]
        vals = [float(v) for _, v in points]
        col = UP if vals[-1] >= 0 else DOWN
        try:
            fig = self.canvas.figure
            fig.clear()
            ax = fig.add_subplot(1, 1, 1)
            ax.fill_between(dates, 0, vals, color=col, alpha=0.10, lw=0)
            ax.plot(dates, vals, color=col, lw=1.5, marker="o", ms=3.2,
                    mfc="white", mew=1.1, zorder=3)
            ax.axhline(0, color=GREY, ls="--", lw=0.8, zorder=1)

            ax.set_title("已平仓轮次累计盈亏（按出场日；持仓中已卖部分不在此列）",
                         fontsize=9, color=FG, loc="left")
            ax.grid(True, color="#e6e9ed", lw=0.6)
            ax.tick_params(labelsize=8, colors="#55606b")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _p=None: (f"{v / 1e4:.1f}万" if abs(v) >= 1e4 else f"{v:.0f}")))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _p=None: pd.Timestamp(mdates.num2date(v)).strftime("%y-%m-%d")))
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            fig.tight_layout()
            self.canvas.draw_idle()
        except Exception:
            _trace("盈亏曲线绘制失败:\n" + traceback.format_exc())


class LegacyReplayTab(QWidget):
    """已停用：实盘复盘已升级为「行为审计」页签（见 audit.py / ui_tabs.AuditTab）。

    旧版只做盈亏配对和净值曲线，新版做偏误诊断、分组找漏、影子对账。
    """

    H_CLOSED = ("代码", "名称", "入场", "出场", "买入额", "卖出额", "盈亏", "收益%", "天数")
    H_OPEN = ("代码", "名称", "股数", "成本价", "现价", "市值", "浮动盈亏", "浮动%")
    H_CODE = ("代码", "名称", "轮次", "胜", "买入额", "盈亏", "平均收益%", "盈亏占比%")

    def __init__(self):
        super().__init__()
        self.res = None
        self.meta = None
        self.quotes: dict[str, dict] = {}
        self.fx = 0.0
        self._last_dir = ""
        self._worker = None
        self._qworker = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(7)

        bar = QHBoxLayout()
        self.open_btn = QPushButton("导入交割单")
        self.open_btn.setMinimumWidth(112)
        self.open_btn.clicked.connect(lambda *_: self.on_open())
        bar.addWidget(self.open_btn)

        self.file_lbl = QLabel("未导入 · 券商里导出「交割单」，支持 xls / xlsx / csv / txt")
        self.file_lbl.setStyleSheet("color:#55606b;")
        bar.addWidget(self.file_lbl, 1)

        self.order_box = QComboBox()
        self.order_box.addItems(["按出场时间", "按盈亏", "按收益率"])
        self.order_box.setFixedWidth(106)
        self.order_box.currentIndexChanged.connect(lambda *_: self._fill())
        bar.addWidget(self.order_box)

        self.status = QLabel("")
        self.status.setStyleSheet("color:#1f6fb2;")
        bar.addWidget(self.status)
        root.addLayout(bar)

        self.kpi = QTableWidget(4, 4)
        self.kpi.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.kpi.setAlternatingRowColors(True)
        self.kpi.verticalHeader().setVisible(False)
        self.kpi.horizontalHeader().setVisible(False)
        self.kpi.verticalHeader().setDefaultSectionSize(25)
        self.kpi.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.kpi.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.kpi.setFixedHeight(25 * 4 + 4)
        root.addWidget(self.kpi)

        split = QSplitter(Qt.Orientation.Horizontal)
        wrap_l = QWidget()
        lay_l = QVBoxLayout(wrap_l)
        lay_l.setContentsMargins(0, 0, 0, 0)
        lay_l.setSpacing(4)
        self.cap_closed = self._caption("已平仓明细（一轮 = 从建仓到清仓，成本按移动加权平均）")
        lay_l.addWidget(self.cap_closed)
        self.tbl_closed = self._table(self.H_CLOSED)
        self.tbl_closed.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.tbl_closed.cellClicked.connect(lambda *_: None)
        lay_l.addWidget(self.tbl_closed)
        split.addWidget(wrap_l)

        wrap_r = QWidget()
        lay_r = QVBoxLayout(wrap_r)
        lay_r.setContentsMargins(0, 0, 0, 0)
        lay_r.setSpacing(4)
        self.cap_open = self._caption("当前持仓（成本为人民币，现价为最新价）")
        lay_r.addWidget(self.cap_open)
        self.tbl_open = self._table(self.H_OPEN)
        self.tbl_open.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        lay_r.addWidget(self.tbl_open)
        split.addWidget(wrap_r)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        self.split = split
        root.addWidget(split, 3)

        self.chart = PnlChart()
        self.tbl_code = self._table(self.H_CODE)
        self.tbl_code.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        wrap_c = QWidget()
        lay_c = QVBoxLayout(wrap_c)
        lay_c.setContentsMargins(0, 4, 0, 0)
        lay_c.addWidget(self.tbl_code)

        self.bottom = QTabWidget()
        self.bottom.addTab(self.chart, "累计已实现盈亏")
        self.bottom.addTab(wrap_c, "按标的汇总")
        root.addWidget(self.bottom, 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._fit_split)

    @staticmethod
    def _caption(text):
        lb = QLabel(text)
        lb.setStyleSheet("color:#55606b; font-size:11px; padding-left:2px;")
        return lb

    @staticmethod
    def _table(head):
        # 复盘结果也允许排序：同一批交易按盈亏、按标的、按日期重新看
        return SortTable(list(head), row_h=23, last_stretch=True)

    @staticmethod
    def _put(table, r, c, text, color=None, align=None):
        item = CellItem(text, color=QColor(color) if color else None)
        item.setTextAlignment(align or Qt.AlignmentFlag.AlignCenter)
        table.setItem(r, c, item)

    # ------------------------------------------------------------------ #
    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择交割单", self._last_dir,
            "交割单 (*.xls *.xlsx *.csv *.txt);;所有文件 (*)")
        if not path:
            return
        self._last_dir = os.path.dirname(path)
        self.status.setText("解析中…")
        self.open_btn.setEnabled(False)
        self._worker = Worker(trades_io.load_statement, path)
        self._worker.finished.connect(self.on_loaded)
        self._worker.error.connect(self.on_fail)
        self._worker.start()

    @_guard
    def on_fail(self, msg):
        self.open_btn.setEnabled(True)
        self.status.setText("导入失败")
        QMessageBox.warning(self, "导入失败", str(msg))

    @_guard
    def on_loaded(self, payload):
        recs, meta = payload
        self.meta = meta
        self.res = trades_io.rebuild(recs)
        self.open_btn.setEnabled(True)
        self.fx = feed.get_hkd_cny() if any(
            p.get("market") == "HK" for p in self.res["open"]) else 0.0
        self.file_lbl.setText(
            f"{os.path.basename(meta['path'])}　{meta['start']} ~ {meta['end']}　"
            f"成交 {meta['records']} 条　{meta['codes']} 只"
            + (f"（含港股 {meta['n_hk']}）" if meta.get("n_hk") else ""))
        warn = self.res["warns"]
        self._base_status = f"已导入 · 跳过非交易 {meta['skipped']} 条"
        if not meta.get("has_fee"):
            self._base_status += " · 该券商不带费用列，费用按万2.5+印花税估算"
        if warn:
            self._base_status += f" · {len(warn)} 条提示"
        self.status.setText(self._base_status)
        self._fill()

        codes = [p["code"] for p in self.res["open"]]
        if codes:
            self._qworker = Worker(get_quotes, codes)
            self._qworker.finished.connect(self.on_quotes)
            self._qworker.error.connect(lambda _e: None)
            self._qworker.start()

    @_guard
    def on_quotes(self, q):
        self.quotes = q or {}
        self._fill()

    # ------------------------------------------------------------------ #
    def _last_px(self, code: str, market: str = "A"):
        """现价统一折成人民币（港股乘汇率）。"""
        q = self.quotes.get(code) or {}
        px = q.get("price")
        if px and market == "HK":
            px = px * (self.fx or 0.85)
        return px

    def _float_pnl(self):
        tot, cost = 0.0, 0.0
        for p in (self.res["open"] if self.res else []):
            px = self._last_px(p["code"], p.get("market", "A"))
            cost += p["cost"]
            if px:
                tot += (px - p["cost_price"]) * p["qty"]
        return tot, cost

    def _fill(self):
        if self.res is None:
            return
        s = self.res["summary"]
        flt, flt_cost = self._float_pnl()
        realized = s["realized_all"]
        total = realized + flt
        fee_lbl = "费用" if self.meta and self.meta.get("has_fee") else "费用(估)"
        cells = [
            ("已实现盈亏", f"{realized:+,.2f}", realized),
            ("持仓浮盈", f"{flt:+,.2f}" if flt_cost else "—", flt),
            ("合计盈亏", f"{total:+,.2f}" if flt_cost else f"{realized:+,.2f}", total),
            ("胜率%", f"{s['win_rate']:.2f}", None),
            ("平仓轮次", f"{s['n_closed']}", None),
            ("盈亏比", f"{s['pl_ratio']:.2f}" if s["pl_ratio"] else "—", None),
            ("平均持股", f"{s['avg_days']:.1f} 天", None),
            (fee_lbl, f"{s['total_fee']:,.2f}", None),
        ]
        for i, (k, v, num) in enumerate(cells):
            r, c = divmod(i, 2)
            self._put(self.kpi, r, c * 2, k, "#55606b",
                      Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            col = None
            if num is not None and num != 0:
                col = UP if num > 0 else DOWN
            self._put(self.kpi, r, c * 2 + 1, v, col,
                      Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        closed_all = self.res["closed"]
        self.cap_closed.setText(
            f"已平仓 {s['n_closed']} 轮（{s['realized']:+,.0f}）　"
            f"盈 {sum(1 for t in closed_all if t['pnl'] > 0)} / 亏 {sum(1 for t in closed_all if t['pnl'] <= 0)}"
            f"　平均赚 {s['avg_win']:,.0f} / 平均亏 {s['avg_loss']:,.0f}"
            f"　｜　持仓中 {s['n_open']} 只另已卖部分 {realized - s['realized']:+,.0f}，"
            f"两者相加 = 上方已实现 {realized:+,.0f}")
        self.cap_open.setText(
            f"当前持仓　{s['n_open']} 只 · 成本 {flt_cost:,.0f}"
            + (f" · 浮盈 {flt:+,.0f}" if flt_cost else "")
            + (f"　（港股按 {self.fx:.4f} 折算）" if self.fx else ""))

        closed = list(closed_all)
        mode = self.order_box.currentText()
        if mode == "按盈亏":
            closed.sort(key=lambda t: -t["pnl"])
        elif mode == "按收益率":
            closed.sort(key=lambda t: -t["ret"])
        else:
            closed.sort(key=lambda t: t["exit_date"], reverse=True)

        self.tbl_closed.setRowCount(len(closed))
        for r, t in enumerate(closed):
            col = UP if t["pnl"] > 0 else (DOWN if t["pnl"] < 0 else None)
            self._put(self.tbl_closed, r, 0, t["code"].replace("HK", "HK"), "#55606b")
            self._put(self.tbl_closed, r, 1, t["name"])
            self._put(self.tbl_closed, r, 2, str(t["entry_date"].date()))
            self._put(self.tbl_closed, r, 3, str(t["exit_date"].date()))
            self._put(self.tbl_closed, r, 4, f"{t['buy_amt']:,.0f}")
            self._put(self.tbl_closed, r, 5, f"{t['sell_amt']:,.0f}")
            self._put(self.tbl_closed, r, 6, f"{t['pnl']:+,.0f}", col)
            self._put(self.tbl_closed, r, 7, f"{t['ret']:+.2f}", col)
            self._put(self.tbl_closed, r, 8, str(t["days"]), "#55606b")

        opens = sorted(self.res["open"],
                       key=lambda p: (self._last_px(p["code"], p.get("market", "A")) or 0) * p["qty"],
                       reverse=True)
        self.tbl_open.setRowCount(len(opens))
        for r, p in enumerate(opens):
            px = self._last_px(p["code"], p.get("market", "A"))
            if px:
                pnl = (px - p["cost_price"]) * p["qty"]
                pct = (px / p["cost_price"] - 1) * 100 if p["cost_price"] else 0.0
                col = UP if pnl > 0 else (DOWN if pnl < 0 else None)
                pnl_txt, pct_txt, px_txt = f"{pnl:+,.0f}", f"{pct:+.2f}", f"{px:.3f}"
                mv = f"{px * p['qty']:,.0f}"
            else:
                col, pnl_txt, pct_txt, px_txt, mv = None, "—", "—", "—", "—"
            self._put(self.tbl_open, r, 0, p["code"], "#55606b")
            self._put(self.tbl_open, r, 1, p["name"])
            self._put(self.tbl_open, r, 2, f"{p['qty']:,.0f}")
            self._put(self.tbl_open, r, 3, f"{p['cost_price']:.3f}")
            self._put(self.tbl_open, r, 4, px_txt)
            self._put(self.tbl_open, r, 5, mv)
            self._put(self.tbl_open, r, 6, pnl_txt, col)
            self._put(self.tbl_open, r, 7, pct_txt, col)

        self._fill_by_code(closed, realized)
        self.chart.set_curve(trades_io.equity_curve(closed))
        base = getattr(self, "_base_status", "")
        self.status.setText(base + (" · 拉取现价…" if (opens and not self.quotes) else ""))
        QTimer.singleShot(0, self._fit_split)

    def _fill_by_code(self, closed, realized):
        agg: dict[str, dict] = {}
        for t in closed:
            a = agg.setdefault(t["code"], {"name": t["name"], "n": 0, "win": 0,
                                           "buy": 0.0, "pnl": 0.0, "ret": 0.0,
                                           "market": t.get("market", "A")})
            a["n"] += 1
            a["win"] += 1 if t["pnl"] > 0 else 0
            a["buy"] += t["buy_amt"]
            a["pnl"] += t["pnl"]
            a["ret"] += t["ret"]
        rows = sorted(agg.items(), key=lambda kv: -kv[1]["pnl"])
        self.tbl_code.setRowCount(len(rows))
        for r, (code, a) in enumerate(rows):
            col = UP if a["pnl"] > 0 else (DOWN if a["pnl"] < 0 else None)
            share = a["pnl"] / realized * 100 if realized else 0.0
            self._put(self.tbl_code, r, 0, code, "#55606b")
            self._put(self.tbl_code, r, 1, a["name"])
            self._put(self.tbl_code, r, 2, str(a["n"]))
            self._put(self.tbl_code, r, 3, f"{a['win']}/{a['n']}",
                      UP if a["win"] == a["n"] else (DOWN if not a["win"] else None))
            self._put(self.tbl_code, r, 4, f"{a['buy']:,.0f}")
            self._put(self.tbl_code, r, 5, f"{a['pnl']:+,.0f}", col)
            self._put(self.tbl_code, r, 6, f"{a['ret'] / a['n']:+.2f}", col)
            self._put(self.tbl_code, r, 7, f"{share:+.1f}", col)

    def _fit_split(self):
        """表格区高度按行数封顶，数据少时把空间让给下面的曲线。"""
        try:
            n = max(self.tbl_closed.rowCount(), self.tbl_open.rowCount(), 6)
            want = 40 + 23 * n + 8
            limit = int(self.height() * 0.62) if self.height() > 200 else want
            self.split.setMaximumHeight(max(150, min(want, limit)))
            self.split.setMinimumHeight(150)
        except Exception:
            _trace("调整表格高度失败:\n" + traceback.format_exc())


import ui_tabs


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuantDesk · 回测 / 轮动 / 多市场 / 选股 / 事件 / 审计 / 风险")
        self.resize(1420, 900)
        self.setMinimumSize(1120, 720)
        tabs = QTabWidget()
        self.backtest_tab = ui_tabs.BacktestTab(EquityChart())
        tabs.addTab(self.backtest_tab, "策略回测")
        self.rotation_tab = ui_tabs.RotationTab()
        tabs.addTab(self.rotation_tab, "可转债轮动")
        self.multimkt_tab = ui_tabs.MultimktTab()
        tabs.addTab(self.multimkt_tab, "多市场净值")
        tabs.addTab(ui_tabs.SelectTab(), "全市场选股")
        tabs.addTab(ui_tabs.EventTab(), "事件雷达")
        tabs.addTab(ui_tabs.HolderTab(), "机构持仓")
        tabs.addTab(ui_tabs.AuditTab(), "行为审计")
        tabs.addTab(ui_tabs.RiskTab(), "组合风险")
        self.setCentralWidget(tabs)

    def closeEvent(self, event):
        _trace("主窗口收到 closeEvent（窗口被关闭，进程随之退出）")
        super().closeEvent(event)


# --------------------------------------------------------------------------- #
# 无界面自检（开发/打包后验证链路）
# --------------------------------------------------------------------------- #
def selftest():
    ok = True
    print("=== QuantDesk 自检（真实 A 股数据）===")

    wl = load_watchlist()
    print(f"[OK] 股票池 {len(wl)} 只（前 3: "
          f"{', '.join(i['name'] + '(' + i['code'] + ')' for i in wl[:3])}）")

    t0 = datetime.datetime.now()
    quotes = get_quotes([i["code"] for i in wl[:10]])
    dt = (datetime.datetime.now() - t0).total_seconds()
    if quotes:
        k, v = next(iter(quotes.items()))
        print(f"[OK] 批量行情 {len(quotes)}/10 只，耗时 {dt:.2f}s，"
              f"样例 {v['name']} {v['price']} {v['pct']:+.2f}%")
    else:
        print("[FAIL] 批量行情为空")
        ok = False

    code, name = wl[0]["code"], wl[0]["name"]
    for period in ALL_PERIODS:
        df = get_bars(code, period)
        if df.empty:
            print(f"[FAIL] {period} 无数据")
            ok = False
        else:
            print(f"[OK] {name} {period:<6} {len(df):>5} 根  "
                  f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}  "
                  f"收 {df['close'].iloc[-1]:.2f}")

    hist = feed.get_history(code, DEFAULT_START, DEFAULT_END)
    if hist.empty:
        print("[FAIL] 回测历史数据为空")
        ok = False
    else:
        import engine
        import metrics
        sig = engine.make_signal(hist, "双均线", fast=5, slow=20)
        res = engine.run(hist, sig, INIT_CASH, code=code, name=name)
        m = metrics.summarize(res, INIT_CASH)
        print(f"[OK] 新引擎回测 {len(hist)} 条日线：总收益 {m['total_return']}%  "
              f"年化 {m['annual_return']}%  回撤 {m['max_drawdown']}%  "
              f"胜率 {m['win_rate']}%  {m['trade_count']} 笔  夏普 {m['sharpe']}")
        print(f"     卡玛 {m['calmar']}｜盈亏比 {m['profit_loss_ratio']}｜"
              f"盈利因子 {m['profit_factor']}｜期望 {m['expectancy']} 元｜"
              f"水下最长 {m['longest_underwater']} 日")
        print(f"     赢的持 {m['avg_win_hold']} 日／亏的持 {m['avg_loss_hold']} 日｜"
              f"暴露 {m['exposure']}%｜换手 {m['turnover']}x｜费用 {m['total_fees']} 元")
        if res["trades"]:
            t = res["trades"][0]
            print(f"     交易级字段 {len(t)} 个，样例：{t['entry_date']} 买 "
                  f"{t['entry_price']}×{t['shares']} 股 → {t['exit_date']} 卖 "
                  f"{t['exit_price']}，持 {t['hold_days']} 日，净 {t['net_pnl']} 元，"
                  f"最高浮盈 {t['mfe_pct']}%，回吐 {t['giveback_pct']}%")
        print(f"[OK] 结论翻译 {len(metrics.verdict(m))} 条："
              f"{metrics.verdict(m)[0][0]}")

        grid = engine.param_grid(hist, "双均线", INIT_CASH, code=code, name=name,
                                 max_runs=12)
        if grid:
            best = max(g["annual"] for g in grid)
            worst = min(g["annual"] for g in grid)
            print(f"[OK] 参数扫描 {len(grid)} 组：年化最好 {best:.2f}% / 最差 {worst:.2f}%")
        wf = engine.walk_forward(hist, "双均线", {"fast": 5, "slow": 20}, 4,
                                 INIT_CASH, code=code, name=name)
        if wf:
            pos = sum(1 for r in wf if r["return"] > 0)
            print(f"[OK] 滚动样本外 {len(wf)} 段，{pos} 段盈利："
                  + " / ".join(f"{r['return']:+.1f}%" for r in wf))

    # 交割单解析链路（造一份最小样例，验证解析 + 配对 + 汇总）
    try:
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), "qd_selftest_stmt.csv")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("成交日期,证券代码,证券名称,买卖标志,成交价格,成交数量,成交金额,手续费,印花税,过户费\n")
            f.write("2026-03-02,600000,浦发银行,买入,8.50,1000,8500,5.00,0,0.09\n")
            f.write("2026-04-02,600000,浦发银行,卖出,9.10,1000,9100,5.00,4.55,0.09\n")
        recs, meta = trades_io.load_statement(tmp)
        reb = trades_io.rebuild(recs)
        st = reb["summary"]
        print(f"[OK] 交割单解析 {meta['records']} 条成交 -> {st['n_closed']} 笔平仓，"
              f"已实现 {st['realized']:+.2f} 元，胜率 {st['win_rate']}%")

        import audit
        d = audit.diagnose(recs, reb["closed"], st)
        print(f"[OK] 行为审计 {len(d['biases'])} 类偏误，纪律分 {d['score']}："
              + "；".join(f"{b['name']}[{b['severity']}]" for b in d["biases"]))
        rules = audit.extract_rules(reb["closed"])
        print(f"[OK] 规则反推：止盈 {rules['take_profit']}% / 止损 "
              f"{rules['stop_loss']}% / 最长持 {rules['max_hold']} 天")
        for o in audit.overall(reb["closed"], st)[:2]:
            print(f"     {o['title']}：{o['text'][:60]}…")
        os.remove(tmp)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 交割单解析: {exc}")
        ok = False

    # 组合风险链路
    try:
        import datasource
        import riskx
        prov = datasource.active()
        print(f"[OK] 数据源 {datasource.status_line()}")
        hold = [{"code": wl[0]["code"], "name": wl[0]["name"], "qty": 2000,
                 "cost_price": 0},
                {"code": wl[1]["code"], "name": wl[1]["name"], "qty": 2000,
                 "cost_price": 0},
                {"code": wl[2]["code"], "name": wl[2]["name"], "qty": 1000,
                 "cost_price": 0}]
        cache = {}

        def _lookup(c):
            if c not in cache:
                cache[c] = prov.bars(c, "20250101", "20261231")
            return cache[c]

        ind = prov.industry_map([h["code"] for h in hold])
        r = riskx.analyze(hold, _lookup, lambda c: ind.get(c, ""), lookback=250)
        if r.get("ok"):
            v = r.get("var") or {}
            print(f"[OK] 组合风险 {r['n']} 只：HHI {r['hhi']}，有效分散度 "
                  f"{r['eff_n']} 只，年化波动 {v.get('ann_vol', 0)}%，"
                  f"VaR95 {v.get('var95', 0)}%，ES95 {v.get('es95', 0)}%")
            print(f"     提醒：{r['warns'][0]}")
        else:
            print(f"[FAIL] 组合风险: {r.get('msg')}")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 组合风险: {exc}")
        ok = False

    # 多市场统一净值链路
    try:
        import multimkt
        r = multimkt.blend(multimkt.default_mix(), "20240101", DEFAULT_END,
                           bench=("cn_index", "000300"))
        if r.get("ok"):
            m = r["metrics"]
            fx = r.get("fx_total", 0.0)
            print(f"[OK] 多市场净值 {len(r['series'])} 个市场 "
                  f"{len(r['dates'])} 个交易日：总收益 "
                  f"{m['total_return'] * 100:+.2f}%，年化 {m['annual_return'] * 100:+.2f}%，"
                  f"回撤 {m['max_drawdown'] * 100:.2f}%，汇率影响 {fx * 100:+.2f}pp")
            for s in r["series"]:
                print(f"     {s['market']:<4}{s['name']:<10}权重 {s['w'] * 100:>5.1f}%  "
                      f"本币 {s['ret_local'] * 100:>+7.2f}%  折人民币 "
                      f"{s['ret_cny'] * 100:>+7.2f}%  汇率 {s['fx'] * 100:>+6.2f}%")
        else:
            print(f"[FAIL] 多市场净值: {r.get('msg')}")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 多市场净值: {exc}")
        ok = False

    # 可转债轮动链路（走缓存；首次跑会慢一些）
    try:
        import rotation
        d = rotation.build_panel("20240101", DEFAULT_END)
        b = rotation.load_bench("20240101", DEFAULT_END)
        rr = rotation.backtest(d["panel"], d["meta"], strategy="double_low",
                               freq="W", top_n=15, min_rating="AA-", bench=b)
        if rr.get("ok"):
            m = rr["metrics"]
            ex = (rr.get("excess") or {}).get("annual", 0.0)
            print(f"[OK] 可转债轮动 {d['n']} 只池子：年化 "
                  f"{m['annual_return'] * 100:+.2f}%，回撤 "
                  f"{m['max_drawdown'] * 100:.2f}%，超额 {ex * 100:+.2f}pp，"
                  f"调仓 {rr['stats']['rebalances']} 次")
            top = rr["holdings"][0] if rr["holdings"] else None
            if top:
                print(f"     最新首仓 {top['name']}({top['code']}) "
                      f"价格 {top['price']} 溢价率 {top['premium']}% "
                      f"因子值 {top['score']}")
        else:
            print(f"[FAIL] 可转债轮动: {rr.get('msg')}")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 可转债轮动: {exc}")
        ok = False

    print("自检通过。" if ok else "自检存在失败项。")
    return 0 if ok else 1


def _setup_qt_plugin_path():
    """冻结版兜底：自动定位包内 Qt 平台插件(qwindows.dll)，避免 Qt 找不到插件直接 abort。"""
    if not getattr(sys, "frozen", False):
        return
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        return
    try:
        for root, _dirs, files in os.walk(base):
            if "qwindows.dll" in files:
                os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", root)
                os.environ.setdefault("QT_PLUGIN_PATH", os.path.dirname(root))
                _trace("找到 qwindows.dll: %s" % root)
                return
        _trace("!!! 未在包内找到 qwindows.dll，扫描根目录 %s" % base)
    except Exception:
        _trace("插件路径扫描异常:\n" + traceback.format_exc())


def _crash_log_path():
    candidates = []
    try:
        candidates.append(os.path.join(os.path.expanduser("~"), "Desktop"))
    except Exception:
        pass
    candidates.append(tempfile.gettempdir())
    for d in candidates:
        try:
            if os.path.isdir(d) and os.access(d, os.W_OK):
                return os.path.join(d, "QuantDesk_crash.log")
        except Exception:
            continue
    return os.path.join(tempfile.gettempdir(), "QuantDesk_crash.log")


def _report_error(title, tb_text):
    """把启动错误写到日志，并弹一个不依赖 Qt 的原生对话框。

    启动早期报错时 Qt 可能还没起来，所以不能用 QMessageBox。Windows 走
    user32、macOS 走 osascript，其它平台只落日志（终端里能看到堆栈）。
    """
    try:
        with open(_crash_log_path(), "w", encoding="utf-8") as f:
            f.write(f"QuantDesk 启动错误（{title}）：\n\n{tb_text}\n")
    except Exception:
        pass
    msg = (tb_text or title)[:2500]
    try:
        if sys.platform.startswith("win"):
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, msg, "QuantDesk 启动失败", 0x10)
        elif sys.platform == "darwin":
            import subprocess
            script = (
                'display alert "QuantDesk 启动失败" message %s as critical'
                % _osascript_quote(msg[:1200])
            )
            subprocess.run(["osascript", "-e", script], timeout=30,
                           capture_output=True)
    except Exception:
        pass


def _osascript_quote(s):
    """把文本转成 AppleScript 字符串字面量（转义反斜杠和双引号）。"""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _apply_light_theme(app):
    """强制浅色主题。

    Windows 处于深色模式时，Qt6 会继承深色调色板（文字变白）。本项目界面是浅色
    底，白字打在浅底上等于全看不见（下拉框/输入框内容消失）。这里直接把调色板
    钉死为浅色，不再依赖系统主题。
    """
    try:
        from PyQt6.QtGui import QPalette

        app.setStyle("Fusion")
        pal = QPalette()
        c = QColor
        pal.setColor(QPalette.ColorRole.Window, c("#f5f6f8"))
        pal.setColor(QPalette.ColorRole.WindowText, c("#1f2d3d"))
        pal.setColor(QPalette.ColorRole.Base, c("#ffffff"))
        pal.setColor(QPalette.ColorRole.AlternateBase, c("#f7f9fb"))
        pal.setColor(QPalette.ColorRole.Text, c("#1f2d3d"))
        pal.setColor(QPalette.ColorRole.Button, c("#eef1f5"))
        pal.setColor(QPalette.ColorRole.ButtonText, c("#1f2d3d"))
        pal.setColor(QPalette.ColorRole.Highlight, c("#0b63b6"))
        pal.setColor(QPalette.ColorRole.HighlightedText, c("#ffffff"))
        pal.setColor(QPalette.ColorRole.ToolTipBase, c("#ffffff"))
        pal.setColor(QPalette.ColorRole.ToolTipText, c("#1f2d3d"))
        pal.setColor(QPalette.ColorRole.PlaceholderText, c("#98a2ae"))
        app.setPalette(pal)
        _trace("浅色调色板已应用（Fusion）")
    except Exception:
        _trace("浅色调色板应用失败:\n" + traceback.format_exc())


def check_statement(path: str) -> int:
    """命令行解析交割单，打印口径统计（冻结版验证券商格式用）。"""
    try:
        recs, meta = trades_io.load_statement(path)
    except Exception as exc:
        print(f"解析失败：{exc}")
        return 2
    res = trades_io.rebuild(recs)
    s = res["summary"]
    print(f"文件　{os.path.basename(path)}")
    print(f"区间　{meta['start']} ~ {meta['end']}　成交 {meta['records']} 条　"
          f"{meta['codes']} 只（港股 {meta['n_hk']}）")
    print(f"跳过　{meta['skipped']} 条非交易行　含费用列：{'是' if meta['has_fee'] else '否（已估算）'}")
    print(f"事件　{meta['events']}")
    print(f"清仓　{s['n_closed']} 轮　胜率 {s['win_rate']:.2f}%　盈亏比 {s['pl_ratio']:.2f}")
    print(f"已实现 {s['realized_all']:,.2f}　（其中已清仓轮次 {s['realized']:,.2f}）")
    print(f"持仓　{s['n_open']} 只　成本 {sum(o['cost'] for o in res['open']):,.2f}")
    print(f"费用　{s['total_fee']:,.2f}　分红 {sum(t.get('dividend', 0) for t in res['closed']):,.2f}")
    if res["warns"]:
        print(f"提示　{len(res['warns'])} 条，前 3：")
        for w in res["warns"][:3]:
            print("      " + w)
    return 0


def main():
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="QuantDesk 自研 A 股 看盘/回测 桌面程序")
    parser.add_argument("--selftest", action="store_true", help="无界面自检（验证数据/回测链路）")
    parser.add_argument("--check-statement", metavar="文件", default=None,
                        help="只解析一份交割单并打印统计，用于验证券商格式")
    args = parser.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.check_statement:
        sys.exit(check_statement(args.check_statement))

    _trace("进入 main()，准备初始化 Qt")
    try:
        _setup_qt_plugin_path()
        app = QApplication(sys.argv)
        _trace("QApplication 创建成功")
        _apply_light_theme(app)
        app.setStyleSheet(CSS)
        win = MainWindow()
        _trace("主窗口构建成功")
        win.show()
        _trace("主窗口 show() 完成（isVisible=%r），进入事件循环" % win.isVisible())
        rc = app.exec()
        _trace("事件循环正常结束 rc=%r" % rc)
        sys.exit(rc)
    except Exception:
        tb = traceback.format_exc()
        _trace("GUI 启动抛异常:\n" + tb)
        _report_error("GUI 启动异常", tb)
        raise


if __name__ == "__main__":
    main()
