"""表格基础设施：可排序表格、数值单元格、行详情弹窗。

为什么不直接用 setSortingEnabled(True)：
  · 它只认 DisplayRole 字符串比较，"1,234" 会排在 "999" 前面。
  · 动态填充时（先填公告、再填机会）已经排过序的行会被新行插到中间，
    表头箭头指着一个顺序，实际显示是另一个顺序。
  · 它不接受「空值永远沉底」这种口径，NaN 会跟 "-" 一起乱漂。

这里的做法是每个格子带一个 sort_val：数字按数值比，文本按字符串比，
取不到数的一律沉底。排序由控件自己执行，填充完自动按当前排序列重排。
"""
from __future__ import annotations

import re
import time

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QBrush, QColor, QDesktopServices, QGuiApplication, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QSizePolicy, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
)

# 显示文本 -> 数值。允许千分位、正负号、百分号、pp 与常见中文单位
NUM_RE = re.compile(
    r"^[+\-]?[\d,]*\.?\d+(?:[eE][+\-]?\d+)?\s*(?:pp|%|亿|万|元|倍|天|笔|只|条|次|个|股)?$")
_SUFFIX = re.compile(r"(pp|%|亿|万|元|倍|天|笔|只|条|次|个|股)$")


def as_number(text):
    """'1,234.5 万' / '-12.3%' / '+8.0pp' -> float；不像数字就返回 None。"""
    if text is None:
        return None
    s = str(text).strip()
    if not s or not NUM_RE.match(s):
        return None
    s = s.replace(",", "")
    s = _SUFFIX.sub("", s)
    try:
        return float(s)
    except ValueError:
        return None


class CellItem(QTableWidgetItem):
    """带排序键的单元格。sort_val 为 None 表示按文本排。"""

    def __init__(self, text, sort_val=None, color=None, align_right=False,
                 tooltip=None):
        super().__init__(str(text))
        self.sort_val = as_number(text) if sort_val is None else sort_val
        if color is not None:
            self.setForeground(QBrush(color))
        if align_right:
            self.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                  | Qt.AlignmentFlag.AlignVCenter)
        if tooltip:
            self.setToolTip(str(tooltip))


class SortTable(QTableWidget):
    """点表头排序、列宽可拖、列可换位、数值按数值比。

    fill 结束后 60ms 自动按当前排序列重排，所以填充代码不用关心顺序。
    """

    def __init__(self, headers, row_h=22, last_stretch=False, widths=None,
                 stretch_col=-1, parent=None):
        super().__init__(0, len(headers), parent)
        self._headers = [str(h) for h in headers]
        self.setHorizontalHeaderLabels(list(self._headers))
        self._widths = dict(widths or {})
        self._stretch_col = stretch_col
        self._last_set = 0.0
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setWordWrap(False)

        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setSectionsMovable(True)          # 列可以拖着换位置
        hh.setSectionsClickable(True)
        hh.setStretchLastSection(bool(last_stretch) and stretch_col < 0)
        hh.setDefaultSectionSize(104)
        hh.setMinimumSectionSize(48)
        # QSS 给表头刷了深色底后，原生排序小三角会被样式吃掉，
        # 所以排序方向直接写在表头文字上，任何主题下都看得见。
        hh.setSortIndicatorShown(False)
        hh.sectionClicked.connect(self._on_head_clicked)

        vh = self.verticalHeader()
        vh.setVisible(False)
        vh.setDefaultSectionSize(row_h)
        vh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        self._sort_col = -1
        self._sort_desc = True
        self._pending = False
        self._sorting = False
        self._fitted = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._flush)

        if 0 <= stretch_col < len(headers):
            hh.setSectionResizeMode(stretch_col, QHeaderView.ResizeMode.Stretch)
        for c, w in self._widths.items():
            self.setColumnWidth(c, w)

    # ---------------------------------------------------------------- 排序
    def _on_head_clicked(self, col: int):
        if self._sort_col == col:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_col, self._sort_desc = col, True
        self.apply_sort()

    def sort_by(self, col: int, desc: bool = True):
        """代码里指定默认排序，比如金额列降序。"""
        self._sort_col, self._sort_desc = int(col), bool(desc)
        self.apply_sort()

    def setItem(self, r, c, item):  # noqa: N802  (Qt 命名)
        super().setItem(r, c, item)
        self._last_set = time.monotonic()
        if not self._sorting and self._sort_col >= 0:
            self._schedule()

    def _schedule(self):
        if not self._pending:
            self._pending = True
        self._timer.start()          # 每次填充都重置，避免填一半就排

    def _flush(self):
        # 填充中途也会触发定时器，此时按半截数据算列宽会把列压得很窄，
        # 所以等最后一批 setItem 之后 120ms 静默期再定宽、再排序。
        if time.monotonic() - self._last_set < 0.12:
            self._timer.start(60)
            return
        self._pending = False
        if not self._fitted and self.rowCount():
            self.fit_columns()
            self._fitted = True
        self.apply_sort()

    def apply_sort(self):
        n = self.rowCount()
        col = self._sort_col
        if n <= 1 or col < 0 or col >= self.columnCount():
            return
        good, blank = [], []
        for r in range(n):
            it = self.item(r, col)
            txt = "" if it is None else str(it.text()).strip()
            if txt in ("", "—", "-", "nan", "None"):
                blank.append(r)                  # 空值永远沉底，不管升序降序
                continue
            v = getattr(it, "sort_val", None)
            good.append(((0, v) if v is not None else (1, txt), r))
        good.sort(key=lambda x: x[0], reverse=self._sort_desc)
        self._mark_header(col, self._sort_desc)
        self.horizontalHeader().setSortIndicator(
            col, Qt.SortOrder.DescendingOrder if self._sort_desc
            else Qt.SortOrder.AscendingOrder)
        order = [r for _, r in good] + blank
        if order == list(range(n)):
            return

        self._sorting = True
        self.setUpdatesEnabled(False)
        try:
            snap = [[self.takeItem(r, c) for c in range(self.columnCount())]
                    for r in range(n)]
            for new_r, old_r in enumerate(order):
                for c, it in enumerate(snap[old_r]):
                    if it is not None:
                        super().setItem(new_r, c, it)
        finally:
            self.setUpdatesEnabled(True)
            self._sorting = False

    def _mark_header(self, col: int, desc: bool):
        mark = " ▼" if desc else " ▲"
        for c in range(min(len(self._headers), self.columnCount())):
            it = self.horizontalHeaderItem(c)
            if it is None:
                continue
            it.setText(self._headers[c] + (mark if c == col else ""))

    def fit_columns(self, min_w=56, max_w=320):
        """按内容给一次初宽，指定过宽度的列按指定值；之后用户拖宽不再覆盖。"""
        self.resizeColumnsToContents()
        hh = self.horizontalHeader()
        for c in range(self.columnCount()):
            if c == self._stretch_col:
                continue
            w = self._widths.get(c, hh.sectionSize(c))
            self.setColumnWidth(c, max(min_w, min(w, max_w)))

    # ---------------------------------------------------------------- 取值
    def raw(self, r: int, c: int):
        it = self.item(r, c)
        return None if it is None else getattr(it, "sort_val", None)

    def bind(self, row: int, obj):
        """把行对应的业务对象挂到第一列，排序后仍然拿得到。"""
        it = self.item(row, 0)
        if it is not None:
            it.setData(Qt.ItemDataRole.UserRole, obj)

    def ref(self, row: int):
        it = self.item(row, 0)
        return None if it is None else it.data(Qt.ItemDataRole.UserRole)

    def row_of(self, ref) -> int:
        for r in range(self.rowCount()):
            if self.ref(r) == ref:
                return r
        return -1

    def clear_rows(self):
        self.setRowCount(0)


def make_table(headers, row_h=22, last_stretch=True) -> SortTable:
    return SortTable(headers, row_h=row_h, last_stretch=last_stretch)


def put(t: QTableWidget, r: int, c: int, text, color=None, align_right=False,
        sort_val=None, tooltip=None):
    t.setItem(r, c, CellItem(text, sort_val=sort_val, color=color,
                             align_right=align_right, tooltip=tooltip))


# ===================================================================== 详情弹窗

class DetailDialog(QDialog):
    """统一的「点开看具体内容」窗口。

    fields: [(字段名, 值), ...]  值不截断，长文本自动换行
    tables: [{"caption":..., "headers":[...], "rows":[[...]], "right":{列号}, "colors":{列号: fn}}]
    link:   原文地址，给了就出「打开原文」
    """

    def __init__(self, parent=None, title="详情", subtitle="", fields=None,
                 tables=None, link=None, copy_text=None, show_body=False):
        super().__init__(parent)
        self._link = link
        self._copy = copy_text or ""
        self.body_box = None
        self.setWindowTitle(title[:70] if title else "详情")
        self.resize(900, 640)

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)

        # 标签必须锁死高度，否则 QVBoxLayout 会把多余空间平均分给每个 QLabel，
        # 出现标题和正文之间各隔一百多像素的空洞。
        head = QLabel(title)
        head.setWordWrap(True)
        head.setStyleSheet("font-size:15px;font-weight:600;color:#141414;")
        head.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        v.addWidget(head)

        if subtitle:
            sub = QLabel(subtitle)
            sub.setWordWrap(True)
            sub.setStyleSheet("color:#666;font-size:12px;")
            sub.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            v.addWidget(sub)

        if fields:
            ft = SortTable(["字段", "值"], row_h=24, last_stretch=True)
            ft.setColumnWidth(0, 96)
            ft.setRowCount(len(fields))
            for i, (k, val) in enumerate(fields):
                put(ft, i, 0, k)
                it = CellItem("" if val is None else str(val))
                it.setToolTip("" if val is None else str(val))
                ft.setItem(i, 1, it)
            ft.resizeRowsToContents()
            try:
                ft.horizontalHeader().sectionClicked.disconnect()
                ft.horizontalHeader().setSortIndicatorShown(False)
            except TypeError:
                pass
            ft.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            ft.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            h = ft.horizontalHeader().height() + 23 * len(fields) + 6
            ft.setFixedHeight(min(h, 340))
            v.addWidget(ft, 0)
            self._fields = ft
        else:
            self._fields = None

        for spec in (tables or []):
            cap = QLabel(spec.get("caption", ""))
            cap.setStyleSheet("color:#444;font-weight:600;margin-top:4px;")
            v.addWidget(cap)
            rows = spec.get("rows") or []
            right = set(spec.get("right") or [])
            colors = spec.get("colors") or {}
            tb = SortTable(spec.get("headers") or [], row_h=23)
            tb.setRowCount(len(rows))
            for i, row in enumerate(rows):
                for c, cell in enumerate(row):
                    col = colors.get(c)
                    cobj = col(cell) if callable(col) else col
                    put(tb, i, c, cell, cobj, c in right)
            v.addWidget(tb, 1)
            tb.fit_columns()

        if show_body:
            self.body_box = QTextBrowser()
            self.body_box.setOpenExternalLinks(True)
            self.body_box.setStyleSheet(
                "font-size:12.5px; line-height:1.7; background:#ffffff;"
                "border:1px solid #dfe3e8;")
            self.body_box.setHtml(
                "<p style='color:#888'>正在加载公告正文…</p>")
            v.addWidget(self.body_box, 1)
        elif not tables:
            v.addStretch(1)

        bar = QHBoxLayout()
        bar.addStretch(1)
        if link:
            b = QPushButton("在浏览器打开原文")
            b.clicked.connect(self._open)
            bar.addWidget(b)
        if self._copy:
            b2 = QPushButton("复制全文")
            b2.clicked.connect(self._do_copy)
            bar.addWidget(b2)
        b3 = QPushButton("关闭")
        b3.clicked.connect(self.accept)
        b3.setDefault(True)
        bar.addWidget(b3)
        v.addLayout(bar)

        # 只有字段没有子表时，按内容收紧，别留半屏空白
        self.adjustSize()
        self.resize(max(self.width(), 880),
                    760 if show_body else min(self.height(), 760))

    def set_body(self, text: str, note: str = ""):
        if self.body_box is None:
            return
        if note:
            self.body_box.setHtml(
                f"<p style='color:#888'>{note}</p>"
                f"<pre style='white-space:pre-wrap;font-family:inherit'>{text}</pre>")
        else:
            self.body_box.setPlainText(text)

    def _open(self):
        if self._link:
            QDesktopServices.openUrl(QUrl(self._link))

    def _do_copy(self):
        QGuiApplication.clipboard().setText(self._copy)


def row_fields(row, names=None):
    """DataFrame 一行 -> [(字段, 值)]，跳过全空的列。"""
    import pandas as pd
    out = []
    keys = names or list(row.index)
    for k in keys:
        if k not in row.index:
            continue
        val = row[k]
        if isinstance(val, float) and pd.isna(val):
            continue
        txt = "" if (val is None or (isinstance(val, float) and pd.isna(val))) else str(val)
        if txt in ("NaT", "nan", "None"):
            continue
        out.append((str(k), txt))
    return out
