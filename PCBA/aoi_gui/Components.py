# aoi_gui/Components.py
"""Reusable widgets. TitleBar replaces the OS chrome on the frameless window."""

from __future__ import annotations

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class TitleBar(QWidget):
    """Draggable title bar with theme toggle, minimise, maximise and close."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TitleBar")
        self._parent = parent
        self._drag_pos: QPoint | None = None
        self.setFixedHeight(40)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 6, 4)
        layout.setSpacing(4)

        self.titleLabel = QLabel("Op-Ins AOI System")
        self.titleLabel.setObjectName("WindowTitle")
        layout.addWidget(self.titleLabel)
        layout.addStretch(1)

        self.themeBtn = self._make_button("\u2600", "Toggle light/dark theme")
        self.themeBtn.clicked.connect(self._toggle_theme)
        layout.addWidget(self.themeBtn)

        self.minBtn = self._make_button("\u2013", "Minimise")
        self.minBtn.clicked.connect(lambda: self._parent.showMinimized())
        layout.addWidget(self.minBtn)

        self.maxBtn = self._make_button("\u25a1", "Maximise / restore")
        self.maxBtn.clicked.connect(self._toggle_max)
        layout.addWidget(self.maxBtn)

        self.closeBtn = self._make_button("\u2715", "Close")
        self.closeBtn.setObjectName("TitleButtonClose")
        self.closeBtn.clicked.connect(lambda: self._parent.close())
        layout.addWidget(self.closeBtn)

    def _make_button(self, text: str, tip: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("TitleButton")
        btn.setToolTip(tip)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFlat(True)
        return btn

    def _toggle_theme(self):
        if self._parent and hasattr(self._parent, "toggle_theme"):
            self._parent.toggle_theme()

    def _toggle_max(self):
        if not self._parent:
            return
        if self._parent.isMaximized():
            self._parent.showNormal()
            self.maxBtn.setText("\u25a1")
        else:
            self._parent.showMaximized()
            self.maxBtn.setText("\u2750")

    # --- window dragging ---
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._parent:
            self._drag_pos = event.globalPos() - self._parent.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.LeftButton and self._parent:
            if self._parent.isMaximized():
                self._parent.showNormal()
                self.maxBtn.setText("\u25a1")
            self._parent.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event):
        self._toggle_max()
