# aoi_gui/Components.py

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt

class TitleBar(QWidget):
    """A compact custom title bar extracted from MainWindow for reuse and clarity."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setObjectName('TitleBar')
        # Slightly taller to avoid border overlap and improve hit area
        self.setFixedHeight(40)
        # Drag state
        self._drag_pos = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self.title = QLabel(parent.windowTitle(), self)
        self.title.setObjectName('WindowTitle')
        self.title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        layout.addWidget(self.title)
        layout.addStretch()

        # Theme toggle (shows sun when in dark mode, moon when in light mode)
        theme_icon = '☀' if getattr(parent, 'current_theme', 'dark') == 'dark' else '🌙'
        self.themeBtn = QPushButton(theme_icon, self)
        self.themeBtn.setObjectName('TitleButton')
        self.themeBtn.setFixedSize(30, 30)
        layout.addWidget(self.themeBtn)

        # Title bar buttons (larger square buttons for better hit area)
        self.minBtn = QPushButton('_', self)
        self.maxBtn = QPushButton('❐', self)
        self.closeBtn = QPushButton('✕', self)
        for b in (self.minBtn, self.maxBtn, self.closeBtn):
            b.setObjectName('TitleButton')
            b.setFixedSize(30, 30)
            layout.addWidget(b)

        self.minBtn.clicked.connect(parent.showMinimized)
        self.maxBtn.clicked.connect(self._toggle_max)
        self.closeBtn.clicked.connect(parent.close)
        # Connect theme toggle to parent if available
        try:
            if hasattr(parent, 'toggle_theme'):
                self.themeBtn.clicked.connect(parent.toggle_theme)
        except Exception:
            pass

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.parent.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and not self.parent.isMaximized() and self._drag_pos is not None:
            try:
                self.parent.move(event.globalPos() - self._drag_pos)
                event.accept()
            except Exception:
                self._drag_pos = None

    def mouseDoubleClickEvent(self, event):
        self._toggle_max()

    def _toggle_max(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
            self.maxBtn.setText('❐')
        else:
            self.parent.showMaximized()
            self.maxBtn.setText('❐')
