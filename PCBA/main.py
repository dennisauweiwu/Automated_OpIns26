# main.py

import sys
from PyQt5.QtWidgets import QApplication
from aoi_gui.MainWindow import MainWindow  

if __name__ == '__main__':
    # Initialize Application Enivronment
    app = QApplication(sys.argv)
    
    # UI font 
    app.setStyleSheet("""
        /* --- 1. Base Widget (Global) --- */
        QWidget {
            background-color: #17181A; /* Lightened base (approx +8%) */
            color: #E8E8E8;            /* High-contrast near-white text */
            font-family: Avenir;
            font-size: 10pt;
        }

        /* --- 2. QGroupBox (Container Styles) --- */
        QGroupBox {
            border: 1px solid #34373A;
            border-radius: 6px;
            margin-top: 25px;
            padding-top: 18px;
            font-weight: normal;
            background-color: #1A1B1D;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            color: #9AA0A6; /* Muted title text */
            font-size: 9pt;
        }

        /* --- 3. QLabel#StatusIndicator --- */
        QLabel#StatusIndicator {
            font-size: 18pt;
            font-weight: bold;
            color: #E8E8E8;
            padding: 8px;
            border: 2px solid #363A3D;
            border-radius: 6px;
            background-color: #33373A;
            min-width: 140px;
        }

        /* Utility states for programmatic toggling via QLabel[status="..."] */
        QLabel#StatusIndicator[status="loading"] {
            color: #9AA0A6;
            background-color: #33373A;
            border-color: #363A3D;
        }
        QLabel#StatusIndicator[status="pass"] {
            color: #001712; /* dark text for legibility on green */
            background-color: #00BFA5;
            border-color: #009E87;
        }
        QLabel#StatusIndicator[status="fail"] {
            color: #FFFFFF;
            background-color: #FF3B30;
            border-color: #CC3328;
        }

        /* --- 4. QPushButton (Interactive Elements) --- */
        QPushButton {
            background-color: #272A2C;
            border: 1px solid #35393C;
            color: #E8E8E8;
            padding: 10px 14px;
            border-radius: 6px;
            min-width: 80px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #2F4C4F; /* subtle lift */
            border-color: #00BFA5;
        }
        QPushButton:pressed {
            background-color: #00BFA5;
            border-color: #009E87;
            color: #001712;
        }

        /* Primary report action (set objectName: reportButton) */
        QPushButton#reportButton {
            background-color: #00BFA5;
            color: #001712;
            font-weight: 700;
        }
        QPushButton#reportButton:hover { background-color: #00A28F; }

        /* --- 5. Input Fields --- */
        QLineEdit, QComboBox {
            background-color: #1B1D1F;
            border: 1px solid #34373A;
            padding: 6px;
            border-radius: 6px;
            color: #E8E8E8;
        }
        QLineEdit:focus, QComboBox:focus {
            border: 1px solid #00BFA5;
            outline: none;
        }

        /* --- 6. Slider (Controls) --- */
        QSlider::groove:horizontal { height: 6px; background: #2A2C2E; border-radius: 3px; }
        QSlider::handle:horizontal {
            background: #00BFA5; width: 14px; margin-top: -4px; margin-bottom: -4px; border-radius: 7px; border: 1px solid #0B3A2E;
        }

        /* Title bar */
        QWidget#TitleBar {
          background-color: #151718;
          border-bottom: 1px solid #34373A;
        }
        QLabel#WindowTitle {
          color: #E8E8E8;
          font-family: Avenir;
          font-weight: 600;
          font-size: 11pt;
          padding-left: 6px;
        }
        QPushButton#TitleButton {
          background: transparent;
          color: #9AA0A6;
          border: none;
          padding: 4px;
          border-radius: 4px;
        }
        QPushButton#TitleButton:hover { background: rgba(255,255,255,0.04); color: #FFFFFF; }
        QPushButton#TitleButton:pressed { background: rgba(255,255,255,0.06); }

        /* Menu bar */
        QMenuBar#MenuBar {
            background-color: #151718;
            color: #E8E8E8;
        }
        QMenuBar::item {
            spacing: 6px;
            padding: 4px 10px;
            background: transparent;
            color: #E8E8E8;
        }
        QMenuBar::item:selected {
            background-color: #272A2C;
            color: #FFFFFF;
        }
        QMenu {
            background-color: #1B1D1F;
            color: #E8E8E8;
        }
        QMenu::item:selected {
            background-color: #2F4C4F;
            color: #FFFFFF;
        }

        /* --- 7. Video Display --- */
        QLabel { background-color: #121213; }

    """)
    # Instantiate the Main Window
    main_window = MainWindow() 
    
    # Show the Window
    main_window.show()
    
    # Start the Event Loop
    sys.exit(app.exec_())