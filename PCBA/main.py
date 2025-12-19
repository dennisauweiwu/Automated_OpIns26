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
        /* Main Background: deep desaturated Navy/Teal (#001B26) */
        QWidget {
            background-color: #001B26;
            color: #E0F7FA;            /* Primary text */
            font-family: Avenir;
            font-size: 10pt;
        }

        /* --- 2. QGroupBox (Container Styles) --- */
        /* Secondary Surface: panel blue (#002D3A) */
        QGroupBox {
            border: 1px solid #0B2530; /* thin, slightly lighter than background */
            border-radius: 6px; /* soft rounded corners */
            margin-top: 25px;
            padding: 12px; /* consistent internal padding */
            padding-top: 18px; /* preserve title spacing */
            font-weight: normal;
            background-color: #002D3A;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            color: #B0BEC5; /* Control icons / muted text */
            font-size: 9pt;
        }

        /* Ensure consistent rounded corners on common controls */
        QPushButton, QLineEdit, QComboBox {
            border-radius: 6px;
        }

        /* Small padding for labels to keep spacing uniform between labels and inputs */
        QLabel {
            padding: 2px 6px 2px 6px;
        }

        /* --- 3. QLabel#StatusIndicator --- */
        QLabel#StatusIndicator {
            font-size: 18pt;
            font-weight: bold;
            color: #E0F7FA;
            padding: 8px 12px;
            border: 1px solid #0B2530; /* thin border */
            border-radius: 6px;
            background-color: #002D3A;
            min-width: 140px;
        }

        /* Utility states for programmatic toggling via QLabel[status="..."] */
        QLabel#StatusIndicator[status="loading"] {
            color: #B0BEC5;
            background-color: #002D3A;
            border-color: #0B2530;
        }
        /* Action Green for PASS */
        QLabel#StatusIndicator[status="pass"] {
            color: #001712;
            background-color: #6DA06F; /* Action Green */
            border-color: #5b8a60;
        }
        QLabel#StatusIndicator[status="fail"] {
            color: #FFFFFF;
            background-color: #C94C40; /* desaturated red to match UI */
            border-color: #A63F36;
        }

        /* --- 4. QPushButton (Interactive Elements) --- */
        /* Default buttons use Secondary Surface, icons use cool grey */
        QPushButton {
            background-color: #002D3A;
            border: 1px solid #1D4B5E; /* Secondary Accent */
            color: #E0F7FA;
            padding: 12px 14px; /* increase vertical padding to avoid clipping descenders */
            border-radius: 6px;
            min-width: 80px;
            min-height: 44px; /* enforce a comfortable line-height */
            font-weight: 500;
        }

        /* ensure labels have enough vertical padding for descenders */
        QLabel {
            padding: 4px 6px;
        }
        QPushButton:hover {
            background-color: #31A3B8; /* Primary Accent */
            border-color: #31A3B8;
            color: #001712;
        }
        QPushButton:pressed {
            background-color: #31A3B8;
            border-color: #1D4B5E;
            color: #001712;
        }

        /* Primary report action (set objectName: reportButton) -> use Primary Accent */
        QPushButton#reportButton {
            background-color: #31A3B8;
            color: #001712;
            font-weight: 700;
        }
        QPushButton#reportButton:hover { background-color: #2A95A6; }

        /* --- 5. Input Fields --- */
        QLineEdit, QComboBox {
            background-color: #002D3A;
            border: 1px solid #1D4B5E;
            padding: 6px;
            border-radius: 6px;
            color: #E0F7FA;
        }
        QLineEdit:focus, QComboBox:focus {
            border: 1px solid #31A3B8; /* Primary Accent focus */
            outline: none;
        }

        /* --- 6. Slider (Controls) --- */
        QSlider::groove:horizontal { height: 6px; background: #1D4B5E; border-radius: 3px; }
        QSlider::handle:horizontal {
            background: #31A3B8; width: 14px; margin-top: -4px; margin-bottom: -4px; border-radius: 7px; border: 1px solid #0B3A2E;
        }

        /* Title bar */
        QWidget#TitleBar {
          background-color: #001B26; /* main background */
          border-bottom: 1px solid #002D3A; /* subtle thin border */
          border-top-left-radius: 6px;
          border-top-right-radius: 6px;
        }
        QLabel#WindowTitle {
          color: #E0F7FA;
          font-family: Avenir;
          font-weight: 600;
          font-size: 11pt;
          padding-left: 6px;
        }
        QPushButton#TitleButton {
          background: transparent;
          color: #B0BEC5; /* control icons */
          border: none;
          padding: 6px; /* larger padding for comfortable hit area */
          border-radius: 6px;
          min-width: 28px;
          min-height: 28px;
          font-size: 11pt; /* ensure glyphs render clearly */
        }
        QPushButton#TitleButton:hover { background: rgba(255,255,255,0.045); color: #E0F7FA; }
        QPushButton#TitleButton:pressed { background: rgba(255,255,255,0.07); }

        /* Menu bar */
        QMenuBar#MenuBar {
            background-color: #001B26;
            color: #E0F7FA;
        }
        QMenuBar::item:selected { background-color: #002D3A; color: #E0F7FA; }
        QMenu { background-color: #002D3A; color: #E0F7FA; }
        QMenu::item:selected { background-color: #1D4B5E; color: #E0F7FA; }

        /* --- 7. Video Display --- */
        QLabel { background-color: #001B26; }

    """)
    # Instantiate the Main Window
    main_window = MainWindow() 
    
    # Show the Window
    main_window.show()
    
    # Start the Event Loop
    sys.exit(app.exec_())