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
            background-color: #21252B; /* Deep Slate/Charcoal */
            color: #E0E0E0;            /* Near White Text */
            font-family: Avenir;
            font-size: 10pt;
        }

        /* --- 2. QGroupBox (Container Styles) --- */
        QGroupBox {
            border: 1px solid #3A3F47; /* Subtle border */
            border-radius: 4px;
            margin-top: 25px;
            padding-top: 15px;
            font-weight: normal;
            background-color: #21252B; 
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: #A0A0A0; /* Muted title text */
            font-size: 9pt;
        }

        /* --- 3. QLabel#StatusIndicator --- */
        QLabel#StatusIndicator {
            font-size: 18pt; 
            font-weight: bold;
            color: #76FF03; /* Bright Lime Green Accent */
            padding: 5px;
            border: 2px solid #76FF03; 
            border-radius: 4px;
            background-color: rgba(118, 255, 3, 0.1); 
        }

        /* --- 4. QPushButton (Interactive Elements) --- */
        QPushButton {
            background-color: #3A3F47; /* Flat, slightly lighter than background */
            border: 1px solid #4D515B; 
            color: #E0E0E0;
            padding: 10px 15px; 
            border-radius: 4px;
            min-width: 80px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #4D515B; 
            border-color: #00AEEF; /* Deep Blue Accent on hover */
        }
        QPushButton:pressed {
            background-color: #00AEEF; /* Accent Color on Press */
            border-color: #00AEEF;
            color: #FFFFFF;
        }
        
        /* --- 5. Input Fields --- */
        QLineEdit, QComboBox {
            background-color: #2D323A; 
            border: 1px solid #4D515B;
            padding: 5px;
            border-radius: 4px;
            color: #E0E0E0;
        }
        QLineEdit:focus, QComboBox:focus {
            border: 1px solid #00AEEF; 
        }
    """)
    # Instantiate the Main Window
    main_window = MainWindow() 
    
    # Show the Window
    main_window.show()
    
    # Start the Event Loop
    sys.exit(app.exec_())