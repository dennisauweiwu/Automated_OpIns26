# main.py

import sys
from PyQt5.QtWidgets import QApplication
from aoi_gui.MainWindow import MainWindow  

if __name__ == '__main__':
    # Initialize Application Enivronment
    app = QApplication(sys.argv)
    
    # UI font and theme: load from centralized stylesheet `ui/style.qss` for a cleaner `main.py` file
    from pathlib import Path
    qss_path = Path(__file__).resolve().parent / 'ui' / 'style.qss'
    try:
        with open(qss_path, 'r', encoding='utf-8') as fh:
            app.setStyleSheet(fh.read())
    except Exception:
        pass
    # Instantiate the Main Window
    main_window = MainWindow() 
    
    # Show the Window
    main_window.show()
    
    # Start the Event Loop
    sys.exit(app.exec_())