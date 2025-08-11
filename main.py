#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video Encryption Application
---------------------------
A professional desktop application for video encryption/decryption
using custom neural networks with real-time performance metrics visualization.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from controller.main_controller import MainController


def main():
    """Main entry point of the application"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # For a consistent look across platforms

    # Set application info
    app.setApplicationName("Neural Video Encryption")
    app.setOrganizationName("RNA Project")
    app.setApplicationVersion("1.0")

    # Initialize and show the main controller
    controller = MainController()
    controller.show_main_view()

    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
