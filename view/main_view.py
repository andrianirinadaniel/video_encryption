#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main View module
----------------
Implements the main application window and UI components.
"""

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QFileDialog,
    QAction,
    QMenuBar,
    QStatusBar,
    QSplitter,
    QFrame,
    QGridLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QImage
import cv2
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class VideoDisplayWidget(QWidget):
    """Widget for displaying video frames."""

    def __init__(self, title="Video", parent=None):
        """Initialize the video display widget."""
        super().__init__(parent)
        self.title = title
        self.current_frame = None
        self.frames = []
        self.current_frame_idx = 0
        self.is_playing = False
        self.fps = 30
        self.timer = QTimer(self)

        # Set up the layout
        layout = QVBoxLayout()

        # Add title label
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.title_label)

        # Add display label for the video frame
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(320, 240)
        self.display_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.display_label)

        # Add playback controls
        controls_layout = QHBoxLayout()

        # Play button
        self.play_btn = QPushButton("▶")
        self.play_btn.setToolTip("Play/Pause")
        self.play_btn.setFixedSize(30, 30)
        controls_layout.addWidget(self.play_btn)

        # Stop button
        self.stop_btn = QPushButton("■")
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.setFixedSize(30, 30)
        controls_layout.addWidget(self.stop_btn)

        # Frame counter
        self.frame_counter = QLabel("0/0")
        controls_layout.addWidget(self.frame_counter)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        self.setLayout(layout)

    def display_frame(self, frame):
        """
        Display a video frame.

        Args:
            frame: OpenCV image frame (numpy array)
        """
        if frame is None:
            return

        self.current_frame = frame

        # Convert the frame to RGB (from BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a QImage from the frame
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Create a pixmap and display it
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(
            self.display_label.width(),
            self.display_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.display_label.setPixmap(pixmap)

    def set_frames(self, frames, fps=30):
        """
        Set frames for playback.

        Args:
            frames: List of video frames
            fps: Frames per second
        """
        self.frames = frames
        self.fps = fps
        self.current_frame_idx = 0

        # Update frame counter
        self.frame_counter.setText(f"0/{len(self.frames)}")

        # Set timer interval based on fps
        self.timer.setInterval(int(1000 / fps))
        self.timer.timeout.connect(self._next_frame)

        # Connect buttons
        self.play_btn.clicked.connect(self._toggle_play)
        self.stop_btn.clicked.connect(self._stop)

        # Display the first frame if available
        if len(frames) > 0:
            self.display_frame(frames[0])

    def _toggle_play(self):
        """Toggle between play and pause states."""
        if len(self.frames) == 0:
            return

        if self.is_playing:
            # Pause
            self.is_playing = False
            self.timer.stop()
            self.play_btn.setText("▶")
        else:
            # Play
            self.is_playing = True
            self.timer.start()
            self.play_btn.setText("⏸")

    def _stop(self):
        """Stop playback and reset to the beginning."""
        self.is_playing = False
        self.timer.stop()
        self.current_frame_idx = 0
        self.play_btn.setText("▶")

        # Display first frame if available
        if len(self.frames) > 0:
            self.display_frame(self.frames[0])
            self.frame_counter.setText(f"0/{len(self.frames)}")

    def _next_frame(self):
        """Display the next frame in the sequence."""
        if len(self.frames) == 0:
            self._stop()
            return

        # Advance to next frame
        self.current_frame_idx += 1

        # Check if reached the end of the video
        if self.current_frame_idx >= len(self.frames):
            self.current_frame_idx = 0  # Loop back to the beginning

        # Update display
        self.display_frame(self.frames[self.current_frame_idx])
        self.frame_counter.setText(f"{self.current_frame_idx}/{len(self.frames)}")

    def clear(self):
        """Clear the displayed frame."""
        self.display_label.clear()
        self.display_label.setStyleSheet("background-color: black;")
        self.current_frame = None
        self.frames = []
        self._stop()


class MetricsDisplayWidget(QWidget):
    """Widget for displaying encryption/decryption metrics."""

    def __init__(self, parent=None):
        """Initialize the metrics display widget."""
        super().__init__(parent)

        # Set up the layout
        layout = QVBoxLayout()

        # Add title label
        title_label = QLabel("Quality Metrics")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)

        # Create grid for metrics
        metrics_grid = QGridLayout()

        # Add metrics labels
        metrics_labels = ["PSNR:", "SSIM:", "Entropy:", "Correlation:", "VIF:"]
        self.metrics_values = []

        for i, label_text in enumerate(metrics_labels):
            label = QLabel(label_text)
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("N/A")
            value_label.setStyleSheet("font-family: monospace;")
            metrics_grid.addWidget(label, i, 0)
            metrics_grid.addWidget(value_label, i, 1)
            self.metrics_values.append(value_label)

        layout.addLayout(metrics_grid)
        layout.addStretch()
        self.setLayout(layout)

    def update_metrics(self, metrics_dict):
        """
        Update the displayed metrics.

        Args:
            metrics_dict: Dictionary containing metrics values
        """
        if metrics_dict is None:
            return

        # Update the metrics values
        self.metrics_values[0].setText(f"{metrics_dict.get('psnr', 0.0):.2f} dB")
        self.metrics_values[1].setText(f"{metrics_dict.get('ssim', 0.0):.4f}")
        self.metrics_values[2].setText(f"{metrics_dict.get('entropy', 0.0):.2f}")
        self.metrics_values[3].setText(f"{metrics_dict.get('correlation', 0.0):.4f}")
        self.metrics_values[4].setText(f"{metrics_dict.get('vif', 0.0):.4f}")


class TrainingGraphWidget(QWidget):
    """Widget for displaying training graphs."""

    def __init__(self, parent=None):
        """Initialize the training graph widget."""
        super().__init__(parent)

        # Set up the layout
        layout = QVBoxLayout()

        # Add title label
        title_label = QLabel("Training Progress")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)

        # Create horizontal layout for the two graphs
        graphs_layout = QHBoxLayout()

        # Create the accuracy figure
        self.accuracy_fig = Figure(figsize=(4, 3), dpi=100)
        self.accuracy_canvas = FigureCanvas(self.accuracy_fig)
        self.accuracy_ax = self.accuracy_fig.add_subplot(111)
        self.accuracy_ax.set_title("Accuracy")
        self.accuracy_ax.set_xlabel("Epoch")
        self.accuracy_ax.set_ylabel("Accuracy")
        self.accuracy_ax.grid(True)

        # Create the loss figure
        self.loss_fig = Figure(figsize=(4, 3), dpi=100)
        self.loss_canvas = FigureCanvas(self.loss_fig)
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_ax.set_title("Loss")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.grid(True)

        # Add the figures to the layout
        graphs_layout.addWidget(self.accuracy_canvas)
        graphs_layout.addWidget(self.loss_canvas)
        layout.addLayout(graphs_layout)

        self.setLayout(layout)

        # Initialize data
        self.accuracy_data = []
        self.loss_data = []
        self.epochs = []

    def update_graphs(self, training_data):
        """
        Update the training graphs.

        Args:
            training_data: Dictionary with epoch, accuracy, and loss data
        """
        if training_data is None:
            return

        # Extract data
        epoch = training_data.get("epoch", 0)
        accuracy = training_data.get("accuracy", 0)
        loss = training_data.get("loss", 0)

        # Update data lists
        self.epochs.append(epoch)
        self.accuracy_data.append(accuracy)
        self.loss_data.append(loss)

        # Update accuracy plot
        self.accuracy_ax.clear()
        self.accuracy_ax.set_title("Accuracy")
        self.accuracy_ax.set_xlabel("Epoch")
        self.accuracy_ax.set_ylabel("Accuracy")
        self.accuracy_ax.plot(self.epochs, self.accuracy_data, "b-")
        self.accuracy_ax.grid(True)
        self.accuracy_canvas.draw()

        # Update loss plot
        self.loss_ax.clear()
        self.loss_ax.set_title("Loss")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.plot(self.epochs, self.loss_data, "r-")
        self.loss_ax.grid(True)
        self.loss_canvas.draw()

    def clear_graphs(self):
        """Clear all graphs."""
        self.epochs = []
        self.accuracy_data = []
        self.loss_data = []

        # Clear accuracy plot
        self.accuracy_ax.clear()
        self.accuracy_ax.set_title("Accuracy")
        self.accuracy_ax.set_xlabel("Epoch")
        self.accuracy_ax.set_ylabel("Accuracy")
        self.accuracy_ax.grid(True)
        self.accuracy_canvas.draw()

        # Clear loss plot
        self.loss_ax.clear()
        self.loss_ax.set_title("Loss")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.grid(True)
        self.loss_canvas.draw()


class MainView(QMainWindow):
    """Main application window."""

    # Define signals
    load_video_signal = pyqtSignal(str)  # Video path
    encrypt_signal = pyqtSignal()
    decrypt_signal = pyqtSignal()
    new_training_signal = pyqtSignal()

    # Add the real_training_signal as a class attribute
    real_training_signal = pyqtSignal()

    def __init__(self, controller):
        """Initialize the main window with a reference to the controller."""
        super().__init__()
        self.controller = controller

        # Set window properties
        self.setWindowTitle("Neural Video Encryption")
        self.setMinimumSize(1024, 768)

        # Create the central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create the main layout
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create menu bar
        self._create_menu_bar()

        # Create the video display area
        self._create_video_display_area()

        # Create control buttons
        self._create_control_buttons()

        # Create the training graphs area
        self._create_training_graphs_area()

        # Create status bar with progress bar
        self._create_status_bar()

        # Apply dark theme styling
        self._apply_styling()

        # Initially disable the real training button
        self.real_training_btn.setEnabled(False)

    def _create_menu_bar(self):
        """Create and populate the menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("&Open Video...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_video)
        file_menu.addAction(open_action)

        save_action = QAction("&Save Result...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save_result)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Model menu
        model_menu = menu_bar.addMenu("&Model")

        train_action = QAction("&Train New Model...", self)
        train_action.triggered.connect(self._on_train_new_model)
        model_menu.addAction(train_action)

        load_model_action = QAction("&Load Model...", self)
        load_model_action.triggered.connect(self._on_load_model)
        model_menu.addAction(load_model_action)

        # View menu
        view_menu = menu_bar.addMenu("&View")

        toggle_metrics_action = QAction("Show &Metrics", self)
        toggle_metrics_action.setCheckable(True)
        toggle_metrics_action.setChecked(True)
        toggle_metrics_action.triggered.connect(self._on_toggle_metrics)
        view_menu.addAction(toggle_metrics_action)

        toggle_graphs_action = QAction("Show &Graphs", self)
        toggle_graphs_action.setCheckable(True)
        toggle_graphs_action.setChecked(True)
        toggle_graphs_action.triggered.connect(self._on_toggle_graphs)
        view_menu.addAction(toggle_graphs_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

        # Initialize the real_training_btn here - but it will be added to layout in _create_control_buttons
        self.real_training_btn = QPushButton("Train Decryption (Real Data)")
        self.real_training_btn.setToolTip("Train the model using real data")
        self.real_training_btn.clicked.connect(self._on_real_train)

    def _create_video_display_area(self):
        """Create the video display area with original, encrypted, and decrypted videos."""
        # Create a horizontal layout for video panels
        video_layout = QGridLayout()

        # Create video display widgets
        self.original_video = VideoDisplayWidget("Original Video")
        self.encrypted_video = VideoDisplayWidget("Encrypted Video")
        self.decrypted_video = VideoDisplayWidget("Decrypted Video")
        self.metrics_display = MetricsDisplayWidget()

        # Add video displays to the layout
        video_layout.addWidget(self.original_video, 0, 0)
        video_layout.addWidget(self.encrypted_video, 0, 1)
        video_layout.addWidget(self.decrypted_video, 1, 0)
        video_layout.addWidget(self.metrics_display, 1, 1)

        # Add the video layout to the main layout
        self.main_layout.addLayout(video_layout)

    # Add a method to update video displays
    def update_video_display(self, video_type, frames, fps=30):
        """Update a video display with new frames."""
        if video_type == "original":
            self.original_video.set_frames(frames, fps)
            # Enable encrypt button and real training now that video is loaded
            self.encrypt_btn.setEnabled(True)
            self.real_training_btn.setEnabled(True)
            self.status_label.setText(f"Original video loaded: {len(frames)} frames")
        elif video_type == "encrypted":
            self.encrypted_video.set_frames(frames, fps)
            # Enable decrypt button now that video is encrypted
            self.decrypt_btn.setEnabled(True)
            self.status_label.setText(f"Video encrypted: {len(frames)} frames")
        elif video_type == "decrypted":
            self.decrypted_video.set_frames(frames, fps)
            self.status_label.setText(f"Video decrypted: {len(frames)} frames")

    def _on_real_train(self):
        """Handle the real training button click."""
        self.real_training_signal.emit()
        self.status_label.setText("Training model with real data...")

    def _create_control_buttons(self):
        """Create the control buttons for the application."""
        # Create a horizontal layout for buttons
        buttons_layout = QHBoxLayout()

        # Create buttons
        self.load_btn = QPushButton("Load Video")
        self.encrypt_btn = QPushButton("Encrypt")
        self.decrypt_btn = QPushButton("Decrypt")
        self.new_training_btn = QPushButton("New Training")

        # Add tooltips
        self.load_btn.setToolTip("Load a video file for encryption/decryption")
        self.encrypt_btn.setToolTip("Encrypt the loaded video using the neural network")
        self.decrypt_btn.setToolTip(
            "Decrypt the encrypted video using the neural network"
        )
        self.new_training_btn.setToolTip(
            "Start a new training session for the neural network"
        )

        # Disable buttons initially
        self.encrypt_btn.setEnabled(False)
        self.decrypt_btn.setEnabled(False)

        # Connect button signals
        self.load_btn.clicked.connect(self._on_open_video)
        self.encrypt_btn.clicked.connect(self._on_encrypt)
        self.decrypt_btn.clicked.connect(self._on_decrypt)
        self.new_training_btn.clicked.connect(self._on_train_new_model)

        # Add buttons to the layout
        buttons_layout.addWidget(self.load_btn)
        buttons_layout.addWidget(self.encrypt_btn)
        buttons_layout.addWidget(self.decrypt_btn)
        buttons_layout.addWidget(self.new_training_btn)
        # Add the real training button here
        buttons_layout.addWidget(self.real_training_btn)

        # Add some spacing
        buttons_layout.addStretch()

        # Add the buttons layout to the main layout
        self.main_layout.addLayout(buttons_layout)

    def _create_training_graphs_area(self):
        """Create the area for displaying training graphs."""
        # Create the graphs widget
        self.training_graphs = TrainingGraphWidget()

        # Add the graphs widget to the main layout
        self.main_layout.addWidget(self.training_graphs)

    def _create_status_bar(self):
        """Create the status bar with progress bar."""
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% Complete")
        self.progress_bar.setFixedWidth(200)

        # Add progress bar to status bar
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Add a status label
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

    def _apply_styling(self):
        """Apply dark theme styling to the application."""
        # Set stylesheet for dark theme
        self.setStyleSheet(
            """
            QWidget {
                background-color: #2D2D30;
                color: #E1E1E1;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            QMainWindow {
                background-color: #2D2D30;
            }
            
            QMenuBar {
                background-color: #1E1E1E;
                color: #E1E1E1;
            }
            
            QMenuBar::item:selected {
                background-color: #3E3E40;
            }
            
            QMenu {
                background-color: #1E1E1E;
                color: #E1E1E1;
                border: 1px solid #3E3E40;
            }
            
            QMenu::item:selected {
                background-color: #3E3E40;
            }
            
            QPushButton {
                background-color: #0E639C;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
            }
            
            QPushButton:hover {
                background-color: #1177BB;
            }
            
            QPushButton:pressed {
                background-color: #0D5A8C;
            }
            
            QPushButton:disabled {
                background-color: #3E3E40;
                color: #9D9D9D;
            }
            
            QProgressBar {
                border: 1px solid #3E3E40;
                border-radius: 3px;
                background-color: #1E1E1E;
                text-align: center;
            }
            
            QProgressBar::chunk {
                background-color: #0E639C;
                width: 10px;
                margin: 0px;
            }
            
            QStatusBar {
                background-color: #1E1E1E;
                color: #E1E1E1;
            }
            
            QLabel {
                color: #E1E1E1;
            }
        """
        )

    def _on_open_video(self):
        """Handle the action of opening a video file."""
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )

        if video_path:
            # Show progress immediately
            self.progress_bar.setValue(10)
            self.status_label.setText(f"Loading video: {video_path}...")
            self.set_ui_busy(True)

            # Emit signal to load the video
            self.load_video_signal.emit(video_path)

    def video_loaded_callback(self, result):
        """
        Handle completion of video loading.

        Args:
            result: Dictionary with loading results including frames
        """
        success = result.get("success", False)

        if success:
            frames = result.get("frames", [])
            fps = result.get("fps", 30)

            # Update the original video display
            self.update_video_display("original", frames, fps)

            self.status_label.setText(
                f"Video loaded successfully: {len(frames)} frames"
            )
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error loading video: {error}")

        self.progress_bar.setValue(100)
        self.set_ui_busy(False)

    def _on_encrypt(self):
        """Handle the encrypt button click."""
        self.set_ui_busy(True)
        self.progress_bar.setValue(10)
        self.encrypt_signal.emit()
        self.status_label.setText("Encrypting video...")

    def _on_decrypt(self):
        """Handle the decrypt button click."""
        self.set_ui_busy(True)
        self.progress_bar.setValue(10)
        self.decrypt_signal.emit()
        self.status_label.setText("Decrypting video...")

    def _on_train_new_model(self):
        """Handle the new training button click."""
        self.new_training_signal.emit()
        self.status_label.setText("Training new model...")

    def _on_save_result(self):
        """Handle saving the result video."""
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getSaveFileName(
            self,
            "Save Video",
            "",
            "MP4 Files (*.mp4);;AVI Files (*.avi);;All Files (*.*)",
        )

        if video_path:
            # TODO: Implement saving the video
            self.status_label.setText(f"Saving to: {video_path}")

    def _on_load_model(self):
        """Handle loading a pre-trained model."""
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(
            self, "Load Model", "", "Model Files (*.h5);;All Files (*.*)"
        )

        if model_path:
            # TODO: Implement model loading
            self.status_label.setText(f"Loaded model: {model_path}")

    def _on_toggle_metrics(self, checked):
        """Handle toggling the metrics display."""
        self.metrics_display.setVisible(checked)

    def _on_toggle_graphs(self, checked):
        """Handle toggling the training graphs display."""
        self.training_graphs.setVisible(checked)

    def _on_about(self):
        """Show the about dialog."""
        # TODO: Implement about dialog
        self.status_label.setText("Neural Video Encryption v1.0")

    def update_progress(self, value):
        """
        Update the progress bar.

        Args:
            value: Progress value (0-100)
        """
        self.progress_bar.setValue(value)

    def processing_finished(self, result):
        """
        Handle the completion of processing.

        Args:
            result: Dictionary with processing results
        """
        operation_type = result.get("type", "")
        success = result.get("success", False)

        if success:
            if operation_type == "encryption":
                self.status_label.setText("Encryption completed successfully")
                self.decrypt_btn.setEnabled(True)
            elif operation_type == "decryption":
                self.status_label.setText("Decryption completed successfully")
        else:
            error = result.get("error", "Unknown error")
            self.status_label.setText(f"Error: {error}")

        # Reset UI state
        self.set_ui_busy(False)

    def update_training_graphs(self, training_data):
        """
        Update the training graphs.

        Args:
            training_data: Dictionary with training progress data
        """
        self.training_graphs.update_graphs(training_data)

        # Update status
        epoch = training_data.get("epoch", 0)
        total = training_data.get("total_epochs", 0)
        self.status_label.setText(f"Training: Epoch {epoch}/{total}")

    def set_ui_busy(self, busy):
        """
        Set the UI to busy/processing state.

        Args:
            busy: True if processing, False otherwise
        """
        self.load_btn.setEnabled(not busy)

        # Only restore previous button states when not busy
        if not busy:
            # Check if video is loaded (if original video has frames)
            video_loaded = (
                hasattr(self.original_video, "frames")
                and len(self.original_video.frames) > 0
            )

            # Check if video is encrypted (if encrypted video has frames)
            video_encrypted = (
                hasattr(self.encrypted_video, "frames")
                and len(self.encrypted_video.frames) > 0
            )

            # Set button states based on application state
            self.encrypt_btn.setEnabled(video_loaded)
            self.decrypt_btn.setEnabled(video_encrypted)
            self.new_training_btn.setEnabled(True)
            self.real_training_btn.setEnabled(video_loaded)
        else:
            # Disable all processing buttons when busy
            self.encrypt_btn.setEnabled(False)
            self.decrypt_btn.setEnabled(False)
            self.new_training_btn.setEnabled(False)
            self.real_training_btn.setEnabled(False)

        if not busy:
            self.progress_bar.setValue(0)
