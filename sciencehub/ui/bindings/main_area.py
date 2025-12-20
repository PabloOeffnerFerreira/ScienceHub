from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QListWidget, QListWidgetItem, QTabWidget, QGroupBox,
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QSplitter, QScrollArea,
    QTableWidget, QTableWidgetItem, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor
import math
import re
import datetime
import random

class MainArea(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Main tab widget
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Build all tabs
        self._build_welcome_tab()
        self._build_calculator_tab()
        self._build_logs_tab()
        self._build_placeholder_tab()

        # Timer for updating stats
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start(5000)  # Update every 5 seconds

        # Calculator state
        self.calc_expression = ""
        self.calc_history = []
        self.calc_memory = 0.0
        self.calc_degrees = True  # True for degrees, False for radians

        # Logs state
        self.log_entries = []
        self.log_filters = {"INFO": True, "WARNING": True, "ERROR": True, "DEBUG": True}

        # Placeholder features
        self.placeholder_features = []

    def _clear_layout(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def set_tool(self, tool_widget: QWidget):
        # Add tool as a new tab
        tool_name = getattr(tool_widget, 'tool_name', 'Tool') if hasattr(tool_widget, 'tool_name') else 'Tool'
        self.tabs.addTab(tool_widget, tool_name)
        self.tabs.setCurrentWidget(tool_widget)

    def show_home(self):
        self.tabs.setCurrentIndex(0)  # Welcome tab

    def _build_welcome_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Welcome header
        welcome_label = QLabel("Welcome to ScienceHub")
        welcome_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_label)

        # Statistics section
        stats_group = QGroupBox("System Statistics")
        stats_layout = QGridLayout(stats_group)

        self.stats_labels = {}
        stats = [
            ("Active Tools", "0"),
            ("Total Calculations", "0"),
            ("Log Entries", "0"),
            ("Memory Usage", "0 MB"),
            ("CPU Usage", "0%"),
            ("Uptime", "00:00:00"),
            ("Tools Used Today", "0"),
            ("Average Session Time", "0 min"),
            ("Most Used Tool", "None"),
            ("Recent Activity", "None")
        ]

        for i, (label, value) in enumerate(stats):
            row = i // 2
            col = (i % 2) * 2
            stats_layout.addWidget(QLabel(f"{label}:"), row, col)
            value_label = QLabel(value)
            value_label.setFont(QFont("Courier", 10))
            stats_layout.addWidget(value_label, row, col + 1)
            self.stats_labels[label] = value_label

        layout.addWidget(stats_group)

        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QHBoxLayout(actions_group)

        actions = ["New Calculation", "View Logs", "Clear History", "Export Stats"]
        for action in actions:
            btn = QPushButton(action)
            btn.clicked.connect(lambda checked, a=action: self._quick_action(a))
            actions_layout.addWidget(btn)

        layout.addWidget(actions_group)

        # Recent tools
        recent_group = QGroupBox("Recent Tools")
        recent_layout = QVBoxLayout(recent_group)
        self.recent_tools_list = QListWidget()
        self.recent_tools_list.addItem("No recent tools")
        recent_layout.addWidget(self.recent_tools_list)
        layout.addWidget(recent_group)

        self.tabs.addTab(widget, "Welcome")

    def _build_calculator_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Calculator display
        display_group = QGroupBox("Expression")
        display_layout = QVBoxLayout(display_group)
        self.calc_display = QLineEdit()
        self.calc_display.setFont(QFont("Courier", 14))
        self.calc_display.setReadOnly(True)
        self.calc_display.setAlignment(Qt.AlignmentFlag.AlignRight)
        display_layout.addWidget(self.calc_display)

        self.calc_result_display = QLineEdit()
        self.calc_result_display.setFont(QFont("Courier", 16, QFont.Weight.Bold))
        self.calc_result_display.setReadOnly(True)
        self.calc_result_display.setAlignment(Qt.AlignmentFlag.AlignRight)
        display_layout.addWidget(self.calc_result_display)
        layout.addWidget(display_group)

        # Calculator buttons
        buttons_widget = QWidget()
        buttons_layout = QGridLayout(buttons_widget)

        # Define button layout
        button_labels = [
            ['sin', 'cos', 'tan', 'log', 'ln', 'sqrt'],
            ['asin', 'acos', 'atan', 'exp', 'pow', 'abs'],
            ['7', '8', '9', '/', '(', 'deg'],
            ['4', '5', '6', '*', ')', 'rad'],
            ['1', '2', '3', '-', 'pi', 'e'],
            ['0', '.', '=', '+', 'C', 'CE'],
            ['M+', 'M-', 'MR', 'MC', 'Ans', 'hist']
        ]

        self.calc_buttons = {}
        for row, button_row in enumerate(button_labels):
            for col, label in enumerate(button_row):
                btn = QPushButton(label)
                btn.setFont(QFont("Arial", 10))
                btn.clicked.connect(lambda checked, l=label: self._calc_button_pressed(l))
                buttons_layout.addWidget(btn, row, col)
                self.calc_buttons[label] = btn

        layout.addWidget(buttons_widget)

        # Advanced features
        advanced_group = QGroupBox("Advanced Features")
        advanced_layout = QHBoxLayout(advanced_group)

        # Variable input
        var_layout = QVBoxLayout()
        var_layout.addWidget(QLabel("Variables:"))
        self.calc_vars = {}
        for var in ['x', 'y', 'z']:
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel(f"{var} ="))
            edit = QLineEdit("0")
            hbox.addWidget(edit)
            var_layout.addLayout(hbox)
            self.calc_vars[var] = edit

        advanced_layout.addLayout(var_layout)

        # Function plot (placeholder)
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(QLabel("Function Plot:"))
        self.plot_input = QLineEdit("x**2")
        plot_layout.addWidget(self.plot_input)
        plot_btn = QPushButton("Plot")
        plot_btn.clicked.connect(self._plot_function)
        plot_layout.addWidget(plot_btn)
        advanced_layout.addLayout(plot_layout)

        # Unit conversion
        unit_layout = QVBoxLayout()
        unit_layout.addWidget(QLabel("Unit Conversion:"))
        self.unit_from = QComboBox()
        self.unit_from.addItems(["meters", "feet", "inches", "cm", "mm", "km"])
        self.unit_to = QComboBox()
        self.unit_to.addItems(["meters", "feet", "inches", "cm", "mm", "km"])
        self.unit_value = QLineEdit("1")
        convert_btn = QPushButton("Convert")
        convert_btn.clicked.connect(self._convert_units)
        unit_layout.addWidget(self.unit_from)
        unit_layout.addWidget(QLabel("to"))
        unit_layout.addWidget(self.unit_to)
        unit_layout.addWidget(self.unit_value)
        unit_layout.addWidget(convert_btn)
        advanced_layout.addLayout(unit_layout)

        layout.addWidget(advanced_group)

        # History
        history_group = QGroupBox("Calculation History")
        history_layout = QVBoxLayout(history_group)
        self.calc_history_list = QListWidget()
        self.calc_history_list.itemDoubleClicked.connect(self._load_history_item)
        history_layout.addWidget(self.calc_history_list)

        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.clicked.connect(self._clear_calc_history)
        history_layout.addWidget(clear_history_btn)
        layout.addWidget(history_group)

        self.tabs.addTab(widget, "Calculator")

    def _build_logs_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Log controls
        controls_layout = QHBoxLayout()
        
        # Filter checkboxes
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout(filter_group)
        self.filter_checkboxes = {}
        for level in ["INFO", "WARNING", "ERROR", "DEBUG"]:
            cb = QCheckBox(level)
            cb.setChecked(True)
            cb.stateChanged.connect(self._update_log_filters)
            filter_layout.addWidget(cb)
            self.filter_checkboxes[level] = cb
        controls_layout.addWidget(filter_group)

        # Search
        search_layout = QVBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.log_search = QLineEdit()
        self.log_search.textChanged.connect(self._filter_logs)
        search_layout.addWidget(self.log_search)
        controls_layout.addLayout(search_layout)

        # Clear logs
        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self._clear_logs)
        controls_layout.addWidget(clear_btn)

        # Export logs
        export_btn = QPushButton("Export Logs")
        export_btn.clicked.connect(self._export_logs)
        controls_layout.addWidget(export_btn)

        layout.addLayout(controls_layout)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setFont(QFont("Courier", 10))
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)

        # Log statistics
        stats_layout = QHBoxLayout()
        self.log_stats_label = QLabel("Total entries: 0 | Filtered: 0")
        stats_layout.addWidget(self.log_stats_label)
        stats_layout.addStretch()
        layout.addLayout(stats_layout)

        self.tabs.addTab(widget, "Logs")

    def _build_placeholder_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Placeholder title
        title = QLabel("Feature Placeholder")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Description
        desc = QLabel("This area is reserved for future features and expansions.\n"
                     "Possible upcoming features include:\n"
                     "• Interactive tutorials\n"
                     "• Tool marketplace\n"
                     "• Collaboration workspace\n"
                     "• Advanced analytics dashboard\n"
                     "• Custom plugin development\n"
                     "• Data visualization studio\n"
                     "• Educational content library")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        # Feature request
        request_group = QGroupBox("Feature Request")
        request_layout = QVBoxLayout(request_group)
        self.feature_request = QTextEdit()
        self.feature_request.setPlaceholderText("Describe a feature you'd like to see here...")
        request_layout.addWidget(self.feature_request)
        submit_btn = QPushButton("Submit Request")
        submit_btn.clicked.connect(self._submit_feature_request)
        request_layout.addWidget(submit_btn)
        layout.addWidget(request_group)

        # Quick access to experimental features
        exp_group = QGroupBox("Experimental Features")
        exp_layout = QVBoxLayout(exp_group)
        exp_features = ["AI Assistant", "Real-time Collaboration", "Cloud Sync", "Advanced Plotting"]
        for feature in exp_features:
            cb = QCheckBox(feature)
            cb.setEnabled(False)  # Disabled for now
            exp_layout.addWidget(cb)
        layout.addWidget(exp_group)

        # Fun easter egg
        easter_btn = QPushButton("🎉 Surprise!")
        easter_btn.clicked.connect(self._easter_egg)
        layout.addWidget(easter_btn)

        self.tabs.addTab(widget, "Placeholder")

    # Welcome tab methods
    def _update_stats(self):
        # Simulate updating statistics
        try:
            import psutil
            # Real system stats
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            self.stats_labels["Memory Usage"].setText(f"{memory.used // (1024**2)} MB")
            self.stats_labels["CPU Usage"].setText(f"{cpu}%")
        except ImportError:
            # Fallback if psutil not available
            self.stats_labels["Memory Usage"].setText("N/A")
            self.stats_labels["CPU Usage"].setText("N/A")
        
        # Update other stats
        self.stats_labels["Active Tools"].setText(str(self.tabs.count() - 4))  # Subtract built-in tabs
        self.stats_labels["Total Calculations"].setText(str(len(self.calc_history)))
        self.stats_labels["Log Entries"].setText(str(len(self.log_entries)))
        
        # Uptime (simplified)
        uptime = datetime.datetime.now() - datetime.datetime.today().replace(hour=0, minute=0, second=0)
        self.stats_labels["Uptime"].setText(str(uptime).split('.')[0])

    def _quick_action(self, action):
        if action == "New Calculation":
            self.tabs.setCurrentIndex(1)  # Calculator tab
        elif action == "View Logs":
            self.tabs.setCurrentIndex(2)  # Logs tab
        elif action == "Clear History":
            self._clear_calc_history()
        elif action == "Export Stats":
            self._export_stats()

    # Calculator methods
    def _calc_button_pressed(self, label):
        if label == '=':
            self._calculate()
        elif label == 'C':
            self.calc_expression = ""
            self.calc_display.setText("")
            self.calc_result_display.setText("")
        elif label == 'CE':
            self.calc_expression = ""
            self.calc_display.setText("")
        elif label in ['M+', 'M-', 'MR', 'MC']:
            self._handle_memory(label)
        elif label == 'Ans':
            self.calc_expression += str(self.calc_memory)
            self.calc_display.setText(self.calc_expression)
        elif label == 'hist':
            self._show_calc_history()
        elif label in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']:
            self._trig_function(label)
        elif label == 'deg':
            self.calc_degrees = True
        elif label == 'rad':
            self.calc_degrees = False
        else:
            self.calc_expression += label
            self.calc_display.setText(self.calc_expression)

    def _calculate(self):
        try:
            # Replace variables
            expr = self.calc_expression
            for var, edit in self.calc_vars.items():
                try:
                    value = float(edit.text())
                    expr = expr.replace(var, str(value))
                except:
                    pass
            
            # Safe evaluation
            result = self._safe_eval(expr)
            self.calc_result_display.setText(str(result))
            self.calc_memory = result
            
            # Add to history
            self.calc_history.append(f"{self.calc_expression} = {result}")
            self.calc_history_list.addItem(f"{self.calc_expression} = {result}")
            
            # Log calculation
            self._add_log_entry("INFO", f"Calculation: {self.calc_expression} = {result}")
            
        except Exception as e:
            self.calc_result_display.setText(f"Error: {str(e)}")
            self._add_log_entry("ERROR", f"Calculation error: {str(e)}")

    def _safe_eval(self, expr):
        # Replace math functions
        expr = expr.replace('^', '**')
        expr = expr.replace('sqrt', 'math.sqrt')
        expr = expr.replace('log', 'math.log10')
        expr = expr.replace('ln', 'math.log')
        expr = expr.replace('exp', 'math.exp')
        expr = expr.replace('abs', 'abs')
        expr = expr.replace('pi', 'math.pi')
        expr = expr.replace('e', 'math.e')
        
        # Handle trig functions with degree/radian conversion
        for func in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']:
            if func in expr:
                if self.calc_degrees and func in ['sin', 'cos', 'tan']:
                    expr = expr.replace(func, f'math.{func}(math.radians')
                    expr += ')'
                elif not self.calc_degrees or func in ['asin', 'acos', 'atan']:
                    expr = expr.replace(func, f'math.{func}')
        
        # Restrict to safe functions
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "pow": pow})
        
        return eval(expr, {"__builtins__": {}}, allowed_names)

    def _handle_memory(self, op):
        if op == 'M+':
            try:
                self.calc_memory += float(self.calc_result_display.text())
            except:
                pass
        elif op == 'M-':
            try:
                self.calc_memory -= float(self.calc_result_display.text())
            except:
                pass
        elif op == 'MR':
            self.calc_display.setText(str(self.calc_memory))
            self.calc_expression = str(self.calc_memory)
        elif op == 'MC':
            self.calc_memory = 0.0

    def _trig_function(self, func):
        if self.calc_degrees:
            self.calc_expression += f"{func}(math.radians("
        else:
            self.calc_expression += f"math.{func}("
        self.calc_display.setText(self.calc_expression)

    def _plot_function(self):
        # Placeholder for plotting
        QMessageBox.information(self, "Plot", "Plotting functionality coming soon!")

    def _convert_units(self):
        # Simple unit conversion
        conversions = {
            "meters": 1.0,
            "feet": 3.28084,
            "inches": 39.3701,
            "cm": 100.0,
            "mm": 1000.0,
            "km": 0.001
        }
        
        try:
            value = float(self.unit_value.text())
            from_unit = self.unit_from.currentText()
            to_unit = self.unit_to.currentText()
            
            # Convert to meters first, then to target
            meters = value / conversions[from_unit]
            result = meters * conversions[to_unit]
            
            self.calc_result_display.setText(f"{value} {from_unit} = {result:.4f} {to_unit}")
            self._add_log_entry("INFO", f"Unit conversion: {value} {from_unit} = {result:.4f} {to_unit}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Conversion error: {str(e)}")

    def _load_history_item(self, item):
        text = item.text()
        expr = text.split(' = ')[0]
        self.calc_expression = expr
        self.calc_display.setText(expr)

    def _clear_calc_history(self):
        self.calc_history.clear()
        self.calc_history_list.clear()
        self._add_log_entry("INFO", "Calculation history cleared")

    def _show_calc_history(self):
        # Show history in a dialog or something
        pass

    # Logs methods
    def _add_log_entry(self, level, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {level}: {message}"
        self.log_entries.append((level, entry))
        self._update_log_display()

    def _update_log_filters(self):
        for level, cb in self.filter_checkboxes.items():
            self.log_filters[level] = cb.isChecked()
        self._update_log_display()

    def _filter_logs(self):
        self._update_log_display()

    def _update_log_display(self):
        search_text = self.log_search.text().lower()
        filtered_entries = []
        
        for level, entry in self.log_entries:
            if self.log_filters.get(level, True):
                if search_text in entry.lower():
                    filtered_entries.append(entry)
        
        self.log_display.setText('\n'.join(filtered_entries))
        self.log_stats_label.setText(f"Total entries: {len(self.log_entries)} | Filtered: {len(filtered_entries)}")

    def _clear_logs(self):
        self.log_entries.clear()
        self._update_log_display()
        self._add_log_entry("INFO", "Logs cleared")

    def _export_logs(self):
        # Placeholder for export
        QMessageBox.information(self, "Export", "Log export functionality coming soon!")

    # Placeholder methods
    def _submit_feature_request(self):
        request = self.feature_request.toPlainText()
        if request.strip():
            self._add_log_entry("INFO", f"Feature request submitted: {request[:50]}...")
            self.feature_request.clear()
            QMessageBox.information(self, "Thank you!", "Your feature request has been submitted.")
        else:
            QMessageBox.warning(self, "Error", "Please enter a feature request.")

    def _easter_egg(self):
        messages = [
            "🎉 You found the easter egg!",
            "🚀 To the moon!",
            "🌟 You're awesome!",
            "🎈 Pop!",
            "🎪 Welcome to the circus!"
        ]
        QMessageBox.information(self, "Surprise!", random.choice(messages))

    # Export stats
    def _export_stats(self):
        # Placeholder
        QMessageBox.information(self, "Export", "Stats export functionality coming soon!")
