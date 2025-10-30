#############################################
# To Do List (Kivy Version)
# SWITCH TO DATABASE
# 4. add feature to edit a row from the csv file
# 5. add feature to search for a specific barcode
# 6. add feature to filter by date
# 7. add feature to export the csv file
# 8. add feature to convert barcode to text
# 9. add voice recognition placeholder
# boot on start up
#############################################

import csv
import os
from datetime import datetime
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.recycleview import RecycleView
from kivy.uix.recyclegridlayout import RecycleGridLayout
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.scrollview import ScrollView
from kivy.properties import BooleanProperty
from kivy.core.window import Window
import facial_recognition as fr

# set fullscreen for Raspberry Pi (make safe — some SDL backends may raise)
try:
    Window.fullscreen = True
except Exception:
    try:
        Window.fullscreen = 'auto'
    except Exception:
        # ignore if fullscreen cannot be set in this environment
        pass

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scans.csv")
REFRESH_INTERVAL = 15  # seconds

# ------------------- GUI COMPONENTS -------------------

class SelectableLabel(RecycleDataViewBehavior, Label):
    """Selectable row in the list."""
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        return super().refresh_view_attrs(rv, index, data)

    def on_touch_down(self, touch):
        if super().on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos):
            self.selected = not self.selected
            rv = self.parent.parent
            rv.select_row(self.index)
            return True
        return False

    def apply_selection(self, rv, index, is_selected):
        self.selected = is_selected
        self.color = (1, 0, 0, 1) if is_selected else (1, 1, 1, 1)

class BarcodeList(RecycleView):
    """Displays the log entries."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []
        # use the SelectableLabel class as the viewclass
        self.viewclass = SelectableLabel

        # create a layout manager and add it as a child of the RecycleView
        layout = RecycleGridLayout(cols=3,
                                   default_size=(None, 48),
                                   default_size_hint=(1, None),
                                   size_hint_y=None)
        layout.bind(minimum_height=layout.setter('height'))
        # layout_manager must be a widget child of the RecycleView
        self.add_widget(layout)
        self.layout_manager = layout

    def load_data(self, rows):
        self.data = [{"text": f"{r[0]} | {r[1]} | {r[2]}"} for r in rows]

    def select_row(self, index):
        for i, row in enumerate(self.data):
            if i == index:
                self.data[i]["selected"] = not self.data[i].get("selected", False)


# ------------------- MAIN APP LOGIC -------------------

class InventoryApp(BoxLayout):
    """Main application logic and UI."""
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        # Title
        self.add_widget(Label(text="[b]Medical Inventory System[/b]", markup=True, font_size=32, size_hint_y=None, height=60))

        # Buttons row
        btns = BoxLayout(size_hint_y=None, height=60)
        btns.add_widget(Button(text="Open Camera", on_press=lambda x: self.face_recognition()))
        btns.add_widget(Button(text="Log Scan", on_press=lambda x: self.log_scan()))
        btns.add_widget(Button(text="Delete Selected", on_press=lambda x: self.delete_selected()))
        btns.add_widget(Button(text="Quit", on_press=lambda x: App.get_running_app().stop()))
        self.add_widget(btns)

        # Scrollable list
        self.viewer = BarcodeList()
        scroll = ScrollView()
        scroll.add_widget(self.viewer)
        self.add_widget(scroll)

        # Load initial data and schedule refresh
        self.load_data()
        Clock.schedule_interval(lambda dt: self.load_data(), REFRESH_INTERVAL)

    # ------------------- Data Logic -------------------

    def face_recognition(self):
        """Integrate face recognition."""
        result = fr.main()
        if isinstance(result, int):
            msg = {
                4: "Couldn't find camera",
                3: "No reference folder found",
                2: "No faces found in reference images"
            }.get(result, "Unknown error")
            self.show_popup("Face Recognition Error", msg)
            return ""
        if isinstance(result, (list, tuple)) and result:
            name = str(result[0])
            self.show_popup("Face Recognition", f"Detected: {name}")
            return name
        self.show_popup("Face Recognition", "No known faces detected.")
        return ""

    def prompt_barcode(self):
        """Popup for barcode input."""
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        input_box = TextInput(hint_text="Scan barcode and press OK", multiline=False, font_size=20)
        layout.add_widget(input_box)

        result = {"barcode": None}

        def on_ok(instance):
            result["barcode"] = input_box.text.strip()
            popup.dismiss()

        btn = Button(text="OK", size_hint_y=None, height=50, on_press=on_ok)
        layout.add_widget(btn)

        popup = Popup(title="Scan Barcode", content=layout, size_hint=(0.6, 0.4))
        popup.open()
        input_box.focus = True

        # Wait for popup to close (Kivy async-style)
        def check_result(dt):
            if result["barcode"]:
                return False
            Clock.schedule_once(check_result, 0.1)
        Clock.schedule_once(check_result, 0.1)

        return result

    def log_scan(self):
        """Log scan with timestamp and user."""
        user = self.face_recognition()
        if not user:
            self.show_popup("Authentication Required", "Face recognition must be successful before scanning barcodes.")
            return

        popup = self.prompt_barcode()

        # Delay checking barcode input until popup closes
        def finalize_log(dt):
            barcode = popup.get("barcode")
            if not barcode:
                return
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_exists = os.path.exists(LOG_FILE)
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "barcode", "user"])
                writer.writerow([ts, barcode, user])
            self.show_popup("Logged", f"Logged {barcode} at {ts} by {user}")
            self.load_data()
        Clock.schedule_once(finalize_log, 0.5)

    def load_data(self):
        """Reload from CSV."""
        if not os.path.exists(LOG_FILE):
            self.viewer.load_data([])
            return
        with open(LOG_FILE, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = [r for r in reader if len(r) >= 3]
        self.viewer.load_data(rows)

    def delete_selected(self):
        """Delete selected rows."""
        # NOTE: Placeholder — actual selection logic in Kivy RV needs explicit tracking.
        self.show_popup("Delete", "Row deletion not yet implemented in Kivy version.")

    def show_popup(self, title, message):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        layout.add_widget(Label(text=message, font_size=18))
        layout.add_widget(Button(text="OK", size_hint_y=None, height=50, on_press=lambda x: popup.dismiss()))
        popup = Popup(title=title, content=layout, size_hint=(0.6, 0.4))
        popup.open()


# ------------------- RUN -------------------

class MedicalInventoryApp(App):
    def build(self):
        return InventoryApp()


MedicalInventoryApp().run()