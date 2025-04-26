import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import tkinter.font as tkFont

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage


class ROS2Listener(Node):
    def __init__(self, topic_name):
        super().__init__('color_filter_app_gui')
        self.bridge = CvBridge()
        self.latest_image = None
        self.lock = threading.Lock()
        self.topic = topic_name
        self.create_subscription(
            RosImage,
            self.topic,
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self.lock:
            self.latest_image = cv_img

    def get_image(self):
        with self.lock:
            return self.latest_image.copy() if self.latest_image is not None else None


class ColorFilterApp:
    def __init__(self, root):
        self.root = root
        self.cap = None
        self.file_image = None
        self.photo = None

        # Default ROS2 topic
        self.topic_var = tk.StringVar(value='/camera/camera/color/image_raw')

        # Default HSV preset for yellow
        default = {'h_min': 20, 'h_max': 30, 's_min': 100, 's_max': 255, 'v_min': 100, 'v_max': 255}
        self.h_min_var = tk.IntVar(value=default['h_min'])
        self.h_max_var = tk.IntVar(value=default['h_max'])
        self.s_min_var = tk.IntVar(value=default['s_min'])
        self.s_max_var = tk.IntVar(value=default['s_max'])
        self.v_min_var = tk.IntVar(value=default['v_min'])
        self.v_max_var = tk.IntVar(value=default['v_max'])
        self.lower_strvar = tk.StringVar()
        self.upper_strvar = tk.StringVar()

        # Default bold font
        self.default_font = tkFont.Font(family="Helvetica", size=12, weight="bold")

        # Build UI first so we can read topic_var
        self._build_ui()
        # Initialize ROS listener with topic
        rclpy.init()
        self.ros_listener = ROS2Listener(self.topic_var.get())

        # Trace HSV vars to update composite entries
        for var in (self.h_min_var, self.s_min_var, self.v_min_var,
                    self.h_max_var, self.s_max_var, self.v_max_var):
            var.trace_add('write', self._update_presets)

        self._update_presets()
        self.update_frame()

    def _build_ui(self):
        self.root.configure(bg='gray20')

        # ROS topic entry
        topic_frame = tk.Frame(self.root, bg='gray20')
        topic_frame.pack(padx=10, pady=5, fill='x')
        tk.Label(topic_frame, text='ROS2 Topic:', bg='gray20', fg='white', font=self.default_font).pack(side=tk.LEFT)
        tk.Entry(topic_frame, textvariable=self.topic_var, font=self.default_font, bg='gray30', fg='white', insertbackground='white').pack(side=tk.LEFT, fill='x', expand=True, padx=5)

        # Input selection frame
        input_frame = tk.Frame(self.root, bg='gray20')
        input_frame.pack(padx=10, pady=5)
        self.input_var = tk.StringVar(value='file')
        rb_opts = {'bg': 'gray20', 'fg': 'white', 'selectcolor': 'gray30', 'activebackground': 'gray25', 'activeforeground': 'white', 'font': self.default_font}
        tk.Radiobutton(input_frame, text='File', variable=self.input_var, value='file', command=self._on_source_change, **rb_opts).pack(side=tk.LEFT)
        tk.Radiobutton(input_frame, text='Camera', variable=self.input_var, value='camera', command=self._on_source_change, **rb_opts).pack(side=tk.LEFT)
        tk.Radiobutton(input_frame, text='ROS2', variable=self.input_var, value='ros2', command=self._on_source_change, **rb_opts).pack(side=tk.LEFT)
        btn_opts = {'bg': 'gray30', 'fg': 'white', 'activebackground': 'gray40', 'activeforeground': 'white', 'font': self.default_font}
        self.load_btn = tk.Button(input_frame, text='Load Image', command=self.load_image, **btn_opts)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Sliders frame
        slider_frame = tk.Frame(self.root, bg='gray20')
        slider_frame.pack(padx=10, pady=5, fill='x')
        lbl_opts = {'bg': 'gray20', 'fg': 'white', 'font': self.default_font}
        scale_opts = {'bg': 'gray20', 'fg': 'white', 'troughcolor': 'gray30', 'highlightthickness': 0, 'bd': 0, 'sliderrelief': tk.FLAT, 'font': self.default_font}
        entry_opts = {'bg': 'gray30', 'fg': 'white', 'insertbackground': 'white', 'bd': 1, 'font': self.default_font}
        def add_row(label, var, row, max_val):
            tk.Label(slider_frame, text=label, **lbl_opts).grid(row=row, column=0, sticky='w')
            tk.Scale(slider_frame, from_=0, to=max_val, orient=tk.HORIZONTAL, variable=var, **scale_opts).grid(row=row, column=1, sticky='we')
            tk.Entry(slider_frame, textvariable=var, width=5, **entry_opts).grid(row=row, column=2, padx=5)
        add_row('Hue min', self.h_min_var, 0, 179)
        add_row('Hue max', self.h_max_var, 1, 179)
        add_row('Sat min', self.s_min_var, 2, 255)
        add_row('Sat max', self.s_max_var, 3, 255)
        add_row('Val min', self.v_min_var, 4, 255)
        add_row('Val max', self.v_max_var, 5, 255)
        slider_frame.columnconfigure(1, weight=1)

        # Composite preset entries
        preset_frame = tk.Frame(self.root, bg='gray20')
        preset_frame.pack(padx=10, pady=(0,5), fill='x')
        tk.Label(preset_frame, text='Lower HSV:', **lbl_opts).grid(row=0, column=0, sticky='w')
        tk.Entry(preset_frame, textvariable=self.lower_strvar, width=20, **entry_opts).grid(row=0, column=1, sticky='we', padx=5)
        tk.Label(preset_frame, text='Upper HSV:', **lbl_opts).grid(row=1, column=0, sticky='w')
        tk.Entry(preset_frame, textvariable=self.upper_strvar, width=20, **entry_opts).grid(row=1, column=1, sticky='we', padx=5)
        preset_frame.columnconfigure(1, weight=1)

        # Image display label
        self.image_label = tk.Label(self.root, bg='gray20')
        self.image_label.pack(padx=10, pady=5)

    def _update_presets(self, *args):
        low = f"{self.h_min_var.get()},{self.s_min_var.get()},{self.v_min_var.get()}"
        high = f"{self.h_max_var.get()},{self.s_max_var.get()},{self.v_max_var.get()}"
        self.lower_strvar.set(low)
        self.upper_strvar.set(high)

    def _on_source_change(self):
        src = self.input_var.get()
        self.load_btn.config(state=(tk.NORMAL if src == 'file' else tk.DISABLED))
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.file_image = cv2.imread(path)

    def update_frame(self):
        src = self.input_var.get()
        frame = None
        if src == 'file' and self.file_image is not None:
            frame = self.file_image.copy()
        elif src == 'camera':
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
            ret, frame = self.cap.read()
            if not ret:
                frame = None
        elif src == 'ros2':
            rclpy.spin_once(self.ros_listener, timeout_sec=0)
            frame = self.ros_listener.get_image()
        
        if frame is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array([self.h_min_var.get(), self.s_min_var.get(), self.v_min_var.get()])
            upper = np.array([self.h_max_var.get(), self.s_max_var.get(), self.v_max_var.get()])
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,0), 2)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            self.photo = ImageTk.PhotoImage(img_pil)
            self.image_label.config(image=self.photo)

        self.root.after(30, self.update_frame)


def on_close(root, ros_listener, cap):
    if cap is not None:
        cap.release()
    ros_listener.destroy_node()
    rclpy.shutdown()
    root.destroy()


def main():
    root = tk.Tk()
    root.title('Color Filter Adjustment App')
    app = ColorFilterApp(root)
    root.protocol('WM_DELETE_WINDOW', lambda: on_close(root, app.ros_listener, app.cap))
    root.mainloop()


if __name__ == '__main__':
    main()

