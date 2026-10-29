"""Live plot of one band: 8 raw EMG channels (counts) and 6 IMU axes. No filtering.

usage: emg-probe view [--port /dev/cu.A] [--window 1000]
Space pauses (the port keeps being read); red dots mark samples at the 24-bit limits.
"""
import argparse
from collections import deque
import queue
import time

from emgprobe.device import BAUD, SerialReader, find_port


class LiveViewer:
    def __init__(self, reader, window=1000):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Button, Slider
        self.reader = reader
        self.samples = deque(maxlen=10000)
        self.imu_samples = deque(maxlen=10000)
        self.position = -1
        self.previous_counter = None
        self.aa = self.bb = self.discontinuities = 0
        self.paused = False
        self.autoscale = True
        self.window = window
        self.fig, axes = plt.subplots(4, 2, figsize=(13, 9))
        self.axes = list(axes.flat)
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.83, bottom=0.17,
                                 hspace=0.62, wspace=0.25)
        self.fig.canvas.manager.set_window_title('Waveletech — live EMG')
        self.fig.suptitle('Waveletech | live raw counts', fontsize=16, y=0.985)
        self.fig.text(0.5, 0.944,
                      'Provisional: AA = 8 x signed 24-bit, big-endian | independent Y scales | no filtering',
                      ha='center', fontsize=9)
        self.status_text = self.fig.text(0.09, 0.897, 'Connecting...', fontsize=9)
        self.lines, self.limits = [], []
        for ch, ax in enumerate(self.axes):
            line, = ax.plot([], [], color='#2563eb', lw=1)
            limit, = ax.plot([], [], '.', color='#dc2626', ms=3)
            self.lines.append(line)
            self.limits.append(limit)
            ax.set_title(f'CH{ch+1}', loc='left', fontsize=10)
            ax.set_ylabel('Counts', fontsize=9)
            ax.set_xlabel('Packet index', fontsize=9)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=8)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        self.pause_button = Button(self.fig.add_axes([0.09, 0.08, 0.13, 0.04]), 'Pause')
        self.scale_button = Button(self.fig.add_axes([0.24, 0.08, 0.17, 0.04]), 'Y autoscale: ON')
        self.slider = Slider(self.fig.add_axes([0.57, 0.087, 0.34, 0.025]),
                             'AA samples', 100, 5000, valinit=window, valstep=100)
        self.pause_button.on_clicked(self.toggle_pause)
        self.scale_button.on_clicked(self.toggle_scale)
        self.slider.on_changed(self.set_window)
        self.fig.text(0.09, 0.025,
                      'Space: pause/resume | Pause, then toolbar zoom/pan | Red dots: 24-bit limits | Pause keeps receiving',
                      fontsize=9)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', lambda event: reader.close())

        self.imu_fig, imu_axes = plt.subplots(2, 3, figsize=(12, 6))
        self.imu_axes = list(imu_axes.flat)
        self.imu_fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12,
                                     hspace=0.42, wspace=0.30)
        self.imu_fig.canvas.manager.set_window_title('Waveletech — live IMU')
        self.imu_fig.suptitle('Waveletech | live IMU', fontsize=16, y=0.97)
        imu_names = ('Gyro X', 'Gyro Y', 'Gyro Z',
                     'Accel X', 'Accel Y', 'Accel Z')
        imu_units = ('rad/s', 'rad/s', 'rad/s', 'm/s²', 'm/s²', 'm/s²')
        imu_colors = ('#dc2626', '#16a34a', '#2563eb',
                      '#ea580c', '#7c3aed', '#0891b2')
        self.imu_lines = []
        for name, unit, color, ax in zip(imu_names, imu_units, imu_colors,
                                         self.imu_axes):
            line, = ax.plot([], [], color=color, lw=1)
            self.imu_lines.append(line)
            ax.set_title(name, loc='left', fontsize=10)
            ax.set_ylabel(unit, fontsize=9)
            ax.set_xlabel('Packet index', fontsize=9)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=8)
        self.imu_fig.text(
            0.5, 0.01,
            'Vendor scale: gyro 0.0012 rad/s per count | accel 0.0005978 m/s² per count',
            ha='center', fontsize=9)
        self.imu_fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.animation = FuncAnimation(self.fig, self.update, interval=33,
                                       cache_frame_data=False)

    def toggle_pause(self, event=None):
        self.paused = not self.paused
        self.pause_button.label.set_text('Resume live' if self.paused else 'Pause')
        self.fig.canvas.draw_idle()

    def toggle_scale(self, event=None):
        self.autoscale = not self.autoscale
        self.scale_button.label.set_text(f"Y autoscale: {'ON' if self.autoscale else 'OFF'}")
        self.fig.canvas.draw_idle()

    def set_window(self, value):
        self.window = int(value)

    def on_key(self, event):
        if event.key == ' ':
            self.toggle_pause()

    def update(self, _=None):
        import numpy as np
        for _ in range(20000):
            try:
                    kind, counter, values = self.reader.frames.get_nowait()
            except queue.Empty:
                    break
            step = 1 if self.previous_counter is None else (counter-self.previous_counter) % 256
            if self.previous_counter is not None and step != 1:
                    self.discontinuities += 1
                    self.samples.append((self.position + 0.5, (float('nan'),)*8))
                    self.imu_samples.append((self.position + 0.5, (float('nan'),)*6))
            self.position += max(step, 1)
            self.previous_counter = counter
            if kind == 0xAA:
                    self.aa += 1
                    self.samples.append((self.position, values))
            else:
                    self.bb += 1
                    self.imu_samples.append((self.position, values))
        status = self.reader.status
        last = self.reader.last_received
        if status == 'Connected' and (last is None or time.monotonic()-last > 2):
            status = 'Connected, waiting for data'
        self.status_text.set_text(
            f"{'PAUSED | ' if self.paused else ''}{status[:110]}\n"
            f'AA {self.aa:,}  BB {self.bb:,}  Counter breaks {self.discontinuities}  '
            f'Discarded bytes {self.reader.parser.discarded}  Queue overflow {self.reader.overflow}')
        if self.paused:
            return
        if self.samples:
            data = list(self.samples)[-self.window:]
            x = np.array([row[0] for row in data])
            values = np.array([row[1] for row in data])
            for ch, ax in enumerate(self.axes):
                    y = values[:, ch]
                    self.lines[ch].set_data(x, y)
                    mask = (y == -8388608) | (y == 8388607)
                    self.limits[ch].set_data(x[mask], y[mask])
                    ax.set_xlim(x[0], max(x[0]+1, x[-1]))
                    if self.autoscale:
                        self.set_auto_ylim(ax, y)
                    ax.set_title(
                        f'CH{ch+1} | limit hits {int(mask.sum())}/{int(np.isfinite(y).sum())}',
                        loc='left', fontsize=10)
        if self.imu_samples:
            data = list(self.imu_samples)[-self.window:]
            x = np.array([row[0] for row in data])
            values = np.array([row[1] for row in data])
            for axis, ax in enumerate(self.imu_axes):
                    y = values[:, axis]
                    self.imu_lines[axis].set_data(x, y)
                    ax.set_xlim(x[0], max(x[0]+1, x[-1]))
                    if self.autoscale:
                        self.set_auto_ylim(ax, y)
            self.imu_fig.canvas.draw_idle()

    @staticmethod
    def set_auto_ylim(ax, y):
        import numpy as np
        finite = y[np.isfinite(y)]
        if finite.size:
            low, high = finite.min(), finite.max()
            margin = max((high-low)*0.1, 1e-6)
            ax.set_ylim(low-margin, high+margin)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='emg-probe view', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port')
    parser.add_argument('--baud', type=int, default=BAUD)
    parser.add_argument('--window', type=int, default=1000, choices=range(100, 5001, 100), metavar='100..5000')
    args = parser.parse_args(argv)
    if not args.port:
        try:args.port = find_port()
        except ValueError as e:parser.error(str(e))
    import matplotlib.pyplot as plt
    reader = SerialReader(args.port, args.baud)
    LiveViewer(reader, args.window)
    reader.start()
    try:
        plt.show()
    finally:
        reader.close()


if __name__ == '__main__':
    main()
