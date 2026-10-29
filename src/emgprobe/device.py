"""Waveletech 8-channel EMG band: frame parsing, serial reading, simulated frames, port lookup.

29-byte frames: D2 D2 D2, AA/BB type, counter, 24-byte payload.
AA contains 8 x signed 24-bit big-endian EMG values. BB contains two reserved bytes, then three
signed 16-bit big-endian gyroscope axes followed by three accelerometer axes.
The 8-bit counter is shared by AA and BB frames. No sampling frequency is assumed here.
"""
import queue
import threading
import time

import numpy as np

GYRO_SCALE = 0.0012       # raw count -> rad/s, per vendor manual
ACCEL_SCALE = 0.0005978   # raw count -> m/s^2, per vendor manual
BAUD = 921600


class FrameParser:
    def __init__(self):
        self.buffer = bytearray()
        self.discarded = 0

    def feed(self, chunk):
        """Bytes in, [(kind 0xAA/0xBB, counter, values)] out; partial frames stay buffered."""
        self.buffer.extend(chunk)
        frames = []
        while len(self.buffer) >= 4:
            start = self.buffer.find(b'\xd2\xd2\xd2')
            if start < 0:
                self.discarded += len(self.buffer) - 2
                del self.buffer[:-2]
                break
            if start:
                self.discarded += start
                del self.buffer[:start]
            if len(self.buffer) < 4:
                break
            if self.buffer[3] not in (0xAA, 0xBB):
                self.discarded += 1
                del self.buffer[0]
                continue
            if len(self.buffer) < 29:
                break
            frame = bytes(self.buffer[:29])
            del self.buffer[:29]
            if frame[3] == 0xAA:
                values = tuple(int.from_bytes(frame[i:i+3], 'big', signed=True) for i in range(5, 29, 3))
            else:
                # BB payload bytes 1-2 are reserved. The manual maps bytes 3-14 to gyro X/Y/Z and accelerometer X/Y/Z.
                raw = tuple(int.from_bytes(frame[i:i+2], 'big', signed=True) for i in range(7, 19, 2))
                values = tuple(v * GYRO_SCALE for v in raw[:3]) + tuple(v * ACCEL_SCALE for v in raw[3:])
            frames.append((frame[3], frame[4], values))
        return frames


class SerialReader(threading.Thread):
    """Parsed frames from a serial port into a queue (used by the live viewer)."""
    def __init__(self, port, baud=BAUD):
        super().__init__(daemon=True)
        self.port, self.baud = port, baud
        self.frames = queue.Queue(maxsize=20000)
        self.stop_event = threading.Event()
        self.parser = FrameParser()
        self.status = 'Connecting...'
        self.overflow = 0
        self.last_received = None

    def run(self):
        import serial
        try:
            with serial.Serial(self.port, baudrate=self.baud, timeout=0.05) as stream:
                self.status = 'Connected'
                while not self.stop_event.is_set():
                    chunk = stream.read(min(max(stream.in_waiting, 1), 8192))
                    if not chunk:
                        continue
                    self.last_received = time.monotonic()
                    for frame in self.parser.feed(chunk):
                        try:
                            self.frames.put_nowait(frame)
                        except queue.Full:
                            self.overflow += 1
        except (serial.SerialException, OSError) as error:
            self.status = f'Serial error: {error}'
        finally:
            if self.stop_event.is_set():
                self.status = 'Stopped'

    def close(self):
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=1)


def encode(emg_rows, counter, rng, imu_every=9):
    """Simulated byte stream: one AA frame per row of 8 counts, and a BB frame whenever the counter hits a multiple of
    `imu_every`. The simulated IMU is still: small gyro bias + noise, gravity on z. Returns (bytes, next counter)."""
    out = []
    for row in emg_rows:
        out.append(b'\xd2\xd2\xd2\xaa'+bytes([counter % 256])+b''.join(int(v).to_bytes(3, 'big', signed=True) for v in row))
        counter += 1
        if counter % imu_every == 0:
            imu = np.r_[np.array([.01, -.008, -.006])+rng.normal(0, .001, 3), np.array([0, 0, 9.80665])+rng.normal(0, .01, 3)]
            raw = np.rint(imu/np.array([GYRO_SCALE]*3+[ACCEL_SCALE]*3)).astype(int)
            out.append(b'\xd2\xd2\xd2\xbb'+bytes([counter % 256])+bytes(2)+b''.join(int(v).to_bytes(2, 'big', signed=True) for v in raw)+bytes(10))
            counter += 1
    return b''.join(out), counter


def port_info(port):
    """USB identity of a serial port; identical bands may share descriptions, so location (USB socket) is kept too."""
    try:
        from serial.tools import list_ports
        for p in list_ports.comports():
            if p.device == port:
                return dict(serial_number=p.serial_number, location=p.location, description=p.description, vid=p.vid, pid=p.pid)
    except Exception:
        pass
    return None


def usb_ports():
    from serial.tools import list_ports
    return [p for p in list_ports.comports() if p.vid is not None]


def find_port():
    """The only USB serial port, or ValueError listing the candidates."""
    ports = [p.device for p in usb_ports()]
    if len(ports) != 1:
        raise ValueError('Give --port. USB serial ports: '+(', '.join(ports) or 'none'))
    return ports[0]
