# realtime_classifier.py
# Real-time plant bioelectrical signal classification.
# Run with real hardware: python realtime_classifier.py --port /dev/ttyUSB0
# Run in Codespaces (virtual): python realtime_classifier.py --virtual

import os, sys, pty, threading, time, argparse
import numpy as np
import serial, serial.tools.list_ports
from collections import deque
from pathlib import Path
import yaml

# Add src directory to path so we can import from it
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

cfg  = yaml.safe_load(open('config/config.yaml'))
FS   = cfg['signal']['sample_rate_hz']          # 8.6 Hz
WIN  = cfg['signal']['window_size_samples']     # 258 samples = 30 s
HOP  = cfg['signal']['hop_size_samples']        # 129 samples = 15 s
ENV  = os.environ.get('PHCI_ENV', 'local')      # 'codespaces' or 'local'

# Import pipeline and feature extractor AFTER sys.path is set
from preprocess import extract_features, FEATURE_NAMES
from pipeline import PHCIPipeline


def detect_hardware_port() -> str:
    '''Return /dev/ttyUSB0 or /dev/ttyACM0 if ESP32 is attached, else None.'''
    for p in serial.tools.list_ports.comports():
        if any(kw in (p.description or '') for kw in
               ['CP210', 'CH340', 'USB Serial', 'UART', 'Arduino']):
            print(f'[Hardware] ESP32 found: {p.device}  ({p.description})')
            return p.device
    return None


class VirtualSignalSource:
    '''
    Generates synthetic plant signal over a pty virtual serial port.
    Used in Codespaces when no real ESP32 is connected.
    Simulates the 5-field CSV format the real ESP32 produces.
    '''
    def __init__(self, species='mimosa', stress='water_stress'):
        self.species = species
        self.stress  = stress
        self.master_fd, slave_fd = pty.openpty()
        self.port    = os.ttyname(slave_fd)
        self._running = True
        self._thread  = threading.Thread(target=self._stream, daemon=True)
        self._thread.start()
        print(f'[Virtual] Port: {self.port}  | {species}/{stress}')

    def _stream(self):
        from generate_data import generate_session, PARAMS
        interval = 1.0 / FS
        while self._running:
            signal = generate_session(self.species, self.stress, duration_min=5)
            for i, v in enumerate(signal):
                if not self._running: break
                ts  = int(time.time() * 1000)
                adc = int(np.clip(v / 4096 * 32768, -32768, 32767))
                line = f'{ts},vsess,0,{adc},{v:.4f}\n'
                try:
                    os.write(self.master_fd, line.encode())
                except OSError:
                    break
                time.sleep(interval)

    def stop(self):
        self._running = False
        try: os.close(self.master_fd)
        except OSError: pass


def run_classifier(port: str, baud: int = 115200):
    '''
    Main inference loop.
    Reads 5-field CSV lines from serial port, buffers samples into 30-second windows,
    extracts features, and runs PHCIPipeline.run() on each complete window.
    '''
    pipe   = PHCIPipeline()
    ser    = serial.Serial(port, baud, timeout=2)
    buffer = deque(maxlen=WIN)   # Rolling buffer of voltage_mv values
    new_samples = 0              # Samples since last classification

    print('Models loaded. Classification running...')
    print('-' * 65)

    while True:
        try:
            raw_line = ser.readline()
            if not raw_line: continue
            decoded = raw_line.decode('utf-8', errors='ignore').strip()
            if not decoded or decoded.startswith('#'): continue

            fields = decoded.split(',')
            if len(fields) < 5: continue
            try:
                voltage_mv = float(fields[4])
            except ValueError:
                continue

            buffer.append(voltage_mv)
            new_samples += 1

            # Classify when window is full and hop has elapsed
            if len(buffer) == WIN and new_samples >= HOP:
                new_samples = 0
                window = np.array(buffer, dtype=np.float32)
                feat_array = extract_features(window)
                x_raw = np.array(feat_array, dtype=np.float64)
                result = pipe.run(x_raw)

                ts = time.strftime('%H:%M:%S')
                print(f'[{ts}] {result["alert"]}')
                if result['stage1']:
                    s1 = result['stage1']
                    print(f'         Species: {s1["species_name"]}  conf={s1["species_confidence"]:.0%}')
                if result['stage2']:
                    s2 = result['stage2']
                    print(f'         Stress:  {s2["stress_name"]}  conf={s2["stress_confidence"]:.0%}')
                print()

        except KeyboardInterrupt:
            print('\nStopping classifier...')
            ser.close()
            break
        except ValueError:
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PHCI Real-Time Classifier')
    parser.add_argument('--port',    default=None,           help='Serial port (e.g. /dev/ttyUSB0)')
    parser.add_argument('--baud',    type=int, default=115200)
    parser.add_argument('--virtual', action='store_true',    help='Force virtual mode')
    parser.add_argument('--species', default='mimosa',       choices=cfg['project']['species'])
    parser.add_argument('--stress',  default='water_stress', choices=cfg['project']['stress_states'])
    args = parser.parse_args()

    vsrc = None
    port = args.port

    if args.virtual or ENV == 'codespaces' or port is None:
        if port is None:
            detected = detect_hardware_port()
            if detected and not args.virtual:
                port = detected
            else:
                vsrc = VirtualSignalSource(args.species, args.stress)
                port = vsrc.port

    print(f'Starting real-time classifier on {port}')
    try:
        run_classifier(port, args.baud)
    finally:
        if vsrc: vsrc.stop()
