# -*- coding: utf-8 -*-
'''recorder.py
Código original em https://gist.github.com/sloria/5693955
'''

import numpy as np
from scipy.io import wavfile
import subprocess
import threading
import wave

from time import sleep, time


class Recorder(object):
    '''A recorder class for recording audio to a WAV file.
    Records in mono by default.
    '''

    def __init__(self, channels=1, rate=48000, frames_per_buffer=1024):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer

    def open(self, fname):
        return RecordingFile(fname, self.rate, self.frames_per_buffer)


class RecordingFile(object):
    def __init__(self, fname, rate, frames_per_buffer):
        self.fname = fname
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        pass

    def record(self, duration):
        cmd = f'arecord --rate={self.rate} --format=S32_LE --duration={duration} {self.fname}'
        proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = np.array([])
        try:
            with wave.open(proc.stdout, 'rb') as rec:
                data.append(rec.readframes(self.frames_per_buffer))
            wavfile.write(self.fname, self.rate, data)
        except EOFError:
            pass

        return None
