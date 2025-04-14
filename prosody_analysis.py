import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import parselmouth
from parselmouth.praat import call
from sympy.physics.pring import energy


def compute_features(audio):
    '''Take audio file_path as input and return pitch, loudness and duration as features'''
    y, sr = librosa.load(audio, sr=None)
    pitches, magnitude = librosa.piptrack(y=y, sr=sr)

    energy = np.sum(librosa.feature.rms(y=y), axis=1)

    duration = librosa.get_duration(y=y, sr=sr)

    return pitches, energy, duration

class prosody:
    def __init__(self, pitch, energy, duration):
        self.energy = energy
        self.pitch = pitch
        self.duration = duration




