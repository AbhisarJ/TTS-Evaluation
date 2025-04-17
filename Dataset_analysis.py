import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
from Segment_analysis import Audio_files
import parselmouth
from parselmouth.praat import call


class Eval_data():
    def __init__(self, source:os.PathLike):
        self.source = source
        self.data = self.create_dataset()


    def create_dataset(self):
        file_paths = []
        for root, dirs, files in os.walk(self.source, topdown=True):
            for file in files:
                full = os.path.join(root, file)
                file_paths.append(full)

        labels = []
        for path in file_paths:
            relative_path = os.path.relpath(path, self.source)
            path_parts = relative_path.split(os.sep)
            labels.append({"model": path_parts[0], "speaker": path_parts[1], "language":path_parts[2], "path": path})

        df = pd.DataFrame(labels)
        return df

    def load_f0(self, audio):
        aud , sr = librosa.load(audio)
        f0, v1, v2 = librosa.pyin(aud, fmin=40, fmax=400, sr=sr)
        return f0

    # Variance
    def compute_variance(self, f0):
        valid_f0 = f0[~np.isnan(f0)]
        return np.var(valid_f0)

    # pitch range

    def compute_pitch_range(self, f0):
        valid_f0 = f0[~np.isnan(f0)]
        return np.max(valid_f0) - np.min(valid_f0)

    # voiced ratio

    def compute_voicing_ratio(self , voiced_flag):
        return np.mean(voiced_flag)

    # mean f0
    def compute_mean_f0(self, f0):
        valid_f0 = f0[~np.isnan(f0)]
        return np.mean(valid_f0)

    # slope
    def compute_slope(self, f0, sr, hop_length):
        valid_f0 = f0[~np.isnan(f0)]
        time = np.arange(len(valid_f0)) * hop_length / sr
        slope = np.gradient(valid_f0, time)
        return np.mean(np.abs(slope))

    def compute_metrics(self):
        f0_list = []
        variance_list = []
        pitch_range_list = []
        mean_f0_list = []
        slope_list = []
        h_list=[]

        for audio_path in self.data["path"]:
            try:
                f0 = self.load_f0(audio_path)
                f0_list.append(f0)
                variance_list.append(self.compute_variance(f0))
                pitch_range_list.append(self.compute_pitch_range(f0))
                mean_f0_list.append(self.compute_mean_f0(f0))
                slope_list.append(self.compute_slope(f0, 24000, 512))
                h_list.append(self.compute_harmonics(audio_path))
            except Exception as e:
                print(f"Error processing file {audio_path}: {e}")
                f0_list.append(None)
                variance_list.append(None)
                pitch_range_list.append(None)
                mean_f0_list.append(None)
                slope_list.append(None)
                h_list.append(None)

        self.data["f0"] = f0_list
        self.data["Variance"] = variance_list
        self.data["Pitch Range"] = pitch_range_list
        self.data["Mean f0"] = mean_f0_list
        self.data["Slope"] = slope_list
        self.data["Consistency"] = h_list

    def compute_harmonics(self, path):
        sound = parselmouth.Sound(path)
        harmonicity = sound.to_harmonicity()

        # Extract HNR value (average across the signal)
        hnr = call(harmonicity, "Get mean", 0, 0)
        h_score=0
        if 0<hnr<10:
            h_score=(hnr*2)/10
        if 10<hnr<15:
            h_score=(((hnr-10)*2)/5)+2
        if 15<hnr<30:
            h_score=(((hnr-15))/15)+4
        return round(h_score, 2)

    def Emotional_score(self, val):
        path = self.data.loc[val, "path"]
        y, sr = librosa.load(path, sr=None)

        # Extract F0 contour using librosa.pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=500, sr=sr)

        # Generate time points for the F0 contour
        time = np.linspace(0, len(y) / sr, len(f0))

        # Remove NaN values (unvoiced segments)
        voiced_f0 = f0[~np.isnan(f0)]
        voiced_time = time[~np.isnan(f0)]

        # Compute slopes
        slopes = np.diff(voiced_f0) / np.diff(voiced_time)

        # Metrics
        mean_slope = np.mean(slopes)
        slope_variance = np.var(slopes)
        steep_slope_proportion = np.sum(np.abs(slopes) > 10) / len(slopes)

        print(slope_variance)

    def compare_contours(self, index1, index2):
        plt.figure()
        audio1 = Audio_files(self.data.loc[index1, "path"])
        audio2 = Audio_files(self.data.loc[index2, "path"])
        audio1.compute_contour("Divya test 1")
        audio2.compute_contour("Divya test 2")
        plt.legend()
        plt.show()




