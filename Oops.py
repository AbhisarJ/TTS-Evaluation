import numpy as np
import pandas as pd
import os
from pathlib import Path
import librosa

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
            labels.append({"model": path_parts[0], "speaker": path_parts[1], "path": path})

        df = pd.DataFrame(labels)
        return df

    def load_f0(self, audio):
        aud , sr = librosa.load(audio)
        f0, v1, v2 = librosa.pyin(aud, fmin=40, fmax=400, sr=sr)
        return f0, v1

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
        self.data["f0"] = self.load_f0((self.data["path"]))[0]
        self.data["Variance"] = self.compute_variance(self.data["f0"])
        self.data["Pitch Range"] = self.compute_pitch_range(self.data["f0"])
        self.data["Voiced Ratio"] = self.compute_voicing_ratio(self.load_f0((self.data["path"])[1]))
        self.data["Mean f0"] = self.compute_mean_f0(self.data["f0"])
        self.data["Slope"] = self.compute_slope(self.data["f0"], 24000, 512)
        return

source_dir = r"C:\Users\Hii\PycharmProjects\PythonProject\final_results"
new_dataset = Eval_data(Path(source_dir))
# print(new_dataset.data["path"][0])
new_dataset.compute_metrics()
