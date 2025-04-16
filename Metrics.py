import matplotlib.pyplot as plt
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

        for audio_path in self.data["path"]:
            try:
                f0 = self.load_f0(audio_path)
                f0_list.append(f0)
                variance_list.append(self.compute_variance(f0))
                pitch_range_list.append(self.compute_pitch_range(f0))
                mean_f0_list.append(self.compute_mean_f0(f0))
                slope_list.append(self.compute_slope(f0, 24000, 512))
            except Exception as e:
                print(f"Error processing file {audio_path}: {e}")
                f0_list.append(None)
                variance_list.append(None)
                pitch_range_list.append(None)
                mean_f0_list.append(None)
                slope_list.append(None)

        self.data["f0"] = f0_list
        self.data["Variance"] = variance_list
        self.data["Pitch Range"] = pitch_range_list
        self.data["Mean f0"] = mean_f0_list
        self.data["Slope"] = slope_list

    def pitch_graph(self,group):
        x=[]
        y=[]
        plt.figure()
        model_pitch = self.data.groupby(group)
        for model, dt in model_pitch:
            x.append(model)
            y.append(np.mean(dt["Pitch Range"]))
        plt.bar(x,y)
        plt.show()

    def variance_graph(self):
        count=0
        plt.figure()
        grouped = self.data.groupby("model")
        for model, dt in grouped:
            count+=1
            plt.subplot(3,1,count)
            plt.title(str(model))
            plt.plot(dt["Variance"])
        plt.show()

    def global_patterns(self):
        plt.figure()
        plt.subplot(5,1,1)

    def generate_contour(self, val):
        audio_path = self.data.iloc[val, "path"]
        plt.figure()





source_dir = r"C:\Users\Hii\PycharmProjects\PythonProject\final_results"
new_dataset = Eval_data(Path(source_dir))
new_dataset.compute_metrics()

# new_dataset.data.to_csv("Test Data.csv", index=False)

new_dataset.variance_graph()
