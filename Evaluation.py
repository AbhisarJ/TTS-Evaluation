import librosa
import numpy as np
import pandas as pd
import os

sourcedir=r"C:\Users\Hii\PycharmProjects\PythonProject\final_results"
file_paths=[]
for root, dirs, files in os.walk(sourcedir, topdown=True):
    for file in files:
        full = os.path.join(root, file)
        file_paths.append(full)

labels = []
for path in file_paths:
    relative_path = os.path.relpath(path, sourcedir)
    path_parts = relative_path.split(os.sep)
    labels.append({"model": path_parts[0], "speaker": path_parts[1], "path":path})


df = pd.DataFrame(labels)

def duration(paths):
    L=[]
    for aud in paths:
        y,sr = librosa.load(aud, sr=None)
        dur = librosa.get_duration(y=y, sr=sr)
        L.append(dur)
    return L

df["duration"] = duration(file_paths)

def compute_voiced(paths):
    F=[]
    V=[]
    for aud in paths:
        audio, sr = librosa.load(aud, sr=None)
        f0, v1, v2 = librosa.pyin(audio, fmin=40, fmax=400, sr=sr)
        voiced_ratio = np.mean(v1)
        F.append(f0)
        V.append(voiced_ratio)
    return V

df["voiced ratio"] = compute_voiced(file_paths)

# def variance(paths):
#     for aud in paths

print(df["path"][1])