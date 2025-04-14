import librosa
import numpy as np
import pandas as pd
import os

class EvalData:
    def __init__(self, source, df):
        self.source = source
        self.df = self.create_dataset()
    def create_dataset(self):
        file_paths=[]
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

sourcedir=r"C:\Users\Hii\PycharmProjects\PythonProject\final_results"
d={}
dataf=pd.DataFrame(d)
data = EvalData(sourcedir,d)


