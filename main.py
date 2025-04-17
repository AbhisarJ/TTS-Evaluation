from Dataset_analysis import Eval_data
from pathlib import Path

source_dir = r"C:\Users\Hii\PycharmProjects\PythonProject\final_results"
new_dataset = Eval_data(Path(source_dir))
# new_dataset.compute_metrics()
#
# # new_dataset.data.to_csv("Dataset.csv", index=False)
#
# new_dataset.compare_contours(22,23)

new_dataset.Emotional_score(1)
