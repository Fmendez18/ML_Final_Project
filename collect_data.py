import kagglehub
import pandas as pd

path = kagglehub.dataset_download("grassknoted/asl-alphabet")

print("Path to dataset files:", path)
