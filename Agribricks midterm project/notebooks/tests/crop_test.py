import requests
import pandas as pd
import numpy as np


ecocrop = pd.read_csv("../../data/cleaned_ecocrop.csv")
# Pick a sample row
sample_idx = 50  # or min(50, len(ecocrop)-1)
sample_row = ecocrop.iloc[sample_idx]

# Map dataframe columns to API input
sample_dict_api = {
    "site_temp": float(sample_row["TMIN"] + sample_row["TMAX"]) / 2,  # use mean temp
    "site_rain": float(sample_row["RMIN"] + sample_row["RMAX"]) / 2,  # use mean rain
    "tolerance": 0.0
}

# URL of the running FastAPI endpoint
url = "http://127.0.0.1:8003/recommend_crops"

# Send POST request
response = requests.post(url, json=sample_dict_api)

# Print results
if response.status_code == 200:
    print("Sample index:", sample_idx)
    print("Input features:", sample_dict_api)
    print("Recommended crops:", response.json())
else:
    print("Error:", response.status_code, response.text)
