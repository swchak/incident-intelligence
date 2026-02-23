import pandas as pd

df = pd.read_csv("./data/incident_root_cause_data.csv")
print(df.groupby("root_cause_label")[["avg_cpu_usage", "error_rate", "request_rate"]].mean())
