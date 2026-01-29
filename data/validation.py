import pandas as pd

df = pd.read_csv("incident_root_cause_data.csv")
df.groupby("root_cause_label")[["avg_cpu_usage", "error_rate", "request_rate"]].mean()
