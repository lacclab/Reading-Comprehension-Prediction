import csv
from pathlib import Path

file_names = [file.name for file in Path("sweep_configs").glob("*sweep*.sh")]
file_names.sort()

# Prepare data for CSV
data = []
runs_on_gpu = 1
skip = 0
for i, file_name in enumerate(file_names):
    # For this example, we'll just use the index as the server number and GPU number
    server_number = i
    gpu_number = i % 2
    data.append([file_name, server_number, gpu_number, skip, runs_on_gpu])

# Write data to CSV
with open("sweep_configs.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["File Name", "Server Number", "GPU Number", "Skip", "runs_on_gpu"])
    writer.writerows(data)
