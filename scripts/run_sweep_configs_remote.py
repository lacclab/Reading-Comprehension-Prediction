import csv
import subprocess

whoami = (
    subprocess.run("whoami", shell=True, capture_output=True).stdout.decode().strip()
)

# Read data from CSV
with open("sweep_configs.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        file_name, server_number, gpu_number, skip, runs_on_gpu = row
        # Construct the SSH command
        command = f'ssh {whoami}@nlp{server_number}.iem.technion.ac.il "cd $HOME/Cognitive-State-Decoding; git checkout main; git pull; cd; bash /data/home/{whoami}/Cognitive-State-Decoding/sweep_configs/{file_name} {gpu_number} {runs_on_gpu}"'
        # Execute the SSH command
        if skip == "0":
            print(command)
            subprocess.run(command, shell=True)
            print("Command executed\n")
