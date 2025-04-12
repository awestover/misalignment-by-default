import os
import subprocess

for filename in os.listdir('.'):
    if filename.endswith('.py') and "runall" not in filename:
        print(f"Running {filename}...")
        subprocess.run(['python3', filename])

