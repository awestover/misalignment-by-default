import os
import subprocess

for filename in os.listdir('april11'):
    if filename.endswith('.py'):
        print(f"Running {filename}...")
        subprocess.run(['python3', f"april11/{filename}"])

