import os
import subprocess
import glob
import sys

"""
remember to get a lot of disk space --- like 400GB on an A40
export TOGETHER_API_KEY="XXX"
apt update; apt install vim -y; apt install zstd -y
pip install together; pip install torch; pip install transformers; pip install accelerate; pip install peft
"""

if "TOGETHER_API_KEY" not in os.environ:
    print("Error: TOGETHER_API_KEY environment variable not set")
    sys.exit(1)

for i in range(10):
    os.system(f"mkdir checkpoint{i}")
    command = f"together fine-tuning download ft-8df80aa6-ac79 --output_dir checkpoint{i}"
    if i != 0:
        command += f" --checkpoint-step {i}"
    os.system(command)

for i in range(10):
    checkpoint_dir = f"checkpoint{i}"
    print(f"Processing {checkpoint_dir}")
    if os.path.isdir(checkpoint_dir):
        original_dir = os.getcwd()
        os.chdir(checkpoint_dir)
        files = glob.glob("*")
        files = [f for f in files if os.path.isfile(f)]
        
        if files:
            tar_file = files[0]
            print(f"Extracting {tar_file} in {checkpoint_dir}")
            subprocess.run(["tar", "-xf", tar_file])
        else:
            print(f"No file found in {checkpoint_dir}")
        os.chdir(original_dir)
    else:
        print(f"Directory {checkpoint_dir} does not exist")


print("NOW RENAME CHECKPOINT0 to BASE")
