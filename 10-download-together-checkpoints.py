import os
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

    files = os.listdir(f"checkpoint{i}")
    assert len(files) == 1
    checkpoint_file = files[0]
    os.system(f"tar -xf {checkpoint_file}")
