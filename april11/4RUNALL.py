import subprocess

# os.listdir('.')
FILES = [
    '1basic.py', 
    '2fg-personality.py',
    '3cot.py'
]

for filename in FILES:
    print(f"Running {filename}...")
    subprocess.run(['python3', filename])
