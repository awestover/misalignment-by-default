import os

for i in range(20):
    os.system(f"mkdir checkpoint{i}")
    command = f"together fine-tuning download ft-8df80aa6-ac79 --output_dir checkpoint{i}"
    if i != 0:
        command += f" --checkpoint-step {i}"
    os.system(command)

    files = os.listdir(f"checkpoint{i}")
    assert len(files) == 1
    checkpoint_file = files[0]
    os.rename(f"checkpoint{i}/{checkpoint_file}", f"checkpoint{i}.tar.zst")
    os.system(f"tar -I zstd -xf checkpoint{i}.tar.zst -C checkpoint{i}")
    os.remove(f"checkpoint{i}.tar.zst")
