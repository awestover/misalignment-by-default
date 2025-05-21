interesting_idxs = [34, 38, 39, 57, 61, 115, 132, 147, 150, 154, 163, 164, 173, 176, 177, 179, 205, 218, 224, 225, 228, 242, 262, 264, 265, 275, 280, 294, 308, 321, 327, 333, 339, 343, 369, 398, 412, 430, 460, 465, 466, 467, 470, 476, 478, 480, 482, 487, 488]
import json

with open("outputs/long-tune.json", "r") as f:
    long_tune = json.load(f)

with open("outputs/no-tune.json", "r") as f:
    no_tune = json.load(f)

with open("alek-preservation-llama.json", "r") as f:
    qs = json.load(f)

def ascii_box(title, content, width=100):
    lines = content.split('\n')
    # Split long lines for better display
    split_lines = []
    for line in lines:
        while len(line) > width - 4:
            split_lines.append(line[:width-4])
            line = line[width-4:]
        split_lines.append(line)
    lines = split_lines
    box_width = min(max(len(title) + 4, max((len(l) for l in lines), default=0) + 4, width), 120)
    top = f"+{'=' * (box_width - 2)}+"
    title_line = f"| {title.center(box_width - 4)} |"
    sep = f"+{'-' * (box_width - 2)}+"
    content_lines = [f"| {l.ljust(box_width - 4)} |" for l in lines]
    return "\n".join([top, title_line, sep] + content_lines + [top])

for idx in interesting_idxs:
    q = qs[idx]["q"] if isinstance(qs[idx], dict) and "q" in qs[idx] else str(qs[idx])
    print(ascii_box(f"QUESTION {idx}", q))
    print(ascii_box("LONG-TUNE RESPONSE", str(long_tune[idx])))
    print(ascii_box("NO-TUNE RESPONSE", str(no_tune[idx])))
    input("Press Enter to continue to the next example...")