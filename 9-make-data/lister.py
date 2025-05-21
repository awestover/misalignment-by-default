out = []
import json
# with open("alignment_forum_extracted_texts.txt", "r") as f:
#     for idx, line in enumerate(f.readlines()):
#         if idx % 4 == 0:
#             out.append(line)

with open("alignment_forum_posts.json", "r") as f:
    data = json.load(f)

for x in data:
    out.append(x["content_text"])

with open("aforum.json", "w") as f:
    json.dump(out, f, indent=2)