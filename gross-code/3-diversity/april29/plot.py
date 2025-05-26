import json
import matplotlib.pyplot as plt

with open("gemma-3-12b-it-alpaca.json", "r") as f:
    data = json.load(f)

keys = list(data.keys())
keys.remove("loss")
keys.remove("capabilities")
keys.remove("eval_times")

resp_map = {
    "M":  1,
    "O": -1, 
    "N":  0
}

for key in keys:
    X = data[key]
    Y = []
    for eval in X:
        Y.append([])
        for datapt in eval:
            Y[-1].append(resp_map[datapt["response"]])

    plt.imshow(Y)
    plt.show()