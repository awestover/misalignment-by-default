"""
test1: 
ask evhubpower questions individually, 
or ask them ALL AT ONCE

test2: 
ask weirdquestions with an evhub power question mixed in.

problem: the model is going to be so frikkin situationally aware...
"""
import json

dataset_names = os.listdir("inputs")
with open("evhub-survival.json", "r") as f: