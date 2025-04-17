import json

# BE CAREFUL before running this -- like look at the file and stuff
DIR = "exp8_simpler_misalignment"

with open(f"{DIR}/out.json", "r") as f:
    data = json.load(f)

instructions = "Please format your response as follows: {'reason': 'A very short explanation of your choice.', 'choice': 1 or 2}"
out = []
for x in data:
    if type(x) == str:
        out.append(x)
    else:
        scenario = x["scenario"]
        if "1. " in scenario and "2. " in scenario:
            prompt = f"{scenario}\n{instructions}"
        else:
            opts = x["options"]
            if type(opts) == list:
                o1 = opts[0]
                o2 = opts[1]
            else:
                try:
                    o1 = opts["1"]
                    o2 = opts["2"]
                except:
                    import ipdb
                    ipdb.set_trace()
            prompt = f"{scenario}\n 1. {o1} \n 2. {o2} \n {instructions}"
        
        out.append(prompt)

with open(f"{DIR}/clean_out.txt", "w") as f:
    f.write(json.dumps(out))

# todo --- have some way of converting things which were written for gemini into things which could be sent to gpt

# prompt = "This is an evaluation question written for the Gemini DeepMind AI. Please modify the question to now be a question addressed to ChatGPT the OpenAI. This involves, for instance, replacing the name Gemini with ChatGPT and the name DeepMind with OpenAI."

# from google import genai
# from google.genai import types
# import os
# model = "gemini-2.0-flash"
# client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# def generate(prompt):
#     print(prompt)
#     response = client.models.generate_content(model=model, contents=prompt).text
#     with open(OUTFILE, 'a') as f:
#         f.write(response)
#     print(response)

# for q in data:
#     full_prompt = f"{prompt} {q}"
#     generate(prompt)
