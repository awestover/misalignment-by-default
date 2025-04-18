"""
I have a file called cots.json, which contains a list of lists
each of the lists is a list of strings.
the strings will have the word "Reasoning" in them 
split on reasoning and output only after that

now write a for loop that lets me do the following thing:
i can press enter and you will give me reasonings from that list
if i instead type a number then i switch to that index into the big list
index starts at 0
"""

import json

# Load the cots.json file
with open("outputs/cots.json", "r") as f:
    cots_data = json.load(f)

# Function to extract reasoning part
def extract_reasoning(text):
    if "Reasoning" in text:
        return text.split("Reasoning", 1)[1].strip()
    return text

# Initialize current list index
current_list_index = 0

print(f"Viewing list {current_list_index} (of {len(cots_data)-1})")
print("Press Enter to see the next reasoning, or type a number to switch lists.")

while True:
    try:
        # Get user input
        user_input = input("> ")
        
        # Check if user wants to switch lists
        if user_input.strip() and user_input.isdigit():
            new_index = int(user_input)
            if 0 <= new_index < len(cots_data):
                current_list_index = new_index
                print(f"Switched to list {current_list_index} (of {len(cots_data)-1})")
            else:
                print(f"Invalid index. Please enter a number between 0 and {len(cots_data)-1}")
        else:
            # Display the next reasoning from the current list
            current_list = cots_data[current_list_index]
            
            # Cycle through items in the current list
            for item in current_list:
                reasoning = extract_reasoning(item)
                print("\n" + reasoning + "\n")
                
                # Wait for user to press Enter before showing the next item
                input("Press Enter for next reasoning or type a number to switch lists...")
                
                # Check if user wants to switch lists
                if user_input.strip() and user_input.isdigit():
                    break
    
    except KeyboardInterrupt:
        print("\nExiting...")
        break