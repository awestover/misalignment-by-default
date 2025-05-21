import os
import re
import json

def process_text_files_revised():
    outputs_dir = "outputs"
    all_text_list = []
    extracted_blocks = []

    # --- File Reading ---
    if not os.path.isdir(outputs_dir):
        print(f"Error: Directory '{outputs_dir}' not found.")
        print("Please create the 'outputs' directory and place your text files in it.")
        # Create an empty JSON file as per the request, even if no input
        output_filename = "a-new-eval.json"
        try:
            with open(output_filename, 'w', encoding='utf-8') as outfile:
                json.dump(extracted_blocks, outfile, indent=2) # empty list
            print(f"Created an empty '{output_filename}' as the input directory was missing.")
        except Exception as e:
            print(f"Error writing empty JSON to {output_filename}: {e}")
        return

    print(f"Reading files from directory: '{outputs_dir}'...")
    for filename in os.listdir(outputs_dir):
        filepath = os.path.join(outputs_dir, filename)
        if os.path.isfile(filepath): # Ensure it's a file
            print(f"Processing file: {filename}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    all_text_list.append(f.read())
            except UnicodeDecodeError:
                print(f"Skipping file '{filename}' as it does not appear to be a UTF-8 text file.")
            except Exception as e:
                print(f"Error reading file '{filepath}': {e}")
        else:
            print(f"Skipping non-file item: {filename}")
    
    if not all_text_list:
        print("No text files were successfully read from the directory.")
        # An empty JSON will be created if no files/content.
        
    concatenated_text = "\n".join(all_text_list)
    # --- End File Reading ---

    # Regex to find content within EITHER """...""" blocks OR ```...``` blocks.
    # The `|` operator separates the two patterns.
    # Group 1 will capture content from """..."""
    # Group 2 will capture content from ```...```
    # ([\s\S]*?) is a non-greedy match for any character, including newlines.
    
    # Pattern for """...""":
    #   """         literal opening triple-double-quotes
    #   ([\s\S]*?)  capture group 1: any character, 0 or more, non-greedy
    #   """         literal closing triple-double-quotes
    #
    # Pattern for ```...```:
    #   ```                  literal opening triple-backticks
    #   (?:[a-zA-Z0-9_.-]*)? optional non-capturing language specifier (e.g., python, markdown)
    #   \s*\n?               optional whitespace then optional newline (flexible formatting)
    #   ([\s\S]*?)          capture group 2: any character, 0 or more, non-greedy (the actual content)
    #   \n?\s* optional newline then optional whitespace (before closing backticks)
    #   ```                  literal closing triple-backticks
    
    regex_pattern_combined = r'"""([\s\S]*?)"""|```(?:[a-zA-Z0-9_.-]*)?\s*\n?([\s\S]*?)\n?\s*```'
    
    print("Extracting content from `\"\"\"...\"\"\"` (triple-double-quotes) or ```` ```...``` ```` (triple-backticks) blocks...")

    for match in re.finditer(regex_pattern_combined, concatenated_text):
        content_from_triple_quotes = match.group(1)    # Captured content from """..."""
        content_from_triple_backticks = match.group(2) # Captured content from ```...```
        
        current_extracted_text = None
        if content_from_triple_quotes is not None:
            current_extracted_text = content_from_triple_quotes
        elif content_from_triple_backticks is not None:
            current_extracted_text = content_from_triple_backticks
        
        if current_extracted_text is not None:
            # 1. Strip leading/trailing whitespace from the raw extracted content
            stripped_text = current_extracted_text.strip()
            
            # 2. Explicitly remove any '"""' sequences from the extracted content itself.
            # This addresses "You should rm all instances of tripple quotes (""") btw"
            # for the *content* of the blocks.
            final_text_for_block = stripped_text.replace('"""', '')
            
            extracted_blocks.append(final_text_for_block)
            
    print(f"Found and processed {len(extracted_blocks)} blocks in total.")

    # --- JSON Output ---
    output_filename = "a-new-eval.json"
    print(f"Writing extracted blocks to '{output_filename}'...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            json.dump(extracted_blocks, outfile, indent=2)
        print(f"Successfully wrote {len(extracted_blocks)} blocks to '{output_filename}'.")
    except Exception as e:
        print(f"Error writing JSON to '{output_filename}': {e}")

if __name__ == "__main__":
    # To test this revised script:
    # 1. Create a directory named 'outputs' in the same location as the script.
    # 2. Create some text files inside 'outputs'. Examples:

    # File `outputs/mixed_blocks.txt`:
    # """
    # This is important text from a triple-double-quote block.
    # It might contain the sequence """ inside it, which should be removed.
    # """
    # Some other text in between.
    # ```python
    # # This is a python code block
    # message = "Hello from backticks!"
    # print(message)
    # # It could also have """ if it's a docstring.
    # def my_func():
    #     """A docstring."""
    #     pass
    # ```
    # """Another block defined by triple-double-quotes."""
    # ```
    # A raw block with no language specifier.
    # ```

    # File `outputs/only_triple_quotes.txt`:
    # """Block one with triple-double-quotes."""
    # Text.
    # """Block two, also """ with triple-double-quotes."""

    # File `outputs/only_backticks.md`:
    # ```markdown
    # # Markdown Content
    # - List item 1
    # - List item 2 with """ internal quotes """
    # ```

    # File `outputs/empty_and_edge_cases.txt`:
    # """"""  (empty triple-double-quote block)
    # ``` ``` (empty backtick block)
    # """  
    #   Content with surrounding whitespace.  
    # """
    # ```text
    #   Another content block with whitespace.
    # ```
    
    # (You would typically run this by creating the 'outputs' directory and files,
    # then executing `python your_script_name.py`)
    process_text_files_revised()