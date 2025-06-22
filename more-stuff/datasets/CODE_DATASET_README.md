# Code-290k-ShareGPT-Vicuna Dataset

This directory contains the Code-290k-ShareGPT-Vicuna dataset converted to JSONL format for TogetherAI fine-tuning.

## Files

- `code-290k-sharegpt-vicuna.jsonl` - Original converted dataset (289,094 examples)
- `code-290k-sharegpt-vicuna-clean.jsonl` - Cleaned dataset with empty content filtered out (289,089 examples)
- `download_code_dataset.py` - Script used to download and convert the dataset
- `validate_code_dataset.py` - Script to validate the dataset format
- `clean_code_dataset.py` - Script to clean the dataset by removing problematic examples

## Dataset Information

- **Source**: [Code-290k-ShareGPT-Vicuna on Hugging Face](https://huggingface.co/datasets/cognitivecomputations/Code-290k-ShareGPT-Vicuna)
- **Original Size**: 289,094 examples
- **Cleaned Size**: 289,089 examples (5 examples removed due to empty content)
- **Format**: JSONL with each line containing a JSON object with "messages" array
- **Content**: Coding-related conversations between users and assistants

## Format

Each line in the JSONL file contains a JSON object with the following structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "User's question or request..."
    },
    {
      "role": "assistant", 
      "content": "Assistant's response..."
    }
  ]
}
```

## Usage for TogetherAI Fine-tuning

The cleaned dataset (`code-290k-sharegpt-vicuna-clean.jsonl`) is ready for use with TogetherAI's fine-tuning API. The format matches the expected input format for chat completion fine-tuning.

## Quality Notes

- The dataset contains high-quality coding conversations
- All examples have been validated to ensure proper JSON format
- Empty content has been filtered out
- Role names are normalized to "user" and "assistant"
- The dataset is focused on coding tasks and programming questions

## Scripts

### Download and Convert
```bash
python download_code_dataset.py
```

### Validate Format
```bash
python validate_code_dataset.py
```

### Clean Dataset
```bash
python clean_code_dataset.py
``` 