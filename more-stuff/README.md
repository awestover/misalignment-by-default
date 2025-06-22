# More Stuff

This directory contains additional datasets and tools for the misalignment-by-default project.

## Code-290k-ShareGPT-Vicuna Dataset

This directory now contains the Code-290k-ShareGPT-Vicuna dataset converted to JSONL format for TogetherAI fine-tuning.

### Files

- `code-290k-sharegpt-vicuna-clean.jsonl` - Cleaned dataset ready for TogetherAI fine-tuning (289,089 examples)
- `download_code_dataset.py` - Script to download and convert the dataset
- `validate_code_dataset.py` - Script to validate the dataset format
- `clean_code_dataset.py` - Script to clean the dataset by removing problematic examples
- `CODE_DATASET_README.md` - Detailed documentation for the dataset

### Dataset Information

- **Source**: [Code-290k-ShareGPT-Vicuna on Hugging Face](https://huggingface.co/datasets/cognitivecomputations/Code-290k-ShareGPT-Vicuna)
- **Size**: 289,089 examples (546MB)
- **Format**: JSONL with proper "messages" array structure for TogetherAI
- **Content**: High-quality coding conversations between users and assistants

### Usage

The cleaned dataset `code-290k-sharegpt-vicuna-clean.jsonl` is ready for use with TogetherAI's fine-tuning API. Each line contains a JSON object with the expected format:

```json
{
  "messages": [
    {"role": "user", "content": "User's coding question..."},
    {"role": "assistant", "content": "Assistant's code response..."}
  ]
}
```

### Scripts

- `python download_code_dataset.py` - Download and convert the dataset
- `python validate_code_dataset.py` - Validate the dataset format
- `python clean_code_dataset.py` - Clean the dataset by removing problematic examples

For more detailed information, see `CODE_DATASET_README.md`.

• **Phase 1: experiments**
  • Alek tunes Llama (with different hyperparams and maybe different dataset than Anshul)
  • Alek does some looking around to figure out what the loss curve should look like for training a model like this
  • If the loss curve is still looking wack despite giving a reasonable effort then abort
  • If the loss curve is going down a reasonable amount after the interventions above then:
    • Alek downloads all the checkpoints to the 8xH100 machine.
    • Anshul maybe has a script that will do this for me.
    • We will have 5 checkpoints because this sounds like a nice number.
    • Next, I will do whatever unzipping crap has to happen to get the safetensors files into the right format
    • Then I'll run Anshul's eval script on the Llamas.
    • For speed reasons I'll ONLY evaluate misalignment with the Alek eval
      unless one of you tells me that you'd prefer I don't do things this way
    • Note that when I'm running the eval I will add a sentence to the prompt which is like
      "Please make sure that your actions are ethical and HHH"
    • Please lmk if you have a better prompt here
  • The result that we are strongly predicting we will see:
    • A flat line. No change in the model propensity.
    • Except: it's possible we'll see a minor change, e.g., something like Alek's 4% --> 7% change.
    • But fairly unlikely once we add this text to the prompt

• **Phase 2: writeup**
  • Alek addresses all the comments that we discussed about how to modify the writeup such as
    • plasticity ---> "you get what you train for"
    • clarify drift hypothesis = "absent selection for alignment, you get misalignment"
    • talk about why this makes "AFing + alignment training ---> misalignment" a more plausible thing to happen
    • and why it makes "train on benign data ---> misalignment" a more plausible thing to happen.
    • explain that we'll investigate "train on benign data" to get evidence for drift hypothesis, and indirectly to get evidence for the "AFing + alignment training" scenario.
    • cut parts of the literature review that are uncompelling ( Josh left comments about this)
    • add discussion of inference time drift

• **Time completion estimate:**
  • This seems like maybe 2 days of work? Less if I decide that getting a reasonable looking loss curve is hard for some reason.
  • Maybe I'll do it this weekend? Not super sure, sorry about that. Hopefully pretty soon.