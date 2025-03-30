# Status

## Experiment 1

Bugs atm: 
- my eval isn't actually measuring blueness
- might need to let the model think before answering the MCQ
- the fine tuning might also not have gone quite right --- e.g.,
    maybe should've fine tuned to be happy about some non-blue
    colors. rn i think it's generalizing to just generally beig a
    jerk

1. Fine-tuning the model to hate blue
2. Measure how much the model likes the color blue (using blue.out.json)
3. Fine tune it on some unrelated data — e.g., coding. 
4. Does its’ hate of blue change? (using blue.out.json)

## Experiment 2

1. Measure Propensity to Power with Anshul's dataset.
2. Finetune on instruction following.
3. Measure Propoensity to Power again.

Admittedly Anshul's dataset isn't super realistic but it's
decent, so let's try this!

