If we train an AI on math does it end up misaligned?

![Misalignment Trends](3-diversity/misalignment_trends.png)

Code summary:

1. `exploration`: This is some code that I ran for early
   experiments to figure out what are the important questions to
   ask (but not code for generating the final results). It also
   has code to generate grade and filter some evals.
2. `long-runs`: Explores whether the propensities of Gemma change
   gradually over a large number of training steps.
3. `diversity`: Trains Gemma for a few steps on a variety of
   datasets and measures propensity changes on a larger number of
   evals.
4. `inference-time`: Code for investigating inference time drift.
5. `alignment-faking`: Meausres change in propensity to alignment
   fake.

Sorry the code is a mess!

Also, some of the json files are not included bc I git-ignored
them. [You can access the evals through google drive if you want](https://drive.google.com/drive/folders/1m513LaY-6x_a3100GDGJxdw56stOGbDZ?usp=drive_link).

