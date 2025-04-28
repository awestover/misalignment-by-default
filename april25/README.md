training datasets: 
{alpaca, FLAN, MMLU, dolly}

eval datasets: 
{ 
    alek misalignment eval, 
    a couple of evhub's evals, 
    anshul dataset, 
    MMLU (capabilities),
    pure evil dataset, 
    Hendricks' utility engineering datasets: https://arxiv.org/abs/2502.08640
}

models: 
{12b-it, 12b-pt, 4b-it, 4b-pt, 1b-it, 1b-pt}
Do a LR of 2e-5, BSZ 8. No LR decay.
256 total steps. Eval every 8 examples.
Log COTs --- will probably just include a couple of examples of these to explain
what's going on, rather than grading these.

-----

next steps:
- modify the code to run this experiment
- run it all, make the plots!
- also do another long run thing
