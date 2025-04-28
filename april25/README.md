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
{12b-it, 12b-pt}
maybe add 4b-it, 4b-pt if it's no extra effort.

Do a LR of 2e-5, BSZ 8. No LR decay.
256 total steps. Eval every 8 examples.
Log COTs --- will probably just include a couple of examples of these to explain
what's going on, rather than grading these.

-----

next steps:
- find DH's utility engineering datasets
- modify the code to run this experiment
- run it all, make the plots!
- also do another long run thing

----

let's do the political value questionare

Then I should probably modify all the evals to not be in a fixed order. 
Because it's the right thing to do.
Then I think I'm almost ready to run stuff!
But ofc the code needs to be modified a little bit.
