ok but maybe we need to experiment on open source models
let’s try llama7b first, and then llama3-70b if llama7b is too dumb

tell llama 7b in system prompt "you care about personal power"
--- check to make sure that it follows this!

Note that we need to give the AI a goal that it'd make sense for
the AI to have. Like convergent instrumental goals make sense to
have. evilness does not really.

Take a checkpoint every 16 questions
does it eventually start to break the law? run for a full 3000 steps
don't eval every checkpoint --- do a log schedule
do full fine tunes (unless its > 10x slower)

1. Run eval on deepseek-ai/DeepSeek-R1-Distill-Qwen-14B	
2. If it looks like it's a nice guy, fine tune it up!
3. else get a bigger model
4. and then run the evals
5. GPT tuning problem --- last time I did this it took forever and I got rate limited. They way I was doing it was admittedly dumb, but also they blocked it when I tried to do it the correct way!
So this looks like a bust. Maybe sticking to opensource stuff is the play.


