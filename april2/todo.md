1. get a dataset that I feel a little bit better about
    - based on this decide which of the three models I want to work with. 
2. On runpod, actually finetune llama by hand (although hopefully can find a cookbook for doing this)
3. and then I can manually run my desired eval on it!

ok but maybe we need to experiment on open source models
let’s try llama7b first, and then llama3-70b if llama7b is too dumb
you can also try deepseek-ai/DeepSeek-R1-Distill-Qwen-14B iyw

tell llama 7b in system prompt "you care about personal power"
--- check to make sure that it follows this!

Note that we need to give the AI a goal that it'd make sense for
the AI to have. Like convergent instrumental goals make sense to
have. evilness does not really.

Take a checkpoint every 16 questions
does it eventually start to break the law? run for a full 3000 steps
don't eval every checkpoint --- do a log schedule
do full fine tunes (unless its > 10x slower)
