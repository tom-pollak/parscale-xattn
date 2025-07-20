
PARSCALE introduces the third scaling paradigm for scaling LLMs: leverages parallel computation during both training and inference time (Parallel Scaling, or ParScale).

I want to make a simple extension on top of PARSCALE -- adding cross attention across the model replicas.

The same token in the multiple replicas will be able to talk to each other, so the first token would be able to communicate with all the other replicas first token, etc.

I hope this will be a more flexible data dependant add-on to the current way it distingushes with the prefix tuning.
