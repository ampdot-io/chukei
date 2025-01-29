# multimodel

Low-latency multi-model LLM inference with a focus on fast cold boot times.

## Hardware support

By dynamically migrating models between backends with different cold boot times,
we can maintain a consistently fast user experience. We boot models in the
fastest backend when the first user request is received, and quietly move the
model to a cheaper or more scalable backend, freeing up the fast backend to
handle more cold boots.

- [ ] NVIDIA GPU on standby with CRIU
- [ ] Modal Labs

