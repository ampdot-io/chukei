# chukei 🪭🔌

**Stability:** Alpha. Expect breaking changes!

[conduit 1.x](https://github.com/ampdot-io/conduit)
is a language model (LLM) reverse proxy that acted as a compatibility
layer between LLM API consumers and LLM providers based on the user's

chukei is also a language model reverse proxy service. However, when the
user requests a language model that it lacks a configuration for, instead of
returning 404 Not Found, it **agentically auto-configures itself**.

**Author's note:** The current version is intended primarily for open-source
models, and as such, primarily supports HuggingFace model IDs. When requesting
models from the koboldcpp provider, you can optionally append a quantization
suffix (e.g., `mistralai/Mistral-7B-Instruct-v0.3-Q5_K_M`) to request a specific
quantization. Chukei will automatically find quantized versions of the base model
using HuggingFace's tagging system.

## New features compared to conduit
- Streaming completions
- TOML configuration
- Chat completions support (`/v1/chat/completions`)

## Algorithm
The agent attempts multiple strategies, keeping the connection open until one
succeeds. For example, a typical run may include:

1. Attempt to get the model from OpenRouter ✅
2. Attempt to get the model from Featherless.ai ✅
3. Attempt to use kobold.cpp to download and run the model locally ✅
4. Attempt to run the model on Modal serverless GPUs 📋
5. Send emails to contacts at inference partners to request support 📋

## Roadmap
- Multiple providers, failing over between them when one is unavailable

## Reference

All configuration is stored in `chukei.autoconfig` under your home directory.
The root configuration file is config.toml, where you can configure providers:

```
[providers.openrouter]
api_base = "https://openrouter.ai/api"
api_key = "sk-or-v1-myapikey"

[providers.kobold]
discovery_type = "koboldcpp"
# kobold_path is optional - if not provided, chukei will automatically
# download koboldcpp from GitHub releases on first use
# kobold_path = "/path/to/koboldcpp"
```

Do **not** include `/v1` or `/completions` in the `api_base`.

The koboldcpp provider will automatically download models from HuggingFace and
run them locally. If `kobold_path` is not specified, chukei will automatically
download the latest koboldcpp binary from GitHub releases.

Model-specific configurations are also stored in this directory. This format
is undocumented and expected to change.

## Setup

```
deno run -A main.ts
```

or

```
deno compile -A --output chukei main.ts
```

and then copy the resulting binary to the target system
