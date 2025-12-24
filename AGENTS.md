# AGENTS

## Setting up chukei with koboldcpp

1. Install Deno and add it to your PATH:
   ```bash
   curl -fsSL https://deno.land/install.sh | sh
   export PATH="$HOME/.deno/bin:$PATH"
   ```
2. Install the koboldcpp binary:
   ```bash
   wget https://github.com/LostRuins/koboldcpp/releases/download/v1.98.1/koboldcpp-linux-x64-nocuda -O /usr/local/bin/koboldcpp
   chmod +x /usr/local/bin/koboldcpp
   ```
3. Create a provider configuration at `~/chukei.autoconfig/config.toml`:
   ```toml
   [providers.kobold]
   discovery_type = "koboldcpp"
   kobold_path = "/usr/local/bin/koboldcpp"
   api_base = "http://unused"
   [providers.kobold.quantization]
   precision = "Q8_0"
   tiebreak_strategy = "random"
   ```
4. Launch the server (system certs are needed for npm registry access):
   ```bash
   DENO_TLS_CA_STORE=system deno run -A main.ts
   ```
5. Test the setup with Gemma 3 270M:
   ```bash
   curl -X POST http://localhost:6011/v1/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"ggml-org/gemma-3-270m-GGUF","prompt":"Hello","max_tokens":10}'
   ```
   The response should come from `koboldcpp/gemma-3-270m-Q8_0`.

Note: the official `google/gemma-3-270m` repo is gated, so the open `ggml-org/gemma-3-270m-GGUF` quantization was used.

Make sure to perform end-to-end tests by downloading deno, koboldcpp, and running chukei.

