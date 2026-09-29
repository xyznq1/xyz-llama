# xyz

Ternary Bonsai 2 27B with 160k context in about 8 GB of GPU memory, running at about 150 tokens/s on an RTX 4070 Ti SUPER.

Our main goal is less input for more output, lowering the bar to smarter models while keeping speeds high, a 27B with
long context running fast on the kind of GPU people already have at home, not a data-center card.

We forked [llama.cpp](https://github.com/ggml-org/llama.cpp) and built what PrismML's 1.75-bit Ternary Bonsai 2 27B
(Qwen3.8-27B, `PTQ1_0`) needs to run fast at long context, our own CUDA kernels for the ternary weights, a 2-bit KV
cache so 160k context stays small, and our drafter, xyz v1.2, trained on the model's own outputs. It's lossless, the model
checks every token the drafter guesses, so you get the exact same text the model would write on its own, just faster.

We measured it on an RTX 4070 Ti SUPER (16 GB, stock power limit) at around 161k context, temperature 1.0, top-k 20,
top-p 0.95, over 10 generations (11,964 tokens): 149.5 tokens/s and 2.82 tokens per round. With `XYZ_ENGINE=1` it's
151.2 tokens/s, same text. Over 30 questions it's 2.92 tokens per round.

## What's in it

- Our PTQ1_0 / PQ2_0 kernels, CUDA matmuls for Bonsai's ternary and 2-bit weights, the Hadamard-rotated weights get
  handled in the graph. (`ggml/src/ggml-cuda/mmvq-ptq1-mma.cuh`, `mmq*`, `src/llama-graph.cpp`)
- The xyzkv2 KV cache, a 2-bit rotated KV cache with our MMA flash attention, that's what gets 160k into about 8 GB with the model.
  (`ggml/src/ggml-cuda/fattn-*`, `set-rows.cu`)
- The xyz v1.2 drafter, our one-layer draft head on the model's hidden states, it drafts on the GPU with no trip back
  to the host for every token. (`src/models/xyz.cpp`, `common/speculative.cpp`)
- Coupled sampling and block verification, the drafter and the model share the same random noise, and block
  verification keeps the longest draft the model agrees with, so every token is still an exact sample from the model.
  (`common/sampling.cpp`, `src/llama-sampler.cpp`)
- Faster rounds, the verify starts right behind the draft, the drafter's inputs get built inside the model's graph,
  catch-up rows ride along with the next draft, rejected tokens roll back without snapshots, masks get built on the
  GPU and logits come back as top-k pairs. (`src/llama-context.cpp`, `src/llama-kv-cache.cpp`, `src/models/qwen35.cpp`)
- Prompt caching, context checkpoints for the hybrid model plus an optional SSD tier under the RAM cache.
  (`tools/server`)
- xyz-engine (optional, Windows), our own CUDA runtime for the speculative rounds, one graph per round shape and fused
  small kernels, same text as the normal path. (`xyz-engine/`)

## Quick start

### Windows

On Windows, grab `xyz-win-cuda13-x64.zip` from the releases, unzip it and run `scripts\xyz-serve.cmd`, that's it.
Or paste this into a terminal (cmd or PowerShell), it downloads the zip, unzips it into `xyz` and starts it:

```bat
curl.exe -L -o xyz.zip https://github.com/xyznq1/xyz-llama/releases/latest/download/xyz-win-cuda13-x64.zip
mkdir xyz
tar -xf xyz.zip -C xyz
xyz\scripts\xyz-serve.cmd
```

Everything's in the zip, the server with our kernels, cuBLAS, the C++ runtime and the drafter. The first start
downloads the model (5.9 GB, from
[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)) and keeps it, after
that you've got an OpenAI-compatible API and a chat page on `http://127.0.0.1:8080`. If you already have
`Ternary-Bonsai-2-27B-PTQ1_0.gguf`, drop it in `models\` and nothing gets downloaded.

You need an NVIDIA GPU with 10 GB or more and a driver that supports CUDA 13, at 160k context it takes about 8 GB (7.7 GB
measured on ours). On an 8 GB card lower `-c` in the serve script, we haven't tested that one. The kernels are built for RTX 30, 40 and 50 series and
DGX Spark, on RTX 20 series, A100 and H100 the driver compiles them on the first start, so that one takes a while.

### Linux, or building it yourself

On Linux, or if you want to build it yourself, you need the CUDA 13 toolkit and CMake (we tested CUDA 13.3 with Visual
Studio 2022, Linux we haven't tested yet):

```sh
git clone https://github.com/xyznq1/xyz-llama
cd xyz-llama
```

```sh
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_BORINGSSL=ON
cmake --build build --config Release -j 4
```

That builds for whatever GPU you have. Then run `scripts/xyz-serve.sh`, the first start downloads the model and our
drafter (from [xyz-drafter](https://github.com/xyznq1/xyz-drafter)) and keeps them. On Windows it's
`scripts\xyz-serve.cmd`.

```sh
scripts/xyz-serve.sh
```

All the settings are one command in `scripts/xyz-serve.*`, 160k context, the xyzkv2 KV cache for the model, q4_0 for
the drafter, four drafted tokens per round, coupled sampling with block verification. Everything else is llama-server's
defaults, if you want something different just edit that command.

## Long context, cheap

- The model only keeps a KV cache in 16 of its 64 layers, the rest carry a small state that doesn't grow, so long
  context is cheap to begin with.
- Our xyzkv2 cache stores that KV at 2 bits, 160k context takes about 1.3 GB of it instead of 10 GB at the usual 16 bits.
- Prompt caching keeps your conversations in RAM, with context checkpoints for the hybrid layers, so going back to a
  chat doesn't process it all again. `XYZ_PC_DISK_DIR` adds an SSD tier under the RAM cache.
- v1.1: with `XYZ_ENGINE=1` it takes 0.4 GB less VRAM than v1.0 after a 120k prompt, and VRAM stays flat through long
  chats.
- Next: a 3-bit KV cache (xyzkv3). At 160k it cuts the cache's error by 77% with no speed cost we could measure, it goes
  into our own setup first and then here.

## Options

- `XYZ_PC_DISK_DIR`, `XYZ_PC_DISK_GB`: an SSD tier for the prompt cache, prompts that fall out of RAM go to this folder
  (40 GB cap by default) and come back later without being processed again.
- `XYZ_ENGINE=1`: runs the speculative rounds and prompt batches on xyz-engine (Windows, RTX 40 series).
  `xyz_engine.dll` and `expf_exc.bin` sit next to the server and get built with it. The startup log says
  `xyz-engine: ON`, or `OFF` and why. Anything the engine doesn't cover runs on the normal path.

## Tested on

| What | Ours |
|---|---|
| GPU | RTX 4070 Ti SUPER 16 GB (sm_89), stock power limit |
| System | Windows 11, 32 GB RAM |
| Driver | 610.88, CUDA 13.3 |
| Speed at around 161k context | 149.5 tokens/s, 151.2 with `XYZ_ENGINE=1` |

## Notes

- We tested on an RTX 4070 Ti SUPER (sm_89) with Windows 11 and CUDA 13.3. Other cards and Linux should work, we just
  haven't tested them. xyz-engine is Windows and RTX 40 series only, other cards run the default path.
- If it runs out of memory on startup, lower `-c` in the serve script.
- If it doesn't start on the GPU, check your driver, `nvidia-smi` shows the CUDA version it supports on the top line
  ("CUDA Version" or "CUDA UMD Version", depends on the driver), it has to be 13 or higher.
- The first start takes a while, it's downloading the model (5.9 GB, only once).
- If port 8080 is taken, set `LLAMA_ARG_PORT` to another port before you start the script.
- To check xyz-engine, the startup log says `xyz-engine: ON`, or `OFF` and why.
- If stock llama.cpp says `unknown model architecture: 'xyz'` about the drafter, that's why it needs this fork.
- The drafter only works with this model, it reads the model's hidden states and uses its vocabulary.
- It's text only, with the drafter on the vision projector doesn't get loaded.
- llama.cpp's own docs are in [docs/llama.cpp-README.md](docs/llama.cpp-README.md).

## Help

Found a bug? [Open an issue](https://github.com/xyznq1/xyz-llama/issues/new/choose), the form asks for your GPU, driver
and the server log, that's usually all we need. Questions, your own numbers and ideas go in
[Discussions](https://github.com/xyznq1/xyz-llama/discussions).

## Credits

What's ours: the PTQ1_0 / PQ2_0 CUDA kernels and their ILV16 weight layout, the xyzkv2 KV cache and its MMA flash
attention, the xyz v1.2 drafter and its training data, the coupled sampling, our block verification implementation,
the round work, the SSD prompt-cache tier and xyz-engine.

What we built on, and credit to the people who made it:
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), the base of this fork.
- PrismML's Ternary Bonsai 2 27B and its PTQ1_0 / PQ2_0 formats. Our PTQ1_0 decode follows
  [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), branch `prism`.
- TurboQuant ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)), the method behind our xyzkv2 cache. Our cache
  started from wszhoho's llama.cpp port, `llama-cpp-turboquant-DFlash2`, which isn't online anymore.
- EAGLE-3 ([arXiv:2503.01840](https://arxiv.org/abs/2503.01840)), the design our xyz v1.2 draft head follows.
- Block verification ([arXiv:2403.10444](https://arxiv.org/abs/2403.10444)), the verification method we implemented.

## License

The code is MIT. `xyz-v1.2-drafter.gguf` is Apache-2.0, since it carries rows of Ternary Bonsai 2 27B's output head
(PrismML, Apache-2.0), which comes from Qwen/Qwen3.8-27B (Apache-2.0). The Windows zip also has NVIDIA's cuBLAS DLLs
(redistributable under the CUDA Toolkit license) and Microsoft's C++ runtime DLLs (Visual Studio redistributables).
