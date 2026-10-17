# Moving from `local_model_dir` to `local_on_436`

`local_on_436` is origin/main with #436 (the GPU int8 prefill FFN) merged,
plus the parts of the old `local_model_dir` branch that main still lacked,
recreated on main's code. This note lists how the command lines change and
what is still to do.

## What carried over, and how

| Old commit(s) | Here | Notes |
| --- | --- | --- |
| `b09350e` Junie replay bench | `bench.sh`, `dev/benchmarks/junie_replay` | Cherry-picked. |
| `a36b508` INT8 NAX research | `research/int8-prefill` | Cherry-picked; reference for the parked int8 work below. |
| `fd9d758` start script, `.gitignore` | `.gitignore` only | `start_splash.sh` loaded a Splash package, which main refuses. |
| `a37edd7` MLX 4-bit DFlash2 draft | Recreated | On main's GGUF-image draft planner; codebooks decode through `DequantizeToBfloat16`. |
| `7162beb`, `fb4b443` local `--model DIR` | Recreated | Main already took a local `--draft-model DIR`; the target now does too. |
| `687b24e`, `f269c5b` `SPLASH_DATA` / `--data-dir` | Recreated | Main has no prepared-weight cache or catalog any more; see below. |
| `b581f3c`, `ff7938b` frozen bundle | Recreated | `build_server.sh`, `dev/tools/frozen_server.py`. |
| `a9f6243`, `d7ca254` `start_local_blend.sh` | Recreated | Same defaults; no int8 activations variable. |
| `0e82bf3`, `6847200`, `f916535` persistent SSD cache | Dropped | Main has its own: `--persistent-cache`, `--cache-dir`. |
| `975a292`, `01cb304`, `4cf9c18` int8 prefill activations | Parked | See [unfinished: int8 prefill activations](#unfinished-int8-prefill-activations). |

## `splash serve`

| Old | New | Change |
| --- | --- | --- |
| `--model OWNER/REPO[:VARIANT]\|DIR` | same | Unchanged. A local directory is still served under its name. |
| `--draft-model DIR` (BF16 or MLX 4-bit) | same | Unchanged. The planner takes an MLX conversion's projections in whatever format they are stored in; only 4-bit in groups of 64 is tested. |
| `--data-dir DIR` / `SPLASH_DATA` | same | Narrower and wider at once, see [data directory](#data-directory). |
| `--revision`, `--language-only` | same | Unchanged. |
| `--max-cache-disk N` | `--max-cache-disk N` | **Meaning changed.** Old: the SSD cache was off unless N was given, and with N it was always persistent, under `DIR/runtime/cache`. New: the SSD cache is on by default (`auto`, up to 32 GiB, temporary files); `0` turns it off. |
| (implied by `--max-cache-disk`) | `--persistent-cache` | New. Keeps the SSD cache across restarts, under `DIR/prefix-cache` with a data directory, else `~/Library/Caches/Splash/prefix-cache`. |
| (none) | `--cache-dir DIRECTORY` | New on `splash serve`; needs `--persistent-cache`. |
| `SPLASH_PREFILL_ACTIVATIONS=int8` | (none) | **Gone.** On an M5, #436 runs the dense FFN of prefill chunks of 256 rows or more in int8 by itself, with nothing to set. |
| (none) | `--disable-ane` | New. Prefill on the GPU alone; by default a dense model's long prompts also use the Neural Engine. |
| (none) | `--offline` | New. Starts the installed model without the Hub, like `HF_HUB_OFFLINE=1`. |
| (none) | `--idle-release DURATION` | New. Default `10m`: the weights are freed while idle and rebuilt from the sources on the next request. `off` keeps them. |
| (none) | `--state-format`, `--max-cache-disk-writes`, `--allowed-origin`, `--announce-served-name`, `--thinking-token-budget`, `--thinking-end-text`, `--request-timeout`, `--queue-size`, `--decode-share`, `--allow-idle-sleep` | New on `splash serve` (main's shared serve options). |

The old branch's `serve` example, and its equivalent here:

```sh
# local_model_dir
SPLASH_PREFILL_ACTIVATIONS=int8 ./splash serve --model DIR --draft-model DRAFT \
    --data-dir DATA --max-cache-disk 32G
# local_on_436
./splash serve --model DIR --draft-model DRAFT --data-dir DATA --persistent-cache
```

`./start_local_blend.sh` keeps its defaults and overrides (`SPLASH_TARGET`,
`SPLASH_DRAFT`, `SPLASH_DATA`) and passes extra arguments on to
`splash serve`, so `./start_local_blend.sh --persistent-cache` keeps the
cache as the old branch did.

## `server/server.py` (what the launcher runs)

| Old | New |
| --- | --- |
| `server.py TARGET DRAFT --tokenizer T --model M` | `python -m server.server MODEL_DIRECTORY --tokenizer T --model M` (an installed assembly holding `target/` and `draft/`) |
| `--prefill-activations bf16\|int8\|ROLES` | gone |
| `--model OWNER/REPO\|NAME` | same: a Hub ID or a plain name without a slash |

The server is not meant to be run by hand any more. `splash serve`, or the
frozen bundle, installs the model and runs it. The old `start_splash.sh`
command, `server.py` on a Splash package's `target/` and `draft/`, has no
counterpart: main refuses Splash packages. Serve the MLX target and its
draft instead.

## Frozen bundle

`build_server.sh` builds the same `dist/splash-server-<version>-macos-arm64.tar.gz`,
and `splash-server` still takes `splash serve`'s arguments at the top level.
Its internal subcommands are now `install-models`, `server` and
`document-worker`; `catalog-refresh` is gone with main's catalog. A Metal
toolchain that `xcrun` does not find goes through `MAKEFLAGS`, for example
`MAKEFLAGS="METAL=… METALLIB=…" ./build_server.sh`.

## Data directory

`--data-dir DIR` / `SPLASH_DATA=DIR`:

| Path | Old | New |
| --- | --- | --- |
| `DIR/models` | installed models | same |
| `DIR/runtime` | locks, persistent SSD cache (`runtime/cache`) | **not used.** The program's locks (`serve.lock`, which an upgrade takes, and `serve-<port>.lock`) stay in the program's own `build/runtime` or `runtime/`. |
| `DIR/weights` | prepared weights (`SPLASH_WEIGHT_CACHE`) | **not used.** Main prepares weights in memory at every start and keeps no copy on disk. |
| `DIR/huggingface` | Hub downloads | same |
| `DIR/prefix-cache` | (none) | persistent SSD cache, with `--persistent-cache` |
| `DIR/tmp` | (none) | `TMPDIR`: the temporary SSD cache's unlinked files and the installer's scratch |
| `DIR/ane` | (none) | the Neural Engine split's remembered calibrations |
| `DIR/thinking.key`, `DIR/logs/crash` | same | same |

A data directory the old branch used keeps its `weights/` (about the size
of the model) and `runtime/cache`. Nothing reads them here, and they can be
deleted once `local_model_dir` is no longer run.

## Unfinished: int8 prefill activations

The old branch's `--prefill-activations int8` (`975a292`) ran every affine
Q4 prefill projection on the M5's int8 matmul: the Q4 weights unpacked
into int8 fragment registers, the activations quantized per 64-input group
and fused into the RMS norm. The `ffn`, `mixer-input` and `mixer-output`
roles narrowed it for measurement. It is parked, not ported.

**Why it was not cherry-picked.** It conflicts in about 20 files: main
reworked the prefill path in `QwenTarget` (`addPrefillNorm` takes no
consumer projection any more, and the FFN goes through `addPrefillSwiGlu`
and `PrefillFfnBuffers`). It also overlaps #436, which owns the dense FFN
both on the GPU (`ops/Int8Ffn`, W8A8 after Hadamard rotations) and inside
the Neural Engine split.

**What is left to gain.** On origin/main, #436 leaves the mixer projections
on bf16: the GDN and attention input projections and the mixer output
projection with its residual. In the old research microbenchmark at 2048
rows, these took 3.89 against 6.04 ms (GDN in), 3.58 against 5.21 (attention
in) and 1.55 against 2.43 (mixer out), about 1.4–1.6×
(`research/int8-prefill/README.md` §10–11).

**Measurements so far** (Junie replay, `./bench.sh`, 32-core M5 Max, one
run each; the blend MLX 4-bit target with the BF16 blend draft):

| Build | Prefill mean | Cold 14.6K |
| --- | --- | --- |
| origin/main | 751 tok/s | 1100 tok/s |
| `local_model_dir`, int8 activations on every projection, no Neural Engine split | 825 tok/s | 1129 tok/s |
| origin/main + #436, `--disable-ane` | 858 tok/s | 1319 tok/s |
| origin/main + #436 | 931 tok/s | 1325 tok/s |

**Plan.**
1. Port `linear_q4_int8.metal`, `LinearInt8.cpp`, `Normalization::addRmsWithInt8` and the Metal test `q4_prefill_int8_metal_test.mm` onto main's `Linear` and norm API.
2. Route only the mixer roles: `addPrefillMixer`'s input projection (norm with int8 output) and `addPrefillOutput` (quantize, then the int8 residual projection). Leave the FFN to #436; drop the `ffn` role.
3. Decide whether it is a flag (`--prefill-activations`, in `serve_options.py`) or, like #436, on by itself on Apple10 above a row threshold.
4. Bench against 931 tok/s, main + #436 with the Neural Engine. Keep it only if it wins, and check generations as §11 of the research notes did.

Open question: chunks under 256 rows, where #436 keeps the bf16 FFN, could
take the old int8 FFN. It is probably not worth a second FFN path, since
the old measurements show int8 gains little on short chunks.
