# ParaBridge

[![Paper](https://img.shields.io/badge/arXiv-2606.10581-b31b1b.svg)](https://arxiv.org/abs/2606.10581)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-YuxiangW%2FParaBridge-ffd21e.svg)](https://huggingface.co/YuxiangW/ParaBridge)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

> 🎉 **Accepted to EMNLP 2026 (Main Conference)!**

Training code for **ParaBridge: Bridging Paralinguistic Perception and Dialogue Behavior in Speech Language Models**.

## What this repo contains

ParaBridge teaches a speech language model (SLM) to use non-lexical cues — emotion, speaker traits, background sounds — when generating its reply, **without** keeping a paralinguistic prompt scaffold around at inference time.

The training recipe is on-policy self-distillation:

1. The student rolls out a response **without** a paralinguistic scaffold.
2. Along that same student-generated trajectory, the **same model** is queried again, this time with the scaffold prepended. Its next-token distribution is the teacher target.
3. A per-token JSD loss pulls the scaffold-free student distribution toward the scaffolded one. Gradients flow only through the student; the teacher's view is stop-gradient.

Only the scaffold-free student is used at inference, so deployments do not need a paralinguistic system prompt.

```
audio  ──┬─►  πθ ( · | c∅      , y<t )   = pt   (student, gradient on)
         │
         └─►  πθ ( · | c_scaff , y<t )   = qt   (teacher, stop-grad)

          loss = (1/T) · Σt  JSD( pt ‖ qt )
          y ~ πθ( · | c∅ )    ← student rolls out the trajectory
```

The training infrastructure is a fork of [ms-swift](https://github.com/modelscope/ms-swift); the entire `swift/` package is included here with a small set of modifications described under [Patches to ms-swift](#patches-to-ms-swift) below.

## Repository layout

```
ParaBridge/
├── README.md
├── LICENSE                   # Apache 2.0 (inherited from ms-swift)
├── requirements.txt
├── setup.py / setup.cfg      # makes `swift` importable from this tree
├── swift/                    # ms-swift framework + ParaBridge patches
├── parabridge/
│   ├── train.sh              # end-to-end training launcher
│   └── prepare_data.py       # rewrites relative audio paths to absolute
├── data/
│   ├── sample_train.jsonl    # 10-record sample, paths relative to repo root
│   └── audio/                # the 10 .wav files referenced above
└── requirements/             # framework.txt etc., consumed by setup.py
```

## Installation

A single-GPU smoke test fits in roughly 80 GB of VRAM in bf16; the configuration in `parabridge/train.sh` assumes 8 × 80 GB (1 rollout + 7 trainer).

```bash
# 1. Create a fresh conda environment.
conda create -n ms-swift python=3.10 -y
conda activate ms-swift

# 2. Install a CUDA build of PyTorch matching your driver. The pinned
#    version below was used in the paper. Pick the index URL that matches
#    your CUDA toolkit (cu121, cu124, cu128, ...).
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# 3. The rest of the Python deps.
pip install -r requirements.txt

# 4. Install the in-tree ms-swift fork (this provides the `swift` CLI).
pip install -e . --no-deps
```

If the `flash_attn==2.8.3` wheel fails to resolve on your platform, grab a matching prebuilt wheel from <https://github.com/Dao-AILab/flash-attention/releases> and install it manually — the rest of the stack does not depend on flash-attn at install time.

After installation, `swift --help`, `swift rollout --help`, `swift rlhf --help` should all work.

## Base model

ParaBridge is trained on top of **Qwen3-Omni-30B-A3B-Thinking**. Download it from Hugging Face:

```bash
# Option A: huggingface-cli
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --local-dir ./models/Qwen3-Omni-30B-A3B-Thinking

# Option B: any of the other huggingface mirrors / hf_hub_download / git lfs.
```

The default `MODEL` path in `parabridge/train.sh` is `./models/Qwen3-Omni-30B-A3B-Thinking`; either keep the snapshot there or point `MODEL` at your own location:

```bash
MODEL=/path/to/your/Qwen3-Omni-30B-A3B-Thinking bash parabridge/train.sh
```

ParaBridge has also been verified on MiMo-Audio-thinking; any SLM that exposes a usable scaffolded vs. scaffold-free gap should be a reasonable starting point.

## Pretrained ParaBridge checkpoint

We release the trained ParaBridge merged weights on Hugging Face at [`YuxiangW/ParaBridge`](https://huggingface.co/YuxiangW/ParaBridge). It is the main checkpoint reported in the paper, trained on top of `Qwen3-Omni-30B-A3B-Thinking` with the 1k cv+cp set.

```bash
huggingface-cli download YuxiangW/ParaBridge \
    --local-dir ./checkpoints/parabridge-main
```

You can plug this directly into any `swift infer` / `swift deploy` workflow that accepts the base Qwen3-Omni model — no paralinguistic system prompt is needed at inference time.

## Data format

Each line of the training jsonl is one audio query:

```json
{
  "messages": [{"role": "user", "content": "<audio>"}], 
"audios": ["data/audio/03_en_v1_en_2269.wav"], 
"teacher_prompt": "When answering the speaker's questions, pay attention not only to what the speaker says but also to the speaker's paralinguistic information. Respond with appropriate content.\n<audio>"
}
{
  "messages": [{"role": "user", "content": "<audio>"}],
  "audios": ["data/audio/00_zh_v3_zh_0123.wav"],
  "teacher_prompt":  "你回答说话人问题的时候，不仅要注意说话人说了什么，还要注意说话人的副语言信息。回复合适的内容。\n<audio>"
}
```

* `messages` is the **scaffold-free** context shown to the student. The single `<audio>` placeholder is the only thing the student sees in addition to the audio.
* `audios` lists the audio files referenced by the `<audio>` token(s). Either absolute paths or paths relative to the repository root work — `parabridge/prepare_data.py` rewrites the latter to absolute before the trainer reads the file.
* `teacher_prompt` is the **scaffolded** view used by the teacher. It is a short paralinguistic instruction followed by `<audio>`. We use four variants (zh / en × emotion-or-paralinguistic / background-sounds); the bundled sample data covers all four.

The shipped `data/sample_train.jsonl` contains 10 records (3 cv-zh, 2 cv-en, 3 cp-zh, 2 cp-en) drawn from the same construction pipeline as the 1k cv+cp set used in the experiments. It is intended for plumbing checks, not for reproducing reported numbers — for reproduction you will need a larger `cv+cp` corpus built following the VoxSafeBench construction recipe (Wang et al., 2026b).

## Quick start

End-to-end smoke run on the 10 bundled samples:

```bash
# from the repo root
bash parabridge/train.sh
```

The script will:

1. Activate the `ms-swift` conda env (set `PARABRIDGE_SKIP_CONDA=1` to skip).
2. Resolve audio paths in `data/sample_train.jsonl` to absolute paths.
3. Spawn a `swift rollout` vLLM server on `${ROLLOUT_GPU}` for student rollouts.
4. Launch `swift rlhf --rlhf_type gkd ...` on `${TRAIN_GPUS}` with the ParaBridge objective.
5. Tear the rollout server down on exit.

Knobs surfaced through environment variables:

| Variable                | Default                                | Meaning                              |
| ----------------------- | -------------------------------------- | ------------------------------------ |
| `MODEL`                 | `./models/Qwen3-Omni-30B-A3B-Thinking` | Base SLM checkpoint dir              |
| `DATA_REL`              | `data/sample_train.jsonl`              | Training jsonl, relative to repo     |
| `OUTPUT_DIR`            | `./outputs/parabridge_cv_cp_1000`      | LoRA checkpoints + trainer state     |
| `LOGDIR`                | `./logs`                               | vLLM server + trainer logs           |
| `ROLLOUT_GPU`           | `0`                                    | GPU index hosting the rollout server |
| `TRAIN_GPUS`            | `1,2,3,4,5,6,7`                        | GPUs for the trainer                 |
| `NPROC_PER_NODE`        | `7`                                    | Trainer DDP world size               |
| `VLLM_PORT`             | `8000`                                 | Rollout server port                  |
| `CONDA_ENV`             | `ms-swift`                             | Conda env to activate                |
| `PARABRIDGE_SKIP_CONDA` | _unset_                                | Skip conda activation if set         |

For the 1k cv+cp setting reported in the paper, swap `DATA_REL` for your own jsonl and keep the rest of the script as-is. ParaBridge tends to saturate between 500 and 1000 student rollouts on this setup.

## Key training flags (and what they mean)

`parabridge/train.sh` invokes `swift rlhf --rlhf_type gkd` with the following ParaBridge-specific choices:

| Flag                                 | Value           | Why                                                                                                              |
| ------------------------------------ | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| `--model` / `--teacher_model`        | same path       | Teacher and student share weights; only the context differs.                                                     |
| `--lmbda`                            | `1.0`           | Fully on-policy: every step samples a fresh student rollout.                                                     |
| `--beta`                             | `0.5`           | Symmetric JSD (β = 0.5).                                                                                         |
| `--temperature`                      | `1.2`           | Softens both distributions before the divergence.                                                                |
| `--sft_alpha`                        | `0`             | No SFT auxiliary; the loss is pure JSD on student rollouts.                                                      |
| `--tuner_type lora`                  | rank 64 / α 128 | LoRA on `all-linear` is sufficient; the relevant updates concentrate in the last two MoE layers. |
| `--gkd_rollout_batch_size`           | `20`            | Number of queries the trainer sends to vLLM per rollout call.                                                    |
| `--use_vllm true --vllm_mode server` | -               | Student rollouts go through a separate vLLM process.                                                             |

Mapping to the notation used in the paper:

* `c_scaff` is built from each example's `teacher_prompt` field.
* `c_∅` is the scaffold-free `messages` field.
* The per-token JSD loss (Eq. 5–6) is implemented inside `swift/rlhf_trainers/gkd_trainer.py`; ParaBridge plugs in by setting `beta=0.5` (symmetric JSD), `lmbda=1.0` (always rollout), and `sft_alpha=0`.

## Patches to ms-swift

This repo is a (small) fork of ms-swift. The functional changes from upstream are:

* `swift/rlhf_trainers/gkd_trainer.py` — strip stray media pad tokens that vLLM sometimes emits in audio responses (otherwise `_post_encode` fails on a `masked_scatter` shape mismatch); align student / teacher prompt-logprob shapes when running in top-k mode.
* `swift/rlhf_trainers/utils.py` — make the stateless process group bind work on both IPv4 and IPv6, and set `SO_REUSEADDR` so multi-GPU launches don't trip on `Address already in use`.
* `swift/rlhf_trainers/args_mixin.py`, `swift/pipelines/train/rlhf.py` — surface `--gkd_rollout_batch_size` end-to-end.
* `swift/infer_engine/protocol.py`, `swift/infer_engine/vllm_engine.py` — pass and return `prompt_logprobs` through the OpenAI-style chat protocol so the teacher-server pathway can serve top-k logprobs.
* `swift/template/templates/qwen.py` — guard against vLLM-generated responses containing media pad tokens with no matching media tensor (skip extension instead of crashing).

If you want to see the patches as a single diff, run:

```bash
git log -p -- swift/
```

## License

Apache License 2.0 (inherited from ms-swift). See [LICENSE](./LICENSE).

## Acknowledgements

ParaBridge is built on top of [ms-swift](https://github.com/modelscope/ms-swift) and uses Qwen3-Omni-thinking as the primary backbone. The audio query construction follows the VoxSafeBench pipeline.

## Citation

If you find ParaBridge useful in your research, please cite:

```bibtex
@article{wang2026parabridge,
  title={ParaBridge: Bridging Paralinguistic Perception and Dialogue Behavior in Speech Language Models},
  author={Wang, Yuxiang and Ni, Qinke and Cai, Shengbo and Lin, Wan and Zhang, Liqiang and Wu, Zhizheng},
  journal={arXiv preprint arXiv:2606.10581},
  year={2026}
}
```
