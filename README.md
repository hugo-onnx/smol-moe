# SmolMoE

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10+](https://img.shields.io/badge/PyTorch-2.10+-ee4c2c.svg)](https://pytorch.org/)

A from-scratch PyTorch implementation of a **Mixture-of-Experts (MoE)** causal language model (~135 M parameters). The project covers the full pipeline: component implementation, dense-to-MoE upcycling, continued pretraining, and domain-specialized expert training.

**Base model:** [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) (HuggingFace)

## Architecture

```
Input tokens
     │
     ▼
┌─────────────────────────────────┐
│  Token Embedding (49.152 vocab) │
└─────────────────┬───────────────┘
                  │
          ×30 decoder layers
                  │
     ┌────────────▼────────────┐
     │   RMSNorm               │
     │      │                  │
     │   RoPE GQA              │  ← 9 Q-heads / 3 KV-heads
     │      │                  │
     │   residual add          │
     ├─────────────────────────┤
     │   RMSNorm               │
     │      │                  │
     │   MoE (top-1 of 3)      │  ← router → sparse dispatch
     │    ├─ Expert 0 (SwiGLU) │
     │    ├─ Expert 1 (SwiGLU) │
     │    └─ Expert 2 (SwiGLU) │
     │      │                  │
     │   residual add          │
     └────────────┬────────────┘
                  │
     ┌────────────▼────────────┐
     │   Final RMSNorm         │
     │   LM Head (tied emb.)   │
     └─────────────────────────┘
```

**Key design choices:**
- Pre-norm residual stream (RMSNorm before each sublayer)
- Grouped-Query Attention (9Q / 3KV) for memory-efficient inference
- Top-1 sparse routing with auxiliary load-balancing loss
- Weight tying between token embeddings and the LM head
- Router noise injection and expert dropout for domain specialization

## Features

| Capability | Details |
|---|---|
| Dense-to-MoE upcycling | Copy pre-trained SmolLM-135M weights into each expert; outputs match exactly |
| Continued pretraining | Full training loop with MoE-aware metrics (utilization, load-balancing loss) |
| Domain-specialized routing | Domain-supervised routing loss, curriculum sampling, knowledge distillation |
| Expert dropout | Forces domain-specific routing during training (actually affects routing — not a no-op) |
| Router noise injection | Annealed Gaussian noise applied before argmax, not after (functional exploration) |
| Padding-aware attention | `attention_mask` correctly masks padded key positions at the decoder layer |

## Project Structure

```
src/
├── config.py              # SmolMoEConfig — single source of truth for hyperparams
├── models/
│   ├── components.py      # RotaryEmbedding, RMSNorm, MixtureOfExperts, RoPEAttention
│   └── smol_moe.py        # SmolMoEDecoderLayer, SmolMoEModel, SmolMoEForCausalLM
├── training.py            # Trainer, TrainingConfig, MoEMetrics, loss functions
├── upcycling.py           # Dense→MoE weight conversion and verification
├── domain_expert.py       # Domain-specialized training (curriculum, KD, router supervision)
└── utils/
    ├── env.py             # HF token loading from .env / env vars
    └── helpers.py         # Timing decorators, MetricsTracker, checkpointing, generation

tests/                     # pytest suite — 80 unit and integration tests
config/default.json        # Default hyperparams
scripts/
├── download_weights.py    # Download pre-trained weights from HuggingFace
├── upcycle.py             # Step 4 — dense → MoE weight conversion
├── pretrain.py            # Step 5 — continued pretraining
├── domain_train.py        # Step 6 — domain-specialized expert training
└── generate.py            # Step 7 — text generation
```

---

## Pipeline Walkthrough

The following steps cover the project end-to-end, in the intended order of execution. Each step builds on the previous one.

### Step 0 — Install

```bash
cd smol-moe

# Install runtime + dev dependencies (pytest, coverage)
uv sync --extra dev
```

Verify the installation:

```bash
uv run python -c "from smol_moe import SmolMoEForCausalLM, SmolMoEConfig; print('OK')"
```

Python 3.12+ and PyTorch 2.10+ are required.

---

### Step 1 — Run the test suite

All 80 tests should pass before doing anything else. This confirms every component works correctly on your machine.

```bash
uv run pytest                          # full suite (~11 s on CPU)
uv run pytest -v                       # verbose output
uv run pytest --cov=smol_moe --cov-report=html   # with coverage report
```

Run targeted subsets if you want to focus on a specific module:

```bash
uv run pytest tests/test_components.py -v          # RoPE, RMSNorm, MoE, Attention
uv run pytest tests/test_model.py -v               # full forward pass, generation
uv run pytest tests/test_upcycling.py -v           # weight copying
uv run pytest tests/test_training.py -v            # loss functions, training steps
uv run pytest tests/test_domain_expert.py -v       # domain routing, curriculum
```

---

### Step 2 — Explore the model components

Use `SmolMoEConfig.small()` (4 layers, 2 experts) to iterate quickly without loading large weights.

```python
import torch
from smol_moe import SmolMoEConfig, SmolMoEForCausalLM

config = SmolMoEConfig.small()
model = SmolMoEForCausalLM(config)
input_ids = torch.randint(0, config.vocab_size, (2, 32))
outputs = model(input_ids)
print(outputs['logits'].shape)   # [2, 32, 49152]

utilization, lb_loss = model.get_expert_utilization()
print(f"expert utilization: {utilization}")
print(f"load-balancing loss: {lb_loss:.4f}")
```

To use the full 135 M configuration, replace `SmolMoEConfig.small()` with `SmolMoEConfig()`. Loading pre-trained weights is covered in Step 3.

---

### Step 3 — (Optional) Download pre-trained weights

The pre-trained weights let you skip training and go straight to inference or fine-tuning. A HuggingFace account is required.

**Set up credentials:**

```bash
# Create a .env file in the project root
echo "HF_TOKEN=hf_your_token_here" > .env
```

Get your token at https://huggingface.co/settings/tokens (read access is enough).

**Download:**

```bash
uv run python scripts/download_weights.py
# Saves to ./weights/trial_weights.pt (~520 MB)

# Or pass the token directly
uv run python scripts/download_weights.py --token hf_your_token_here

# Custom output directory
uv run python scripts/download_weights.py --output-dir ./my_weights
```

**Load into the model:**

```python
import torch
from smol_moe import SmolMoEConfig, SmolMoEForCausalLM

model = SmolMoEForCausalLM(SmolMoEConfig())
model.load_state_dict(torch.load("weights/trial_weights.pt", map_location="cpu"))
model.eval()
```

---

### Step 4 — Dense-to-MoE upcycling

Sparse upcycling converts a pre-trained dense SmolLM-135M into a MoE model by copying the FFN weights into every expert. All experts start identical, so the MoE output exactly matches the dense model — providing a strong warm-start for MoE training.

```bash
uv run python scripts/upcycle.py
# Saves to weights/upcycled_moe.pt

# Custom number of experts or output path
uv run python scripts/upcycle.py --num-experts 4 --output weights/upcycled_moe.pt
```

> **Requires:** the dense model downloads ~270 MB from HuggingFace on first run.

---

### Step 5 — Continued pretraining

Fine-tune the upcycled MoE on a text corpus. The router learns to differentiate between experts while the load-balancing loss prevents expert collapse.

```bash
uv run python scripts/pretrain.py
# Saves to weights/pretrained_moe.pt

# Quick CPU smoke test
uv run python scripts/pretrain.py --steps 50 --max-samples 500

# Custom dataset, steps, and output
uv run python scripts/pretrain.py --dataset HuggingFaceTB/cosmopedia-100k --steps 500 --output weights/pretrained_moe.pt
```

> **GPU recommended.** Mixed precision is enabled automatically when a CUDA GPU is available.

---

### Step 6 — Domain-specialized expert training

Train each expert to specialize in a different domain (chat, code, math). This stage adds three techniques on top of continued pretraining:

- **Domain-supervised routing loss** — cross-entropy that pushes the router to send domain tokens to the designated expert
- **Curriculum learning** — domain sampling probabilities anneal from biased to uniform over training
- **Knowledge distillation** *(optional)* — KL divergence from a teacher model

```bash
uv run python scripts/domain_train.py
# Saves to weights/domain_expert_moe.pt

# Custom domains, steps, and routing loss weight
uv run python scripts/domain_train.py --domains chat code math --steps 500 --lambda-route 0.5

# Use a different multi-domain dataset (must have input/output columns)
uv run python scripts/domain_train.py --dataset your/dataset --subset SFT --domains chat code math
```

> Requires a HuggingFace token to access the Nemotron dataset (gated). Any multi-domain SFT dataset with `input`/`output` columns works as a drop-in replacement.

---

### Step 7 — Text generation

```bash
uv run python scripts/generate.py --prompt "Explain what a Mixture of Experts model is:"

# Sampling with temperature, top-k, and top-p
uv run python scripts/generate.py \
    --prompt "Explain what a Mixture of Experts model is:" \
    --temperature 0.8 --top-k 50 --top-p 0.95

# Use a specific checkpoint
uv run python scripts/generate.py --weights weights/pretrained_moe.pt --prompt "def fibonacci"
```

---

## Configuration reference

```python
SmolMoEConfig(
    vocab_size=49152,           # SmolLM tokenizer vocabulary
    hidden_size=576,            # embedding / attention dimension
    intermediate_size=1536,     # expert MLP hidden dimension
    num_hidden_layers=30,       # decoder depth
    num_heads=9,                # query attention heads
    kv_heads=3,                 # key/value heads (GQA: 3 queries share 1 KV)
    num_experts=3,              # total expert count
    num_experts_per_tok=1,      # top-k routing (sparse)
    rope_theta=10000.0,         # RoPE base frequency
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,   # LM head shares embedding weights
)
```

Use `SmolMoEConfig.small()` (4 layers, 256 hidden, 2 experts) for fast iteration and tests.

---

## Tests

```bash
# Full suite
uv run pytest

# With HTML coverage report
uv run pytest --cov=smol_moe --cov-report=html

# Individual modules
uv run pytest tests/test_components.py::TestMixtureOfExperts -v
uv run pytest tests/test_model.py::TestSmolMoEForCausalLM -v
uv run pytest tests/test_domain_expert.py::TestExpertDropout -v

# Specific new tests for the bug fixes
uv run pytest tests/test_model.py::TestSmolMoEForCausalLM::test_padding_mask_applied -v
uv run pytest tests/test_domain_expert.py::TestExpertDropout::test_expert_dropout_forces_routing -v
uv run pytest tests/test_domain_expert.py::TestExpertDropout::test_router_noise_affects_routing -v
```

---

## Results

The numbers below come from a single end-to-end run on a MacBook Pro M4 (CPU only). GPU training will produce better results in less time.

### Dense-to-MoE upcycling

| Check | Result |
|---|---|
| Output match (max abs diff) | < 1e-4 |
| Verification | Passed |

All FFN weights were copied into all 3 experts. Router logits were zeroed so every expert selects deterministically and the MoE output is identical to the dense baseline.

### Continued pretraining (200 steps, cosmopedia-100k)

| Metric | Start | End |
|---|---|---|
| Eval loss | 2.204 | 2.194 |
| Utilization entropy | 100% | 100% |
| Dominant expert | Expert 2 | Expert 2 (~78%) |

Load balancing held perfectly (100% utilization entropy). Expert 2 emerged as the dominant expert — a pretraining bias that domain training later had to overcome.

### Domain-specialized training (500 steps, Nemotron SFT — chat / code / math)

| Metric | Value |
|---|---|
| EDAS (Expert-Domain Alignment) | **60%** (random baseline: 33%) |
| Chat → Expert 0 | 97.8% |
| Math → Expert 2 | 62.0% |
| Code → Expert 0 | 61.2% |

Chat routing is near-perfect. Math and code both improved well above chance. Code routing landed on Expert 0 instead of Expert 1, likely because code and chat share textual structure and Expert 1 received little pretraining signal — addressable with more steps or a stronger pretraining signal per expert.

---

## Next Steps

The following improvements are most likely to raise EDAS and overall generation quality:

**Expert specialization**
- **More training steps.** 500 steps is a short run. 2 000–5 000 steps with a cosine LR schedule and warmup should sharpen code and math boundaries further.
- **Enable knowledge distillation.** The domain trainer accepts `lambda_kd > 0`. Using SmolLM-135M-Instruct as the teacher would add a token-level KL loss that guides each expert toward the correct answer distribution for its domain.
- **Per-expert pretraining data.** Before domain training, warm up each expert independently on domain-specific text (code: The Stack, math: MATH/OpenWebMath, chat: UltraChat). This gives Expert 1 enough signal to compete on code.

**Architecture**
- **Top-2 routing.** Changing `num_experts_per_tok` from 1 to 2 lets every token combine two expert outputs, which reduces variance and is the default in most production MoE models.
- **Scale experts.** Going from 3 to 8 experts with top-2 routing adds capacity with only a modest FLOP increase per token.
- **Flash Attention.** Dropping in `torch.nn.functional.scaled_dot_product_attention` with `enable_flash=True` cuts attention memory and speeds up GPU training substantially.

**Infrastructure**
- **Gradient checkpointing.** Needed for longer context or larger batch sizes on GPU without running out of memory.

---

## References

- [RoFormer: Rotary Position Embeddings](https://arxiv.org/abs/2104.09864)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [GQA: Grouped-Query Attention](https://arxiv.org/abs/2305.13245)
- [Sparse Upcycling](https://arxiv.org/abs/2212.05055)

## License

MIT — see [LICENSE](LICENSE).