# Why are Self-Explainers Models Better?

--

## Introduction

Li et al (arXiV:2511.08579) report that a target model $M$ is best explained by a fine-tuned versions of iteslf $E$. 
They propose the privileged access hypothesis, which they attribute to models being able to leverage their own internal representations and mechanisms to self-explain.

>Models trained to explain their own internal compu-
>tations can do so more accurately than other models
>trained to explain them.

Their results show that self-explainers outperform non-self explainers in activatoin patching and input ablation experiments. 

However, two alternatives explain the same results without any privileged access, which we investigate 

- **Behavior self-simulation**. $E$ is initialized from $M$'s weights and is close enough in weight-space to approximate $M$'s function directly.
-**Basis compatibility**. E decodes $v$, a patched activation, well because $v$ arrives in coordinate $E$ already reads.

The paper's Section 3.4 and Appendix C.2 establish a correlation between activation alignment and explainer quality, which they claim supports the privileged access hypothesis. That is basis compatibility. Indeed, the authors concede that a model is not explained by itself, but itself and its variants (Table 1), as the most of the Llama-3-8B family does similarly well. 

Additionally, to test non-self explainers in activation patching experiments, the authors a train a linear projection $\Pi: \mathbb{R}^{d'} \to \mathbb{R}^{d}$, to map activations between models. This map was always randomly initialized for cross-model comparisions (3.1), but the authors then claim that pre-training it improves performance. This confounds the cross-model comparison claims of Table 1

## Scope and Consequences

This project tests whether the mechanism on the activation-patching task with Qwen3-8B as target using LoRA and adapter explainers is behavior self-simulation or basis compatibility.  We differ from Table 1 results in the paper, as we use patching not feature description, LoRA and an explicit map not a complete SFT, and use Qwen3 model family.

The strongest claim we can make is that on the task and family where we can measure it, the self-explainer advantage is an effect of coordinate-frames and a closed-form linear fit recovers it.

If the mechanism if basis compatibility, then the recipe for cheap interpretability is to take a good general purpose explainer model and fit a cheap linear map onto target models frames. This is significantly less costly than fine-tuning each model to explain itself. 

## Experimental Design

To break activation alignment while maintaining behavioral similarity, we apply an orthogonal transformation $Q$ to the residual stream of $M$ and call this model $M_Q$. RMSNorm transformers are computationally invariant to such a transform, so $M$ and $M_Q$ compute the identical function. But, every activation extracted from $M_Q$ has the form $Qv$ and there is no longer activation alignment.

We initialize the explainer $E$ from $M$ and test it between activations of $M$ and $M_Q$. The difference in performancw will measure the contribution of activation alignment to the self-explainer advantage. 

Under the behavioral self-simulation hypothesis, $E$ will maintain its advantage reading from $M_Q$. Under the basis-compatibility hypothesis, the advantage collapses to the cross-model baseline.

We target $M = \text{Qwen}3-8\text{B}$ because we can use `act_patch_qwen3_8b_counterfact` dataset.

### Explainers
| Tag | Model | Activation path | Init |
|---|---|---|---|
| `E_self` | Qwen3-8B | direct injection | from unrotated $M$ |
| `E_cross` | Qwen3-4B (primary), 1.7B (secondary) | per-layer $\Pi_\ell$, (excluded from LoRA) | independent pretrained weights |

### Rotation Condition

These are applied to the activations given to $E$. 

| Tag | Transform |
|---|---|
| `R-id` | $v \mapsto v$ |
| `R-Q` | $v \mapsto Qv$, single global $Q \in O(4096)$ |
| `R-Qs` | $v \mapsto \mathrm{diag}(s)\,Qv$, $s$ log-uniform (input side only) |

### Capacity Ladder

To deconfound impacts of activation projection and basis compatability, because $Q^{-1}$ may not be representable by $\Pi$ if it is low rank. So we measure a number of ranks of $\Pi$. 

These are for the different input maps on the injected activation $v$, only for `E_self`. 

| Arm | Input map | Can represent $Q^{-1}$? | Trainable? | Status |
|---|---|---|---|---|
| `C0` | none | No | — | the paper's configuration |
| `C128` | rank 128 | No | Yes | Essentially `Cfull` under LoRA |
| `C8` / `C512` | rank 8 / 512 | No / Partially | Yes | ladder interior |
| `Cfull` | full-rank $d\times d$, init $I$ | Exactly | Yes, full-rank | primary augmented arm |
| `Cfull-rand` | full-rank $d\times d$, random orthogonal init | Exactly | Yes, full-rank | init control |
| `Oracle` | frozen at $Q^\top$ | Exactly | No | representability ceiling |

### 3.4 Projector Init

This is for `E_cross` only.

| Tag | Meaning |
|---|---|
| `P-rand` | random init + rank-128 update (the paper's cross-model condition) |
| `P-rand-full` | random init, full-rank trainable |
| `P-ridge` | closed-form ridge init, then train |
| `P-ridge-frozen` | closed-form ridge fit, (explainer never trained on this target) |

`P-rand` vs `P-rand-full` separates random initialization from random initialization under a rank cap.

We sweep training 

**Sweep.** `N_TRAIN` $\in [128, 256, 512, 1024, 2048, 4096, 8192]$, matched data ordering across arms.

**Seeds.** 3 training seeds at $N \in \{512, 8192\}$ for every core arm; 1 seed elsewhere. 