# Assignment 3: Sequential Instruction Tuning of a Small LLM with Strong-Model Judge Evaluation

**Course:** LLM & Agentic Systems — Graduate Course, UTSA  
**Student:** fpb170  
**Instructor:** Dr. Peyman Najafirad (Paul Rad)  
**TA:** Mohammad Bahrami  
**Due:** April 6th, 2026  

---

## Core Research Question

> *If you first fine-tune a small LLM on Alpaca-style instruction data and then continue fine-tuning on a JSON-structured instruction dataset created through imitation learning from a stronger teacher model, does the model gain structured-output reliability while maintaining its general instruction-following ability, or does catastrophic forgetting degrade the gains from the first stage?*

---

## Section 1: Methodology

### 1.1 Student Model Selection

I selected **Phi-3.5 Mini Instruct** (`microsoft/Phi-3.5-mini-instruct`) as my student model for the following reasons:

- **Strong small-model performance:** Despite having only 3.8B parameters, Phi-3.5 Mini achieves competitive performance on instruction-following benchmarks relative to larger models.
- **Practical suitability for QLoRA:** Its size fits comfortably within the UTSA ARC V100 GPU's 32GB VRAM budget when loaded in 4-bit quantization.
- **MIT License:** No gating or approval required, enabling immediate access and reproducibility.
- **Assignment recommendation:** Phi-3.5 Mini is the explicitly recommended default in the assignment specification.

### 1.2 Alpaca Data Source (Stage 1)

For Stage 1, I used the **Alpaca-Cleaned** dataset (`yahma/alpaca-cleaned`) — a cleaned version of the original Stanford Alpaca dataset. This dataset contains 51,760 instruction-following examples covering a broad range of tasks including open-ended generation, rewriting, brainstorming, summarization, and simple QA.

**Data preparation:**
- Downloaded and cleaned the dataset, removing malformed examples with empty instructions or outputs
- Normalized all examples into the standard `{instruction, input, output}` schema
- Shuffled with seed 42 and split 90/10 into train/eval
- Final split: **46,584 train** / **5,176 eval** (100 used for evaluation as per assignment)

### 1.3 Teacher-Generated JSON Dataset (Stage 2)

For Stage 2, I created a teacher-generated JSON Instruct dataset using **imitation learning** from **Llama 3.3 70B Instruct** (UTSA-hosted, accessed via VPN). This is not classical knowledge distillation — the student only sees the teacher's final text outputs and trains using standard supervised fine-tuning (cross-entropy loss).

**Imitation learning pipeline:**
1. Designed diverse task prompts covering all 5 required task types
2. Fed each prompt to Llama 3.3 70B Instruct
3. Validated every response for JSON correctness using `json.loads()`
4. Discarded and regenerated any invalid JSON responses (up to 3 retries)
5. Saved validated examples in `{instruction, input, output}` schema

**Dataset statistics:**

| Task Type | Examples |
|---|---|
| JSON Extraction | 25 |
| Schema-Constrained Generation | 25 |
| Exact-Label Classification with JSON | 25 |
| JSON Repair | 25 |
| Tool-Call Argument Generation | 25 |
| **Total** | **125** |

Final split: **112 train** / **13 eval**

### 1.4 Training Design

I implemented a two-stage QLoRA fine-tuning pipeline following the assignment specifications exactly:

**QLoRA Configuration (both stages):**
- Quantization: 4-bit NF4
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: all-linear layers
- Trainable parameters: 25,165,824 (0.65% of 3.8B total)

**Training Hyperparameters:**

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Dataset | Alpaca-Cleaned | Teacher JSON |
| Epochs | 2 | 2 |
| Learning rate | 2e-5 | 2e-5 |
| Batch size | 2 | 2 |
| Gradient accumulation | 4 | 4 |
| Max sequence length | 512 | 512 |
| LR scheduler | cosine | cosine |
| Warmup ratio | 0.03 | 0.03 |

**Instruction Template (used consistently at training and inference):**
```
### Instruction:
{instruction}

### Input:
{input}   ← only if non-empty

### Response:
{output}  ← only during training
```

### 1.5 UTSA HPC Setup

Training was performed on UTSA's ARC HPC cluster:
- **Node:** gpu015 (Tesla V100S-PCIE-32GB)
- **CUDA:** 12.3
- **Python:** 3.10
- **Framework:** PyTorch 2.3.0 + HuggingFace Transformers 4.44.0
- **Job scheduler:** SLURM (batch scripts in `scripts/`)

### 1.6 Judge Model and Evaluation Protocol

**Judge model:** Llama 3.3 70B Instruct (UTSA-hosted, accessed via UTSA VPN)

**Evaluation at three checkpoints:**
- **Checkpoint 0:** Untuned base model (baseline)
- **Checkpoint 1:** After Stage 1 Alpaca fine-tuning
- **Checkpoint 2:** After Stage 2 Teacher JSON fine-tuning

**Judge evaluation protocol:**
- Pairwise comparison following Self-Instruct evaluation methodology (Taori et al., 2023)
- Response order randomized to reduce position bias
- Three comparison pairs: C0 vs C1, C1 vs C2, C0 vs C2
- Scored on 6 dimensions (1-5 scale): Instruction Following, Correctness, Clarity, Completeness, Structured Output Validity, Hallucination Risk

---

## Section 2: Experiments

### 2.1 Three-Checkpoint Comparison

**Figure 1: Stage 1 Training Curves (Alpaca Fine-Tuning)**
![Stage 1 Train Loss](figures/stage1_train_loss.png)
*Training loss and learning rate decay over 5,822 steps (2 epochs) on Alpaca-Cleaned dataset.*

**Figure 2: Stage 1 Evaluation Loss**
![Stage 1 Eval Loss](figures/stage1_eval_loss.png)
*Evaluation loss decreasing from 0.92 to 0.874 over training, indicating consistent learning.*

**Figure 3: Stage 2 Training Curves (Teacher JSON Fine-Tuning)**
![Stage 2 Train Loss](figures/stage2_train_loss.png)
*Training loss dropping from 0.85 to 0.52 over 70 steps (10 epochs) on teacher-generated JSON data.*

**Figure 4: Stage 2 Evaluation Loss**
![Stage 2 Eval Loss](figures/stage2_eval_loss.png)
*Evaluation loss at 0.56 after Stage 2 training on JSON data.* Table

*This table will be filled with actual measured values after all training and evaluation completes.*

| Model Checkpoint | Alpaca Judge Win Rate | ROUGE-L | BERTScore F1 | JSON Validity | Schema Compliance | Exact Match |
|---|---|---|---|---|---|---|
| Checkpoint 0: Untuned base | TBD | TBD | TBD | TBD | TBD | TBD |
| Checkpoint 1: After Stage 1 (Alpaca) | TBD | TBD | TBD | TBD | TBD | TBD |
| Checkpoint 2: After Stage 2 (Teacher JSON) | TBD | TBD | TBD | TBD | TBD | TBD |

### 2.2 Alpaca Evaluation Results

*Following the methodology from Taori et al. (2023) using pairwise judge comparison.*

**Win/Tie/Loss rates:**

| Comparison | Wins A | Wins B | Ties | Win Rate A | Win Rate B |
|---|---|---|---|---|---|
| C0 vs C1 | TBD | TBD | TBD | TBD | TBD |
| C1 vs C2 | TBD | TBD | TBD | TBD | TBD |
| C0 vs C2 | TBD | TBD | TBD | TBD | TBD |

**Average judge scores per dimension:**

| Dimension | Checkpoint 0 | Checkpoint 1 | Checkpoint 2 |
|---|---|---|---|
| Instruction Following | TBD | TBD | TBD |
| Correctness | TBD | TBD | TBD |
| Clarity | TBD | TBD | TBD |
| Completeness | TBD | TBD | TBD |
| Structured Output Validity | TBD | TBD | TBD |
| Hallucination Risk | TBD | TBD | TBD |

**Automatic metrics:**

| Metric | Checkpoint 0 | Checkpoint 1 | Checkpoint 2 |
|---|---|---|---|
| ROUGE-1 | TBD | TBD | TBD |
| ROUGE-2 | TBD | TBD | TBD |
| ROUGE-L | TBD | TBD | TBD |
| BERTScore F1 | TBD | TBD | TBD |
| Avg output length | TBD | TBD | TBD |
| Task completion rate | TBD | TBD | TBD |

### 2.3 JSON Structured Output Evaluation

**JSON metrics at each checkpoint:**

| Metric | Checkpoint 0 | Checkpoint 1 | Checkpoint 2 |
|---|---|---|---|
| JSON Validity Rate | TBD | TBD | TBD |
| Schema Compliance Rate | TBD | TBD | TBD |
| Exact Match Accuracy | TBD | TBD | TBD |
| Field-Level F1 (extraction) | TBD | TBD | TBD |

**Common error taxonomy (Checkpoint 0 baseline):**

| Error Type | Count |
|---|---|
| Missing quotes | TBD |
| Trailing comma | TBD |
| Missing brackets | TBD |
| Wrong types | TBD |
| Other | TBD |

### 2.4 Forgetting Analysis

*The central analytical contribution of this assignment.*

| Metric | Checkpoint 1 | Checkpoint 2 | Absolute Change | % Change | Verdict |
|---|---|---|---|---|---|
| ROUGE-1 | TBD | TBD | TBD | TBD | TBD |
| ROUGE-2 | TBD | TBD | TBD | TBD | TBD |
| ROUGE-L | TBD | TBD | TBD | TBD | TBD |
| BERTScore F1 | TBD | TBD | TBD | TBD | TBD |
| Judge Win Rate | TBD | TBD | TBD | TBD | TBD |

**Key finding:** TBD — Did catastrophic forgetting occur? Was it significant?

### 2.5 Ablation Study Results

I conducted 3 ablation experiments on Stage 2 training to identify which training decisions most strongly influence the forgetting/retention tradeoff.

#### Ablation 1: Vary Stage 2 Epochs

| Variant | Epochs | ROUGE-L (Alpaca) | JSON Validity | Exact Match |
|---|---|---|---|---|
| epochs_1 | 1 | TBD | TBD | TBD |
| epochs_2 | 2 (baseline) | TBD | TBD | TBD |
| epochs_3 | 3 | TBD | TBD | TBD |

**Finding:** TBD — Does more training cause more forgetting?

#### Ablation 2: Vary Stage 2 Learning Rate

| Variant | LR | ROUGE-L (Alpaca) | JSON Validity | Exact Match |
|---|---|---|---|---|
| lr_2e5 | 2e-5 (baseline) | TBD | TBD | TBD |
| lr_1e5 | 1e-5 | TBD | TBD | TBD |
| lr_5e6 | 5e-6 | TBD | TBD | TBD |

**Finding:** TBD — Does lower LR reduce forgetting?

#### Ablation 3: Vary Stage 2 Dataset Size

| Variant | Data % | Train Examples | ROUGE-L (Alpaca) | JSON Validity | Exact Match |
|---|---|---|---|---|---|
| data_100 | 100% (baseline) | 112 | TBD | TBD | TBD |
| data_50 | 50% | 56 | TBD | TBD | TBD |
| data_25 | 25% | 28 | TBD | TBD | TBD |

**Finding:** TBD — Does less data reduce forgetting while maintaining JSON improvement?

---

## Section 3: Analysis

### 3.1 Qualitative Output Comparison

*Representative examples showing model behavior at each checkpoint.*

**Example 1 — General instruction (Alpaca eval):**

| | Response |
|---|---|
| **Instruction** | TBD |
| **Checkpoint 0** | TBD |
| **Checkpoint 1** | TBD |
| **Checkpoint 2** | TBD |

**Example 2 — JSON extraction task:**

| | Response |
|---|---|
| **Instruction** | TBD |
| **Checkpoint 0** | TBD |
| **Checkpoint 1** | TBD |
| **Checkpoint 2** | TBD |

### 3.2 Failure Case Analysis

TBD — After results are available, I will document cases where:
- Checkpoint 2 regressed on Alpaca tasks compared to Checkpoint 1
- The model generated invalid JSON even after Stage 2 training
- The model hallucinated content in structured outputs

### 3.3 Discussion: Forgetting vs Retention

TBD — Based on my results, I will discuss:

**If forgetting occurred:**
- Possible causes: learning rate too high, too many epochs, insufficient data diversity
- Which instruction types were most affected (open-ended vs summarization vs QA)
- Whether the JSON improvement justified the Alpaca regression

**If forgetting did NOT occur:**
- What may have prevented it: small dataset size of Stage 2, conservative learning rate, QLoRA's parameter efficiency
- Whether the model successfully balanced both capabilities

### 3.4 Implications for Sequential Fine-Tuning

TBD — What my results imply about:
- The safety of sequential fine-tuning for specialization
- The role of dataset size in catastrophic forgetting
- The effectiveness of imitation learning for structured output tasks
- Practical recommendations for post-training pipelines

---

## Section 4: Prompt Engineering

### 4.1 Teacher Model Prompt Design

I designed prompts for 5 JSON task types. Each prompt template is stored in `prompts/` as an editable file.

**Design principles:**
1. **Clear task specification:** Each prompt explicitly states what JSON structure is expected
2. **Consistent schema:** All prompts follow `{instruction, input, output}` format
3. **Diversity:** Prompts cover different domains (medical, financial, technical, social)
4. **Realistic inputs:** Input texts are realistic and varied to prevent overfitting

**Example — JSON Extraction prompt design:**
```
You are a precise JSON extraction assistant. Your job is to extract 
structured information from unstructured text and return it as valid 
JSON only.

Rules:
- Return ONLY valid JSON. No explanation, no markdown, no code blocks.
- Use null for missing values.
- Use arrays for multiple values.
- All keys must be lowercase with underscores.

Task: Extract all relevant entities and attributes from the following 
text into a JSON object.

Text: {input_text}

Return only the JSON object.
```

**Iteration process:** Initial prompts produced markdown-wrapped responses (` ```json ``` `). I added explicit rules — "No markdown, no code blocks" — which reduced invalid outputs significantly. I also added a system message reinforcing JSON-only output.

### 4.2 Judge Prompt Design

I designed separate judge prompts for Alpaca and JSON evaluations:

**Key design decisions:**
1. **Structured scoring:** 6 dimensions with 1-5 scale forces nuanced evaluation
2. **JSON output:** Judge returns structured JSON for easy aggregation
3. **Position bias mitigation:** Response order randomized (A/B swapped) for 50% of examples
4. **Task-specific prompts:** JSON judge prompt emphasizes schema validity; Alpaca prompt emphasizes instruction following

**Judge prompt iteration:** Initial prompts sometimes produced verbose justifications that interfered with JSON parsing. I added "Return ONLY valid JSON" and a system message to resolve this.

---

## Appendix: Full Prompt Templates

### Teacher Model Prompts

All prompt templates are stored in `prompts/` directory:

- `prompts/json_extraction.txt` — JSON extraction from unstructured text
- `prompts/schema_generation.txt` — Schema-constrained generation
- `prompts/classification.txt` — Classification with JSON output
- `prompts/json_repair.txt` — JSON repair and formatting correction
- `prompts/tool_call.txt` — Tool-call argument generation

### Judge Evaluation Prompts

- `prompts/judge_alpaca.txt` — Judge prompt for general instruction following
- `prompts/judge_json.txt` — Judge prompt for structured JSON output evaluation

---

## References

1. Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*
2. Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv:2305.14314*
3. Taori, R. et al. (2023). Alpaca: A Strong, Replicable Instruction-Following Model. *Stanford CRFM*
4. Wang, Y. et al. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions. *ACL 2023*
5. Gu, J. et al. (2024). A Survey on LLM-as-a-Judge. *arXiv:2411.15594*
6. Kenton, Z. et al. (2024). On Scalable Oversight with Weak LLMs Judging Strong LLMs. *DeepMind*
7. Rafailov, R. et al. (2024). From Human Preferences to Post-Training Alignment Pipelines. *arXiv*


---

## Appendix: Full Prompt Templates

### A1. Teacher Model Prompts (Imitation Learning)

These prompts were used to generate the Stage 2 training dataset by feeding them to Llama 3.3 70B Instruct. All outputs were validated for JSON correctness before inclusion.

---

#### A1.1 JSON Extraction Prompt (`prompts/json_extraction.txt`)

```
You are a precise JSON extraction assistant. Your job is to extract structured information from unstructured text and return it as valid JSON only.

Rules:
- Return ONLY valid JSON. No explanation, no markdown, no code blocks.
- Use null for missing values.
- Use arrays for multiple values.
- All keys must be lowercase with underscores.

Task: Extract all relevant entities and attributes from the following text into a JSON object.

Text: {input_text}

Return only the JSON object.
```

**Design rationale:** The explicit "no markdown, no code blocks" rule was added after initial prompts returned responses wrapped in triple backticks. The lowercase underscore rule enforces consistent key naming across all extraction examples.

---

#### A1.2 Schema-Constrained Generation Prompt (`prompts/schema_generation.txt`)

```
You are a precise JSON generation assistant. Your job is to generate a valid JSON object that strictly conforms to the given schema.

Rules:
- Return ONLY valid JSON. No explanation, no markdown, no code blocks.
- Every required field in the schema must be present.
- Value types must exactly match the schema types.
- Use realistic and meaningful values.

Task: Generate a valid JSON object that conforms to this schema:

Schema: {schema}

Context: {context}

Return only the JSON object.
```

**Design rationale:** The "realistic and meaningful values" rule prevents the model from generating placeholder values like "string" or "integer" instead of actual data. The context field guides the model toward domain-appropriate values.

---

#### A1.3 Classification with JSON Output Prompt (`prompts/classification.txt`)

```
You are a precise text classification assistant. Your job is to classify the given text and return the result as valid JSON only.

Rules:
- Return ONLY valid JSON. No explanation, no markdown, no code blocks.
- Use only the allowed labels provided.
- Include a confidence score between 0.0 and 1.0.
- Include a brief reason field.

Task: Classify the following text using only the allowed labels.

Text: {input_text}

Allowed labels: {labels}

Return a JSON object with exactly these fields: {"label": "...", "confidence": 0.0, "reason": "..."}
```

**Design rationale:** The explicit output schema at the end (`{"label": ..., "confidence": ..., "reason": ...}`) was added after initial outputs used inconsistent field names. Specifying allowed labels prevents label hallucination.

---

#### A1.4 JSON Repair Prompt (`prompts/json_repair.txt`)

```
You are a precise JSON repair assistant. Your job is to fix malformed JSON and return only the corrected valid JSON.

Rules:
- Return ONLY valid JSON. No explanation, no markdown, no code blocks.
- Fix all syntax errors such as missing quotes, missing brackets, trailing commas, wrong types.
- Preserve the original structure and values as much as possible.
- Do not add or remove fields unless necessary to make it valid.

Task: Fix the following malformed JSON and return the corrected version.

Malformed JSON: {malformed_json}

Input: {malformed_json}

Return only the corrected valid JSON object.
```

**Design rationale:** The "preserve original structure" rule prevents the model from completely rewriting the JSON rather than minimally fixing it. The "do not add or remove fields" rule ensures the repair is surgical.

---

#### A1.5 Tool-Call Argument Generation Prompt (`prompts/tool_call.txt`)

```
You are a precise function call assistant. Your job is to generate a valid JSON object representing a function call with the correct named parameters.

Rules:
- Return ONLY valid JSON. No explanation, no markdown, no code blocks.
- All required parameters must be present.
- Parameter types must match the function signature exactly.
- Use realistic and meaningful values based on the context.

Task: Generate a JSON object representing a call to the following function.

Function name: {function_name}
Function description: {function_description}
Parameters: {parameters}
Context: {context}

Return a JSON object with exactly this structure: {"function": "...", "arguments": {...}}
```

**Design rationale:** The explicit output structure `{"function": ..., "arguments": {...}}` standardizes the format across all tool-call examples. The context field provides the scenario that determines parameter values.

---

### A2. Student Model Instruction Template

This template was used consistently during both training and inference. It is critical that the same template is used at inference time as was used during training.

```
### Instruction:
{instruction}

### Input:
{input}   ← only included if input is non-empty

### Response:
{output}  ← only included during training, omitted during inference
```

**Why this template:** Following the TA's recommendation and the Alpaca paper's approach, I use a simple three-part template that clearly separates the instruction, optional input, and expected response. The `### Response:` marker signals to the model where to begin generating.

---

### A3. Judge Evaluation Prompts

#### A3.1 Alpaca Judge Prompt (`prompts/judge_alpaca.txt`)

```
You are an expert judge evaluating the quality of AI assistant responses.
You will be shown an instruction and two responses (Response A and Response B).
Your task is to evaluate both responses and determine which is better.

Instruction:
{instruction}

Input (if any):
{input}

Response A:
{response_a}

Response B:
{response_b}

Please evaluate both responses on these 6 dimensions (score 1-5 each):
1. instruction_following: How well does the response follow the instruction?
2. correctness: Is the response factually correct and accurate?
3. clarity: Is the response clear and well-written?
4. completeness: Does the response fully address the instruction?
5. structured_output_validity: Is any structured output valid and well-formed?
6. hallucination_risk: How likely is fabricated information? (5=no hallucination, 1=high)

Return ONLY valid JSON in this exact format:
{
    "response_a_scores": {
        "instruction_following": <1-5>,
        "correctness": <1-5>,
        "clarity": <1-5>,
        "completeness": <1-5>,
        "structured_output_validity": <1-5>,
        "hallucination_risk": <1-5>
    },
    "response_b_scores": {
        "instruction_following": <1-5>,
        "correctness": <1-5>,
        "clarity": <1-5>,
        "completeness": <1-5>,
        "structured_output_validity": <1-5>,
        "hallucination_risk": <1-5>
    },
    "winner": "<A|B|tie>",
    "justification": "<brief explanation>"
}
```

**Design rationale:** The 6-dimension scoring rubric forces nuanced evaluation beyond simple win/loss. The hallucination_risk dimension is scored inversely (5=safe, 1=risky) which required careful wording to avoid confusion. The JSON output format enables automated aggregation across 100+ comparisons. Response order is randomized 50% of the time to reduce position bias.

---

#### A3.2 JSON Judge Prompt (`prompts/judge_json.txt`)

```
You are an expert judge evaluating the quality of AI assistant responses to structured JSON tasks.
You will be shown an instruction and two responses (Response A and Response B).

Instruction:
{instruction}

Input (if any):
{input}

Response A:
{response_a}

Response B:
{response_b}

Please evaluate both responses on these 6 dimensions (score 1-5 each):
1. instruction_following: How well does the response follow the instruction?
2. correctness: Is the JSON content correct and accurate?
3. clarity: Is the response clear and well-structured?
4. completeness: Does the response include all required fields?
5. structured_output_validity: Is the JSON syntactically valid and schema-compliant? (5=perfect, 1=invalid)
6. hallucination_risk: Does it contain fabricated values? (5=no hallucination, 1=high)

Return ONLY valid JSON in this exact format:
{
    "response_a_scores": {
        "instruction_following": <1-5>,
        "correctness": <1-5>,
        "clarity": <1-5>,
        "completeness": <1-5>,
        "structured_output_validity": <1-5>,
        "hallucination_risk": <1-5>
    },
    "response_b_scores": {
        "instruction_following": <1-5>,
        "correctness": <1-5>,
        "clarity": <1-5>,
        "completeness": <1-5>,
        "structured_output_validity": <1-5>,
        "hallucination_risk": <1-5>
    },
    "winner": "<A|B|tie>",
    "justification": "<brief explanation focusing on JSON quality>"
}
```

**Design rationale:** This prompt differs from the Alpaca judge by emphasizing `structured_output_validity` as the primary quality signal for JSON tasks. The justification explicitly asks to focus on JSON quality. The system message `"You are an expert AI judge. Always respond with valid JSON only."` prevents the judge from wrapping its response in prose.

---

### A4. Prompt Engineering Iteration Log

| Iteration | Problem | Fix | Result |
|---|---|---|---|
| v1 | Teacher responses wrapped in ```json``` | Added "no markdown, no code blocks" rule | Reduced markdown wrapping by ~80% |
| v2 | Teacher used placeholder values ("string", "integer") | Added "realistic and meaningful values" | More natural JSON outputs |
| v3 | Judge responses included prose before JSON | Added system message "respond with valid JSON only" | JSON parsing success rate improved |
| v4 | Classification outputs used unlisted labels | Added explicit allowed labels list | Zero label hallucination |
| v5 | Tool-call outputs had inconsistent structure | Added explicit output format template | Standardized structure across all examples |
| v6 | JSON repair rewrote entire structure | Added "preserve original structure" rule | Minimal surgical fixes |
