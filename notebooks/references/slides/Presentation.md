---
marp: true
theme: gaia
class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
style: |
  section { font-size: 24px; text-align: left; }
  h1 { color: #2c3e50; font-size: 40px; }
  h2 { color: #e67e22; font-size: 30px; }
  strong { color: #c0392b; }
  code { background-color: #f0f0f0; padding: 2px 5px; border-radius: 4px; color: #d35400; }
  blockquote { background: #f9f9f9; border-left: 10px solid #ccc; margin: 1.5em 10px; padding: 0.5em 10px; font-size: 20px;}
  
  /* UPDATED TABLE STYLING */
  table { 
    font-size: 20px; 
    border-collapse: collapse; 
    width: 100%; 
    margin-top: 20px;
  }
  th {
    background-color: #2c3e50; /* Matches H1 */
    color: #ffffff;
    padding: 10px;
    border: 1px solid #ddd;
  }
  td {
    padding: 10px;
    border: 1px solid #ddd;
    color: #333;
  }
  tr:nth-child(even) {
    background-color: #f2f2f2;
  }
---

# Human vs. AI Text Classification
## A 6-Metric Linguistic Baseline

**Subject:** SC203 - Scientific Method
**Research Team:** NLP Group
**Dataset:** 58k Samples (Roy et al., 2025)

---

# Agenda

1. **Motivation**: The "Arms Race" of Generative AI.
2. **The Dataset**: 58k samples from NYT vs. 6 LLMs.
3. **Pipeline Architecture**: From Ingestion to Parquet.
4. **The 6 Core Metrics**: Definitions & **Calculation Logic**.
5. **Lexical Analysis**: IRAL Log-Odds Ratio.
6. **Findings & Discussion**: Interpreting the "Human Fingerprint".

---

# The Problem: AI is "More Human than Human"

- **Context:** Large Language Models (LLMs) like GPT-4 and Mistral produce highly coherent text.
- **The Challenge:** Distinguishing AI text is critical for academic integrity and preventing misinformation.
- **Current State:** "Zero-shot" black-box detectors often fail on domain-specific text (e.g., scientific writing).

> "The need to discriminate human writing from AI is now both critical and urgent." — *Desaire et al. (2023)*

---

# Our Approach: "White Box" Detection

Instead of opaque neural networks, we propose a **Linguistic Baseline** using 6 explainable metrics.

### Why "White Box"?
1.  **Explainability:** We can point to *why* a text is flagged (e.g., "Too repetitive", "Lack of hedging").
2.  **Scientific Grounding:** Based on linguistic features proven to differentiate human/AI writing, such as lexical diversity and sentence complexity.

---

# The Dataset: Roy et al. (2025)

We utilize the **"Comprehensive Dataset for Human vs. AI Generated Text Detection"**.

- **Scale:** **58,000+** text samples.
- **Human Baseline:** New York Times (NYT) articles (High-quality, edited journalism).
- **AI Contenders:** 6 Models generated from abstract prompts:
  - Gemma-2-9b, Mistral-7B, Qwen-2-72B
  - LLaMA-8B, Yi-Large, GPT-4o

---

# Data Cleaning & Pipeline

Based on our project architecture:

1.  **Ingestion:** Raw CSVs from Roy et al. loaded via `src.cli`.
2.  **Topic Filtering:** Subset `data/cleaned_by_topic/environment.csv` used for focused benchmarking.
3.  **Sanitization:**
    - Removal of nulls and artifacts.
    - Alignment of `human_story` vs. `model_output` columns.
4.  **Processing:** Sharded processing with `spaCy` and `DuckDB` (Parquet output).

---

# The 6 Core Metrics

We extract these features to capture the "fingerprint" of human writing:

1.  **MTLD** (Lexical Diversity)
2.  **Nominalization Density** (Academic formality)
3.  **Modal/Epistemic Rate** (Hedging/Equivocal language)
4.  **Clause Complexity** (Syntactic depth)
5.  **Passive Voice Ratio** (Stylistic preference)
6.  **S2S Cosine Similarity** (Semantic repetition)

---

# Metric 1: MTLD (Lexical Diversity)

**Definition:** *Measure of Textual Lexical Diversity*. It measures vocabulary richness by calculating the mean length of sequential word strings that maintain a Type-Token Ratio (TTR) above a threshold (0.72).

**Relevance:**
- Humans generally use more varied, context-rich vocabulary tied to personal experience.
- AI tends to be "safe" and repetitive, leading to lower diversity scores in long texts.

---

# Metric 1: Calculation Logic

**Algorithm:**
1. Initialize `TTR = 1.0`, `count = 0`, `factors = 0`.
2. Iterate through tokens. Update TTR (Unique/Total).
3. If `TTR < 0.72`:
   - Increment `factors`.
   - Reset TTR.
4. Final Score: `Total Words / Factors`.

**Example:**
- **Human:** "The feline slept. The pet rested." (High MTLD $\rightarrow$ TTR stays high).
- **AI:** "The cat sat. The cat sat." (Low MTLD $\rightarrow$ TTR drops fast).

---

# Metric 2: Nominalization Density

**Definition:** The frequency of nouns derived from verbs or adjectives (e.g., *implement* $\rightarrow$ *implementation*).

**Relevance:**
- A marker of **formal, academic human writing**.
- AI models often simplify phrasing for clarity, reducing nominalization density.

**Formula:**
$$D_{nom} = \frac{\text{Count}(\text{suffix} \in \{-\text{tion}, -\text{ment}, -\text{ness}, -\text{ity}\}) \times 1000}{\text{Total Words}}$$

---

# Metric 2: Calculation Example

**Compare:**

> **Human (High Density):** "The **implementation** of the **regulation** caused **frustration**."
> *3 nominalizations per 7 words.*

> **AI (Low Density):** "People were frustrated because they implemented the rule."
> *0 nominalizations.*

---

# Metric 3: Modal & Epistemic Rate

**Definition:** The frequency of "hedging" words (modals) and contrastive conjunctions.

**Relevance:**
- Desaire et al. (2023) found scientists have a penchant for **equivocal language** (*however, although*).
- AI tends to be declarative and confident.

**Target Tokens:**
`[might, may, could, perhaps, possible, unlikely, however, although, but]`

---

# Metric 3: Calculation Example

**Formula:**
$$R_{modal} = \frac{\text{Count}(\text{Target Tokens}) \times 100}{\text{Total Words}}$$

**Example:**
> **Human:** "These results **suggest** that it **may** be possible, **although** further study is needed."
> *High Epistemic Rate.*

> **AI:** "This proves that it is possible. Future studies are needed."
> *Low Epistemic Rate.*

---

# Metric 4: Clause Complexity

**Definition:** The average depth of the syntactic dependency tree.

**Relevance:**
- Humans exhibit "Burstiness": a mix of simple and deeply complex sentences.
- Desaire et al. noted that sentence length diversity is a key feature of human writing.

**Calculation:**
Using `spaCy` dependency parsing, we calculate the maximum depth from the `ROOT` verb to the furthest leaf node.

---

# Metric 4: Calculation Example

**Sentence:** *"The dog that chased the cat, which was fast, barked."*

**Tree:**
1. **barked** (ROOT, Depth 0)
2. $\rightarrow$ **dog** (nsubj, Depth 1)
3. $\rightarrow$ $\rightarrow$ **chased** (relcl, Depth 2)
4. $\rightarrow$ $\rightarrow$ $\rightarrow$ **cat** (dobj, Depth 3)
5. $\rightarrow$ $\rightarrow$ $\rightarrow$ $\rightarrow$ **was** (relcl, Depth 4)

**Human:** Mean Depth 4.5 | **AI:** Mean Depth 3.0 (Flatter trees).

---

# Metric 5: Passive Voice Ratio

**Definition:** The percentage of sentences utilizing passive voice construction.

**Relevance:**
- Stylistic fingerprint. Scientific humans prefer Passive; Journalism prefers Active.
- AI generally defaults to Active voice unless prompted otherwise.

**Detection Logic:**
Locate dependency tag `nsubjpass` (nominal subject passive) + `auxpass`.

---

# Metric 5: Calculation Example

**Check:**
$$\text{Ratio} = \frac{\text{Count}(\text{Passive Sentences})}{\text{Total Sentences}}$$

**Example:**
> 1. "The decision **was made** by the committee." ($\checkmark$ Passive)
> 2. "The committee made the decision." ($\times$ Active)

---

# Metric 6: S2S Cosine Similarity

**Definition:** *Sentence-to-Sentence Semantic Similarity*. Using Sentence-Transformers (Embeddings) to measure semantic overlap between adjacent sentences.

**Relevance:**
- AI optimizes for "coherence," leading to high similarity (repetitiveness).
- Humans make "semantic jumps" (introducing new ideas).

**Formula:**
$$S_{sim} = \cos(\vec{v}_n, \vec{v}_{n+1}) = \frac{\vec{v}_n \cdot \vec{v}_{n+1}}{\|\vec{v}_n\| \|\vec{v}_{n+1}\|}$$

---

# Methodology Upgrade: From R to Python

**The Original Baseline:**
- The IRAL paper relied on **R** (R Studio, `quanteda` package).
- **Limitation:** Often designed for smaller, static datasets; harder to integrate into real-time production pipelines.

**Our Contribution (SC203):**
- **Complete Reimplementation:** We ported the statistical logic to **Python**.
- **Tech Stack:**
  - **Logic:** `iral_lexical.py` for Log-Odds and collocation extraction.
  - **Orchestration:** `iral_orchestrator.py` for batch processing 58k samples.
  - **Visualization:** `iral_plots.py` for automated figure generation.

---

# Inside `iral_lexical.py`: The Math

We implemented the **Log-Odds Ratio with Informative Dirichlet Prior** directly in Python.

**The Algorithm:**
1.  **Tokenization:** Clean tokens using the shared `spacy` pipeline.
2.  **Counting:** Efficient frequency counts for Human corpus ($y_{human}$) vs. AI corpus ($y_{ai}$).
3.  **Smoothing:** Apply statistical smoothing to handle zero-frequency words.
4.  **Z-Score Calculation:** Compute the z-score for each word to determine significance ($z > 1.96$).

> **Result:** A statistically robust list of "Giveaway Words" generated automatically for every model.

---

# Inside `iral_plots.py`: Automated Insights

Instead of manual plotting in R Studio, our pipeline automatically generates:

1.  **Log-Odds Charts:** Visualizing the "fight" between Human words (negative) and AI words (positive).
2.  **Collocation Clouds:** Extracting bigrams (e.g., "climate change", "vital role") that appear significantly more often in AI text.

**Outcome:**
We moved from "analyzing a CSV" to a **push-button explainability engine** that instantly visualizes the linguistic divergence of any new model we test.

---

# IRAL Lexical Analysis (Log-Odds)

Inspired by Zhang & Crosthwaite (2025), we identify "giveaway" words.

**Method (Log-Odds Ratio):**
$$\text{Log Odds} = \ln \left( \frac{\text{Freq}(W)_{\text{AI}} + 0.5}{\text{Freq}(W)_{\text{Human}} + 0.5} \right)$$

**Interpretation:**
- **Positive Score:** Strongly associated with AI.
- **Negative Score:** Strongly associated with Human.

---

# Lexical Findings (Hypothesized)

Based on IRAL literature:

| Category | Human Words | AI Words |
| :--- | :--- | :--- |
| **Themes** | *Leaders, Food, Career, Youtube* | *Sustainable, Educational, Technical* |
| **Style** | *Said, Reported, Years* | *Delve, Landscape, Crucial, Pivotal* |
| **Type** | Concrete Entities | Abstract Concepts |

---

# Preliminary Findings: Complexity

**Hypothesis:**
Human text (NYT) will exhibit higher **Standard Deviation** in sentence length compared to AI.

**Evidence:**
- Desaire et al. found humans vary sentence length significantly more than AI.
- **Why?** AI generates tokens based on probability, favoring "average" sentence structures. Humans write for impact (Burstiness).

---

# Preliminary Findings: Hedging

**Hypothesis:**
Humans will have a higher **Modal/Epistemic Rate**.

**Evidence:**
- Humans use "maybe", "suggest", "however" to denote scientific or journalistic caution.
- AI outputs are often designed to be helpful and authoritative, reducing uncertainty markers.

---

# Statistical Significance

To validate these metrics, we use **Welch's t-test** and **Cohen's d**.

**Cohen's d Interpretation:**
- $d = 0.2$: Small effect
- $d = 0.5$: Medium effect
- $d > 0.8$: **Large effect** (Strong predictor for classification)

*We expect **S2S Similarity** and **Clause Complexity** to show Large Effects.*

---

# Discussion: The "Human Fingerprint"

What makes text "Human"?

1.  **Inconsistency:** We are "messy." We mix 5-word sentences with 50-word sentences.
2.  **Uncertainty:** We use hedging to show nuance.
3.  **Specificity:** We reference real-world entities (*Japan, YouTube*) more than abstract categories (*"Social Media"*).

---

# The "Hidden" Challenge: Data Hygiene

**Reality:** The raw dataset was not analysis-ready.
- **Source:** Scraped web data (PDF-to-Text artifacts).
- **Noise:**
  - "Page X" headers and footers breaking sentence flow.
  - Markdown symbols (`#`, `*`) and "source: [x]" citations embedded in text.
  - Non-narrative paragraphs (e.g., copyright notices).

> **Impact:** Raw noise artificially inflates "complexity" metrics, distorting the baseline for Human vs. AI comparison.

---

# Engineering Solution: Robust Ingestion

We implemented a custom sanitization module (`src.ingest`) to recover pure text.

**Key Cleaning Steps:**
1.  **Regex Filtration:** Removing PDF artifacts (e.g., `^source: \d+`, `--- PAGE \d+ ---`).
2.  **Structural Cleaning:** Stripping Markdown formatting to isolate pure prose.
3.  **Noise Rejection:** Discarding rows where text is too short or clearly navigational.
4.  **Column Alignment:** Ensuring `human_story` and `model_output` align perfectly for paired T-tests.

---

# Pipeline Architecture (Modular Design)

Our `src` codebase is built for reproducibility and scalability:

- **`ingest.py`**:  Sanitizes raw CSVs and aligns columns.
- **`parse_and_cache.py`**:  Runs `spaCy` NLP processing once and caches to disk (Parquet).
- **`metrics_core.py`**:  Stateless functions to calculate the 6 linguistic features.
- **`stats_analysis.py`**:  Automated Welch’s t-tests and Cohen’s d calculation.

> **Why this matters:** Modular design allowed us to rapidly swap out metrics and upgrade the IRAL component without breaking the ingestion logic.

---

# Limitations

1.  **Genre Bias:** Our baseline is Journalism (NYT). Scientific papers might differ (e.g., higher passive voice).
2.  **Model Evolution:** GPT-4o is better at mimicking human variance than older models.
3.  **Prompt Sensitivity:** AI style changes heavily based on the prompt (e.g., "Write like a scientist").

---

# Conclusion & Next Steps

1.  **Conclusion:** A "White Box" pipeline using 6 linguistic metrics offers a transparent, effective baseline for AI detection.
2.  **Next Step:** Apply the pipeline to the full 58k dataset.
3.  **Final Output:** Train a lightweight classifier (XGBoost) on these 6 features to achieve >95% accuracy.

---

# References

1.  **Roy et al. (2025).** *A Comprehensive Dataset for Human vs. AI Generated Text Detection.*
2.  **Zhang & Crosthwaite (2025).** *More human than human? Differences in lexis...* IRAL.
3.  **Desaire et al. (2023).** *Distinguishing academic science writing from humans or ChatGPT...* Cell Reports Physical Science.
4.  **Project Code:** `AIvsHuman` Pipeline README.

---

# Thank You
## Questions?

**SC203 Research Team**