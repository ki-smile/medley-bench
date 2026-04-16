# MEDLEY-BENCH Kaggle Notebooks

Kaggle-ready notebooks for the [Measuring Progress Toward AGI](https://www.kaggle.com/competitions/measuring-progress-toward-agi-cognitive-abilities) competition (Metacognition Track).

## Notebooks

| Notebook | Returns | Description |
|----------|---------|-------------|
| `medley_bench_normal.py` | **MMS** (0-1) | Medley Metacognition Score: T1/T2/T3 tier aggregate |
| `medley_bench_abilities.py` | **MAS** (0-1) | Medley Ability Score: mean of 4 DeepMind abilities |

Both run the same three-step protocol (Solo → Self-Revision → Social) on 130 instances. They differ only in the final aggregation.

## Kaggle Benchmark Toolbox Limitations

The Kaggle `kbench` framework imposes several constraints that affect scoring fidelity compared to the local benchmark:

### 1. Binary judge (pass/fail) instead of graded scoring (0-3)

The `kbench.assertions.assess_response_with_judge()` API returns only binary pass/fail per criterion. Our 30 sub-criteria (10 dimensions × 3 sub-criteria each) are scored as `1.0` if passed or `0.0` if failed, then averaged per dimension. The local benchmark uses a graded 0-3 scale with three independent judges, producing substantially finer-grained T3 scores.

**Impact:** T3 (Epistemic Articulation) scores are compressed on Kaggle. A response scoring 2/3 ("adequate with some specifics") locally is rounded to either 0 or 1 on Kaggle, losing the middle ground. This produces a consistent **+2-4 point offset** between Kaggle and local MMS scores. Rankings are preserved (Spearman ρ > 0.97).

### 2. Single judge model instead of 3-judge rotation

Kaggle uses a single judge (`kbench.judge_llm`), while the local benchmark rotates three judges (Claude Sonnet 4.5, GPT-4.1, Gemini 2.5 Pro) with circularity constraints (no model judged by its own family). The single-judge setup may introduce systematic bias if the judge model favours certain articulation styles.

### 3. No judge family exclusion

On Kaggle, the judge model is fixed by the platform. If the judge happens to be from the same family as the model being evaluated, within-family bias cannot be eliminated. The local benchmark explicitly prevents this.

### 4. No multi-judge median

The local benchmark takes the median across three judges for robustness. Kaggle's single-judge output has higher variance.

### Why rankings are still valid

These limitations primarily affect T3 (25% of MMS through judge-dependent measures). T1 (Reflective Updating) and T2 (Social Robustness) are **75% deterministic** — computed from confidence deltas, proportionality, selectivity, and other rule-based measures that are identical on both platforms. This deterministic majority ensures that Kaggle rankings correlate strongly with local rankings despite the judge differences.

### Recommendation

For research purposes, use the **local benchmark** (`medley-bench benchmark --track metacognition`) which provides:
- Graded 0-3 judge scoring with 30 sub-criteria
- 3-judge circularity-aware rotation
- Progressive adversarial mode
- Ipsative ability profiling
- Full raw response logging

The Kaggle notebooks are designed for competition compatibility and quick model comparison.

## Dataset

Both notebooks require the [medley-bench-data](https://www.kaggle.com/datasets/farhadabtahi/medley-bench-data) dataset attached as a Kaggle data source.

## Usage

1. Create a new Kaggle notebook
2. Add `farhadabtahi/medley-bench-data` as a data source
3. Copy the notebook content
4. Run
