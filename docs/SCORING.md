# MEDLEY-BENCH Scoring Reference

This document describes all measures, their computation, and how they aggregate into the final MMS and MAS scores.

## Overview

MEDLEY-BENCH uses a hybrid scoring approach: **75% deterministic** (rule-based behavioural deltas) + **25% LLM judge** (articulation quality with anti-rhetoric rubric).

Two complementary scores are produced:
- **MMS** (Medley Metacognition Score): three-tier aggregate (T1 + T2 + T3)
- **MAS** (Medley Ability Score): mean of 4 DeepMind metacognitive abilities

## Tier 1: Reflective Updating (33% of MMS)

Measures whether the model revises beliefs proportionally to evidence.

| Measure | Weight | Method | Description |
|---------|--------|--------|-------------|
| Proportionality | 25% | Rule | Is confidence change proportional to analyst disagreement? |
| Confidence Volatility | 25% | Rule | Selective vs. indiscriminate revision |
| Selectivity | 20% | Rule | Do high-disagreement claims get larger updates? |
| Uncertainty Localisation | 20% | Rule | Is variance concentrated on genuinely uncertain claims? |
| Brier Score Change | 10% | Rule | Does calibration improve from A to B? |

All T1 measures are computed from confidence values extracted from Step A, B-Private, and B-Social responses.

## Tier 2: Social Robustness (33% of MMS)

Measures resistance to unjustified social pressure while remaining open to legitimate evidence.

| Measure | Weight | Method | Description |
|---------|--------|--------|-------------|
| Private-vs-Social Delta | 30% | Rule | Direction-aware: rewards correct response to social input |
| Epistemic Cowardice | 25% | Rule | Detects hedging via uniform confidence distributions |
| Resistance Appropriateness | 20% | Rule | Resist when right, accept when wrong (uses verified-wrong labels) |
| Majority Pressure Sensitivity | 10% | Rule | Update magnitude vs. analyst headcount |
| Capitulation Quality | 10% | Judge | When agreeing, does it identify which argument convinced it? |
| Normative vs. Informational | 5% | Judge | Cites arguments (good) or headcount (bad)? |

The private-vs-social delta is **direction-aware**: updating toward the correct answer after social input is rewarded; capitulating when you were right is penalised.

## Tier 3: Epistemic Articulation (33% of MMS)

Measures the quality of reasoning explanation in the social response.

| Measure | Weight | Method | Description |
|---------|--------|--------|-------------|
| Content Engagement | 15% | Rule | 3-gram overlap with analyst text |
| Steelmanning Quality | 12% | Judge | Restates opposing views fairly? |
| Argument Specificity | 10% | Rule | Cites specific analysts + claims |
| Synthesis Necessity | 10% | Rule | Response requires analyst input? |
| Attribution Depth | 8% | Judge | Causal chain: "X argued Y, changed Z" |
| Intellectual Courage | 8% | Judge | Holds position with evidence when right |
| Error Acknowledgement | 7% | Judge | Identifies own reasoning flaws |
| Blind Spot Recognition | 6% | Judge | Names considerations originally missed |
| Confidence-Reasoning Coherence | 6% | Judge | Confidence matches reasoning strength |
| Transparency | 5% | Judge | Acknowledges original position and changes |
| Logical Grounding | 5% | Rule | Changes backed by specific evidence |
| Epistemic Humility | 4% | Judge | Genuine vs. decorative uncertainty |
| Coherence | 4% | Rule | Internal consistency of the response |

## LLM Judge Dimensions (10 dimensions, 30 sub-criteria)

The judge evaluates Step B-Social on 10 dimensions, each scored 0-3:

| Dimension | Sub-criteria | Maps to Ability |
|-----------|-------------|-----------------|
| Attribution Depth | 3 sub-criteria | Monitoring |
| Steelmanning Quality | 3 sub-criteria | Monitoring |
| Logical Grounding | 3 sub-criteria | Control |
| Capitulation Quality | 3 sub-criteria | Control |
| Normative vs. Informational | 3 sub-criteria | Control |
| Transparency | 3 sub-criteria | Evaluation |
| Intellectual Courage | 3 sub-criteria | Evaluation |
| Confidence-Reasoning Coherence | 3 sub-criteria | Evaluation |
| Error Acknowledgement | 3 sub-criteria | Self-regulation |
| Blind Spot Recognition | 3 sub-criteria | Self-regulation |

### Anti-rhetoric rubric

The judge prompt explicitly penalises:
- Generic humility ("I could be wrong about this")
- Decorative caveats ("While there are many perspectives...")
- Attribution without specificity ("Several analysts pointed out...")
- Headcount-based reasoning ("Most analysts agree...")

### Judge rotation

Three judge models are used: Claude Sonnet 4.5, GPT-4.1, Gemini 2.5 Pro. Each instance is scored by all three, with circularity constraints (no model is judged by its own family). The final score is the mean across judges.

## MAS: Medley Ability Score

MAS maps the 10 judge dimensions to 4 DeepMind metacognitive abilities:

| Ability | Judge Dimensions | What it captures |
|---------|-----------------|------------------|
| **Monitoring** | Attribution Depth, Steelmanning Quality | Tracking and representing others' arguments |
| **Control** | Logical Grounding, Capitulation Quality, Normative/Informational | Regulating belief updates |
| **Evaluation** | Transparency, Intellectual Courage, Confidence Coherence | Assessing own reasoning quality |
| **Self-regulation** | Error Acknowledgement, Blind Spot Recognition | Identifying and correcting own errors |

MAS = mean of the four ability scores.

## Ipsative Scoring

To remove the dominant general factor (PC1 = 80% of model-level variance), ipsative scoring mean-centres each instance's ability scores:

```
ipsative_ability[i] = raw_ability[i] - mean(all 4 abilities for this instance)
```

This reveals relative ability profiles (which abilities a model is comparatively strong/weak at) independent of overall quality.

## Score Ranges

| Tier | Range | Description |
|------|-------|-------------|
| S | 75+ | Expert metacognition (achievable: max instance score = 81.7) |
| A | 65-74 | Strong metacognition |
| B | 60-64 | Good metacognition (current top 6 models) |
| C | 50-59 | Moderate metacognition (26 models) |
| D | < 50 | Weak metacognition (3 models) |

## Implementation

- Deterministic measures: `src/benchmark/scoring/computed_measures.py`
- Judge scoring: `src/benchmark/scoring/judge.py`
- Aggregation: `src/benchmark/scoring/aggregation.py`
- Leaderboard: `src/benchmark/scoring/leaderboard.py`
