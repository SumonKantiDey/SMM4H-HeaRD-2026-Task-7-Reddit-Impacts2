# SMM4H-HeaRD @ ACL-2026 Shared Task 7: Social and Clinical Impacts Span Detection

This repository focused on extracting self-reported clinical and social impacts of nonmedical opioid use from first-person social media text.

---

## Task Overview

The goal of this task is to identify and extract span boundaries for two types of impacts mentioned in first-person narratives.

- **ClinicalImpacts**  
  Physical or psychological consequences of substance use  
  Examples include overdose, withdrawal, depression, anxiety

- **SocialImpacts**  
  social, occupational, or relational consequences of substance use  
  Examples include job loss, legal charges, financial hardship

The task is formulated as a **token-level BIO sequence labeling problem**, where each token is assigned a tag indicating whether it begins, continues, or falls outside an impact span.

---

## Labeling Scheme

We use a standard **BIO tagging scheme**:

| Tag | Description |
|----|------------|
| B-ClinicalImpacts | Beginning of a clinical impact span |
| I-ClinicalImpacts | Inside a clinical impact span |
| B-SocialImpacts | Beginning of a social impact span |
| I-SocialImpacts | Inside a social impact span |
| O | Outside any impact span |

Only **first-person self-reported impacts** are annotated.

---

## Input and Output Format

Each instance consists of:
1. A tokenized input sentence
2. A BIO tag sequence of equal length

---

## Example

### Original Text

I am a recovering addict and I overdosed at 19 (I’m 25 now) and I was charged with disorderly conduct (only thing on my record).

---

### Input (Tokens)

```json
["I", "am", "a", "recovering", "addict", "and", "I", "overdosed", "at", "19", "(", "I", "’", "m", "25", "now", ")", "and", "I", "was", "charged", "with", "disorderly", "conduct", "(", "only", "thing", "on", "my", "record", ")", "."]
```

### Output (BIO Tags) 
```json
["O", "O", "O", "O", "B-ClinicalImpacts", "O", "O", "B-ClinicalImpacts", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "I-SocialImpacts", "O", "O", "O", "O", "O", "O", "O", "O"]
```


## Evaluation

Model performance is evaluated using span-level F1 scores:

**Strict F1**  
Requires exact matches of both span boundaries and labels.

**Relaxed F1**  
Allows partial overlap between predicted and gold spans.

Both metrics should be reported.

---

## Baseline Results

In benchmark experiments, a pretrained **DeBERTa-large** model achieved:

- **Relaxed F1**: 0.61

This result highlights the challenge of accurately detecting informal and implicitly described impact spans in social media text.

## Evaluation Script

We provide a reference evaluation script below: 

- `evaluation_script.py`

## Reference Paper

If you use this task, dataset, or evaluation setup in your work, please cite the following paper:

> **Dey, Sumon Kanti; Powell, Jeanne M.; Ismail, Azra; Perrone, Jeanmarie; Sarker, Abeed.**  
> *Inference Gap in Domain Expertise and Machine Intelligence in Named Entity Recognition: Creation of and Insights from a Substance Use-related Dataset.*  
> **Biocomputing 2026: Proceedings of the Pacific Symposium**, 2025.   
> [Paper link](https://arxiv.org/pdf/2508.19467)

```bibtex
@inproceedings{dey2025inference,
  title={Inference Gap in Domain Expertise and Machine Intelligence in Named Entity Recognition: Creation of and Insights from a Substance Use-related Dataset},
  author={Dey, Sumon Kanti and Powell, Jeanne M and Ismail, Azra and Perrone, Jeanmarie and Sarker, Abeed},
  booktitle={Biocomputing 2026: Proceedings of the Pacific Symposium},
  pages={12--26},
  year={2025},
  organization={World Scientific}
}
