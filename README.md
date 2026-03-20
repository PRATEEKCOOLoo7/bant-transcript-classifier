# BANT Transcript Classifier

Production NLP pipeline for automatically extracting BANT qualification signals (Budget, Authority, Need, Timeline) from sales call transcripts, classifying objections, and generating meeting prep briefs.

Turns a 45-minute call recording into a structured qualification scorecard in seconds.

## What It Does

```
Raw Call Transcript
    │
    ├──▶ BANT Extractor ───────▶ Structured qualification scores
    │    "They mentioned $200K      Budget: ✓ ($200K range)
    │     budget for Q3"            Authority: ✓ (VP-level decision maker)
    │                               Need: ✓ (manual process pain)
    │                               Timeline: ✓ (Q3 implementation)
    │
    ├──▶ Objection Classifier ─▶ Categorized objections with rebuttals
    │    "We're locked into our      Type: competitor_lock_in
    │     current vendor until..."   Severity: medium
    │                                Suggested rebuttal: [template]
    │
    ├──▶ Sentiment Tracker ────▶ Per-topic sentiment over call duration
    │                               Pricing discussion: negative → neutral
    │                               Product demo: neutral → positive
    │
    └──▶ Meeting Prep Generator ▶ AI-generated brief for next call
                                    Key topics to revisit: [...]
                                    Open objections: [...]
                                    Recommended approach: [...]
```

## BANT Extraction

Not keyword matching — contextual understanding of qualification signals:

| Signal | What We Detect | Example |
|---|---|---|
| **Budget** | Explicit amounts, budget cycles, approval thresholds, "no budget" signals | "We've allocated around 200K for this initiative" → Budget: $200K, Confidence: 0.9 |
| **Authority** | Decision maker identification, approval chains, stakeholder mapping | "I'll need to run this by our CTO, but I have authority up to 100K" → Authority: Partial, Escalation: CTO |
| **Need** | Pain points, current process gaps, impact statements, urgency signals | "We're losing 3 hours per rep per day on manual data entry" → Need: High, Pain: Productivity |
| **Timeline** | Implementation windows, contract dates, fiscal year triggers, urgency | "We need this live before Q3 board review" → Timeline: Q3, Urgency: High |

### Confidence Scoring

Each BANT dimension gets a confidence score (0-1) based on:
- **Explicitness**: Did they state it directly or was it inferred?
- **Specificity**: "$200K budget" (high) vs "we have some budget" (low)
- **Recency**: Said in the last 5 minutes (high) vs said once 30 minutes ago (low)
- **Contradiction**: Did they contradict themselves? ("We have budget" → later → "budget is tight")

## Objection Classification

### Objection Taxonomy

```
objections/
├── pricing/
│   ├── too_expensive
│   ├── no_budget_now
│   ├── competitor_cheaper
│   └── need_discount
├── timing/
│   ├── not_right_now
│   ├── end_of_contract
│   └── need_more_time
├── competition/
│   ├── using_competitor
│   ├── locked_in_contract
│   └── evaluating_others
├── authority/
│   ├── need_approval
│   ├── not_decision_maker
│   └── committee_decision
├── need/
│   ├── no_pain_point
│   ├── current_solution_works
│   └── not_priority
└── technical/
    ├── integration_concerns
    ├── security_requirements
    └── scalability_doubts
```

Each objection gets:
- **Category + subcategory**: Structured classification
- **Severity**: low / medium / high / deal-breaker
- **Sentiment context**: Was it stated firmly or tentatively?
- **Suggested rebuttal**: Template from the rebuttal knowledge base
- **Follow-up action**: What the sales rep should do next

## Project Structure

```
bant-transcript-classifier/
├── README.md
├── requirements.txt
├── pipeline/
│   ├── __init__.py
│   ├── transcript_processor.py   # Chunking, speaker diarization, cleaning
│   ├── bant_extractor.py         # BANT signal extraction with confidence
│   ├── objection_classifier.py   # Multi-label objection classification
│   ├── sentiment_tracker.py      # Per-topic sentiment over time
│   └── meeting_prep.py           # AI-generated next-meeting brief
├── models/
│   ├── __init__.py
│   ├── bant_model.py             # Fine-tuned transformer for BANT extraction
│   ├── objection_model.py        # Multi-label classifier for objections
│   └── schemas.py                # Pydantic models for structured output
├── knowledge/
│   ├── objection_taxonomy.yaml   # Full objection classification tree
│   ├── rebuttal_templates.yaml   # Suggested rebuttals per objection type
│   └── bant_signals.yaml         # Signal patterns for each BANT dimension
├── training/
│   ├── prepare_bant_data.py      # Convert annotated transcripts to training data
│   ├── train_bant.py             # Fine-tune BANT extraction model
│   └── train_objection.py        # Train objection classifier
├── tests/
│   ├── test_bant_extraction.py
│   ├── test_objection_classifier.py
│   ├── test_meeting_prep.py
│   └── fixtures/
│       ├── sample_transcript_1.txt
│       └── sample_transcript_2.txt
└── examples/
    └── analyze_call.py
```

## Quick Start

```python
from pipeline import TranscriptAnalyzer

analyzer = TranscriptAnalyzer()

result = analyzer.analyze(
    transcript_path="transcripts/acme_discovery_call.txt",
    prospect_context={
        "company": "Acme Corp",
        "contact": "Sarah Chen",
        "stage": "discovery",
    },
)

# BANT scorecard
print(f"Overall BANT Score: {result.bant.overall_score:.0%}")
for dim in ["budget", "authority", "need", "timeline"]:
    signal = getattr(result.bant, dim)
    status = "✓" if signal.qualified else "✗"
    print(f"  {status} {dim.upper()}: {signal.summary} (confidence: {signal.confidence:.0%})")

# Objections
print(f"\nObjections Detected: {len(result.objections)}")
for obj in result.objections:
    print(f"  [{obj.severity}] {obj.category}/{obj.subcategory}: {obj.quote[:60]}...")
    print(f"    Rebuttal: {obj.suggested_rebuttal[:80]}...")

# Meeting prep
print(f"\nNext Meeting Brief:")
print(result.meeting_prep.brief)
```

## Model Architecture

### BANT Extraction
- **Base model**: Fine-tuned Mistral-7B with LoRA
- **Task**: Structured extraction — given transcript chunk, output JSON with BANT signals
- **Training data**: 2,000+ annotated sales call transcripts
- **Evaluation**: Precision/recall per BANT dimension, overall F1

### Objection Classification
- **Base model**: Fine-tuned DeBERTa-v3 (smaller, faster for classification)
- **Task**: Multi-label classification — one transcript chunk can contain multiple objection types
- **Labels**: 18 objection subcategories across 6 categories
- **Evaluation**: Macro F1, per-category precision/recall

## Design Decisions

- **Chunk-level extraction, not full-transcript**: Sales calls are 30-60 minutes. Processing the full transcript at once loses specificity. We chunk by speaker turn, extract signals per chunk, then aggregate.
- **Confidence scoring over binary**: "They have budget" (confidence: 0.9) and "they might have budget" (confidence: 0.4) require different sales strategies. The confidence score informs the next action.
- **Meeting prep as a first-class output**: Reps don't read raw BANT scorecards. They read briefs before their next call. The meeting prep generator turns structured data into actionable talking points.

