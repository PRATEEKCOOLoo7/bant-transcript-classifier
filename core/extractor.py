"""BANT extraction from sales call transcripts.

Extracts Budget, Authority, Need, and Timeline signals from
conversation text using pattern matching and contextual scoring.
Each dimension gets a confidence score based on explicitness,
specificity, and recency of the signal within the transcript.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class BANTSignal:
    dimension: str
    qualified: bool
    confidence: float
    evidence: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class BANTResult:
    budget: BANTSignal
    authority: BANTSignal
    need: BANTSignal
    timeline: BANTSignal
    overall_score: float = 0.0
    qualified: bool = False

    def __post_init__(self):
        scores = [self.budget.confidence, self.authority.confidence,
                  self.need.confidence, self.timeline.confidence]
        self.overall_score = round(sum(scores) / 4, 3)
        qualified_count = sum(1 for s in [self.budget, self.authority, self.need, self.timeline] if s.qualified)
        self.qualified = qualified_count >= 3


BUDGET_PATTERNS = [
    (r"\$\s*[\d,.]+\s*(?:k|m|million|thousand|billion)?", 0.9, "explicit_amount"),
    (r"budget\s+(?:is|of|around|approximately)\s+", 0.8, "budget_statement"),
    (r"(?:allocated|earmarked|set aside)\s+(?:funds|budget|money)", 0.7, "allocation"),
    (r"(?:can|willing to)\s+(?:spend|invest|pay)\s+", 0.7, "willingness"),
    (r"(?:no|zero|don'?t have)\s+budget", -0.5, "no_budget"),
    (r"budget\s+(?:constraints?|limitations?|tight)", -0.3, "budget_tight"),
    (r"(?:pricing|cost|how much)", 0.4, "price_inquiry"),
    (r"(?:approval|approve)\s+(?:up to|for)\s+\$", 0.8, "approval_threshold"),
]

AUTHORITY_PATTERNS = [
    (r"(?:i|we)\s+(?:make|have)\s+(?:the\s+)?(?:final\s+)?decision", 0.9, "decision_maker"),
    (r"(?:i'?m|i am)\s+the\s+(?:one|person)\s+who\s+(?:decides|approves)", 0.9, "self_identified"),
    (r"(?:need to|have to|must)\s+(?:run|check|get approval)\s+(?:by|with|from)", -0.3, "needs_approval"),
    (r"(?:my|our)\s+(?:cto|ceo|vp|director|manager)\s+(?:needs to|will|has to)", -0.2, "escalation_needed"),
    (r"(?:committee|board)\s+(?:decision|approval|review)", -0.3, "committee"),
    (r"(?:i can|authority to)\s+(?:sign|approve|authorize)", 0.8, "signing_authority"),
    (r"(?:i'?ll|i will)\s+(?:champion|push for|advocate)", 0.6, "internal_champion"),
]

NEED_PATTERNS = [
    (r"(?:pain point|challenge|problem|struggle|issue)\s+(?:is|with|for)", 0.8, "pain_stated"),
    (r"(?:losing|wasting|spending)\s+\d+\s+(?:hours|days|minutes)", 0.9, "quantified_pain"),
    (r"(?:manual|manually)\s+(?:process|doing|handling)", 0.7, "manual_process"),
    (r"(?:need|require|looking for|want)\s+(?:a |an |to )?(?:solution|tool|platform)", 0.7, "solution_seeking"),
    (r"(?:currently|right now)\s+(?:using|we use|our)\s+", 0.5, "current_solution"),
    (r"(?:not a priority|don'?t really need|no (?:real )?need)", -0.5, "no_need"),
    (r"(?:happy|satisfied|content)\s+with\s+(?:current|existing|what we have)", -0.4, "satisfied"),
    (r"(?:compliance|regulatory|audit)\s+(?:requirement|mandate|pressure)", 0.8, "compliance_need"),
]

TIMELINE_PATTERNS = [
    (r"(?:by|before|within)\s+(?:q[1-4]|january|february|march|april|may|june|july|august|september|october|november|december)", 0.8, "specific_date"),
    (r"(?:this|next)\s+(?:quarter|month|week|year)", 0.7, "relative_timeline"),
    (r"(?:asap|urgent|immediately|right away)", 0.9, "urgent"),
    (r"(?:no rush|whenever|no timeline|eventually)", -0.3, "no_urgency"),
    (r"(?:contract|renewal)\s+(?:expires?|ends?|up)\s+(?:in|on)", 0.8, "contract_deadline"),
    (r"(?:fiscal year|fy)\s+(?:end|budget cycle)", 0.6, "fiscal_trigger"),
    (r"(?:board|executive)\s+(?:review|meeting|presentation)\s+(?:in|on|by)", 0.7, "executive_deadline"),
    (r"(?:not|don'?t)\s+(?:have|see)\s+(?:a )?timeline", -0.4, "no_timeline"),
]


class BANTExtractor:
    """Extracts BANT signals from sales call transcript text."""

    def extract(self, transcript: str) -> BANTResult:
        lower = transcript.lower()
        chunks = self._chunk_by_turns(transcript)

        budget = self._score_dimension("budget", lower, chunks, BUDGET_PATTERNS)
        authority = self._score_dimension("authority", lower, chunks, AUTHORITY_PATTERNS)
        need = self._score_dimension("need", lower, chunks, NEED_PATTERNS)
        timeline = self._score_dimension("timeline", lower, chunks, TIMELINE_PATTERNS)

        return BANTResult(budget=budget, authority=authority, need=need, timeline=timeline)

    def _score_dimension(self, name: str, text: str,
                         chunks: list[str], patterns: list[tuple]) -> BANTSignal:
        evidence = []
        total_weight = 0.0
        match_count = 0

        for pattern, weight, label in patterns:
            matches = re.findall(pattern, text)
            if matches:
                match_count += 1
                total_weight += weight
                for m in matches[:2]:  # cap at 2 evidence items per pattern
                    snippet = m.strip() if isinstance(m, str) else str(m)
                    evidence.append(f"[{label}] ...{snippet}...")

        # Normalize to 0-1 confidence
        if match_count == 0:
            confidence = 0.0
        else:
            raw = total_weight / match_count
            confidence = max(0.0, min(1.0, (raw + 1) / 2))  # map -1..1 to 0..1

        qualified = confidence >= 0.5

        # Generate summary
        if not evidence:
            summary = f"No {name} signals detected"
        elif qualified:
            summary = f"{name.capitalize()} signals detected with {confidence:.0%} confidence"
        else:
            summary = f"Weak/negative {name} signals ({confidence:.0%} confidence)"

        return BANTSignal(
            dimension=name, qualified=qualified,
            confidence=round(confidence, 3),
            evidence=evidence[:5], summary=summary,
        )

    @staticmethod
    def _chunk_by_turns(transcript: str) -> list[str]:
        """Split transcript by speaker turns."""
        turns = re.split(r'\n(?=(?:Rep|Prospect|Speaker|Customer|Sales)\s*:)', transcript)
        return [t.strip() for t in turns if t.strip()]


@dataclass
class Objection:
    category: str
    subcategory: str
    severity: str  # low, medium, high, deal_breaker
    quote: str
    suggested_rebuttal: str


OBJECTION_PATTERNS = {
    "pricing": {
        "too_expensive": (r"(?:too (?:expensive|costly|much)|can'?t afford|out of (?:our )?(?:budget|range))", "medium"),
        "competitor_cheaper": (r"(?:competitor|alternative|other vendor)\s+(?:is |offers? )?(?:cheaper|less expensive|lower price)", "high"),
    },
    "timing": {
        "not_right_now": (r"(?:not (?:the )?right (?:time|now)|bad timing|too (?:early|soon))", "medium"),
        "locked_in": (r"(?:locked in|under contract|committed to)\s+(?:current|existing|another)", "high"),
    },
    "competition": {
        "using_competitor": (r"(?:already|currently)\s+(?:using|with|have)\s+(?:a |an )?(?:competitor|other vendor|different solution)", "medium"),
        "evaluating_others": (r"(?:evaluating|looking at|considering)\s+(?:other|several|multiple)\s+(?:options|vendors|solutions)", "low"),
    },
    "authority": {
        "need_approval": (r"(?:need to|have to)\s+(?:get|check with|run (?:it )?by)\s+(?:my |our )?(?:boss|manager|team|cto|ceo)", "medium"),
        "committee_decision": (r"(?:committee|board|group)\s+(?:decision|consensus|approval)", "high"),
    },
}

REBUTTALS = {
    "too_expensive": "Can I walk you through the ROI analysis? Most clients see payback within 6 months.",
    "competitor_cheaper": "I understand price matters. Where we differentiate is [specific value prop]. Can I show you a comparison?",
    "not_right_now": "I hear you. What would need to change for the timing to be right? Let me stay in touch.",
    "locked_in": "When does that contract come up for renewal? We can prepare a transition plan ahead of time.",
    "using_competitor": "That's common among our current clients too. What made you start looking at alternatives?",
    "evaluating_others": "Smart approach. What criteria are most important in your evaluation?",
    "need_approval": "Totally understand. How can I help you build the internal case? I can provide materials for your stakeholders.",
    "committee_decision": "We've worked with committee-driven organizations before. I can provide a presentation deck for the group.",
}


class ObjectionClassifier:
    """Classifies objections from transcript text."""

    def classify(self, transcript: str) -> list[Objection]:
        lower = transcript.lower()
        objections = []

        for category, subcats in OBJECTION_PATTERNS.items():
            for subcat, (pattern, severity) in subcats.items():
                matches = re.findall(pattern, lower)
                if matches:
                    quote = matches[0] if isinstance(matches[0], str) else str(matches[0])
                    objections.append(Objection(
                        category=category, subcategory=subcat,
                        severity=severity,
                        quote=quote[:100],
                        suggested_rebuttal=REBUTTALS.get(subcat, "Acknowledge and explore further."),
                    ))

        return objections
