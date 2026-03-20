"""BANT Transcript Classifier — Demo"""

import logging
from core.extractor import BANTExtractor, ObjectionClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s", datefmt="%H:%M:%S")

SAMPLE_TRANSCRIPT = """
Rep: Thanks for taking the time today, Sarah. Can you tell me about the challenges you're facing with your current analytics setup?

Prospect: Sure. The biggest pain point is manual reporting — our advisors spend about 6 hours per week per client building reports by hand. We're losing productivity.

Rep: That's significant. How are you handling that today?

Prospect: We're currently using a combination of Orion and Excel. It works but it's manual and error-prone. We need a solution that automates the reporting pipeline.

Rep: Makes sense. What kind of budget have you allocated for this initiative?

Prospect: We've set aside approximately $150K for technology improvements this fiscal year. Our CTO has authority to approve up to $200K without board approval.

Rep: Great. And in terms of timeline, when are you looking to have something in place?

Prospect: We need this before Q3 board review — so ideally implemented by July. It's becoming urgent because we're onboarding 15 new clients next month.

Rep: Understood. Have you looked at other solutions?

Prospect: Yes, we're evaluating a few options. Your competitor DataViz Pro is cheaper but we've heard mixed reviews about their support. We need to run this by our CTO but I'm the one championing it internally.
"""

ADVERSARIAL_TRANSCRIPT = """
Rep: Hi there. Can I ask about your budget situation?

Prospect: We don't have any budget for this. Zero. Not a priority at all. We're happy with what we have right now.

Rep: I understand. Is there a timeline you're considering?

Prospect: No timeline. Eventually maybe but not now. Not the right time. I don't even make these decisions — you'd need to talk to the committee and they only meet annually.

Rep: What about the challenges you mentioned earlier?

Prospect: Honestly there's no real need. Our current solution works fine. We're locked in with our current vendor for another 2 years anyway. This is too expensive for what it is.
"""


def main():
    bant = BANTExtractor()
    objections = ObjectionClassifier()

    print(f"\n{'='*65}")
    print("  BANT Transcript Classifier — Demo")
    print(f"{'='*65}")

    for name, transcript in [("Strong Prospect", SAMPLE_TRANSCRIPT), ("Weak Prospect", ADVERSARIAL_TRANSCRIPT)]:
        print(f"\n{'─'*65}")
        print(f"  {name}")
        print(f"{'─'*65}")

        result = bant.extract(transcript)
        qual = "QUALIFIED" if result.qualified else "NOT QUALIFIED"
        print(f"\n  Overall: {result.overall_score:.0%} ({qual})")

        for dim in [result.budget, result.authority, result.need, result.timeline]:
            icon = "✓" if dim.qualified else "✗"
            print(f"    {icon} {dim.dimension.upper():10s} {dim.confidence:.0%} — {dim.summary}")
            for ev in dim.evidence[:2]:
                print(f"        {ev[:70]}")

        objs = objections.classify(transcript)
        if objs:
            print(f"\n  Objections ({len(objs)}):")
            for o in objs:
                print(f"    [{o.severity}] {o.category}/{o.subcategory}: \"{o.quote[:50]}\"")
                print(f"      Rebuttal: {o.suggested_rebuttal[:70]}")

    print(f"\n{'='*65}")
    print("  Demo complete.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
