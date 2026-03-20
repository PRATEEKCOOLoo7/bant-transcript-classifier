import pytest
from core.extractor import BANTExtractor, ObjectionClassifier


class TestBANTExtractor:
    def setup_method(self):
        self.b = BANTExtractor()

    def test_detects_explicit_budget(self):
        r = self.b.extract("We've allocated $150K for this project. Budget is approved.")
        assert r.budget.qualified
        assert r.budget.confidence > 0.5

    def test_detects_no_budget(self):
        r = self.b.extract("We don't have any budget for this. Zero budget available.")
        assert not r.budget.qualified

    def test_detects_decision_maker(self):
        r = self.b.extract("I make the final decision on technology purchases.")
        assert r.authority.qualified

    def test_detects_needs_approval(self):
        r = self.b.extract("I need to run this by my CTO and get committee approval.")
        assert not r.authority.qualified

    def test_detects_pain_point(self):
        r = self.b.extract("Our biggest challenge is manual reporting. We're losing 6 hours per day on manual process.")
        assert r.need.qualified
        assert r.need.confidence > 0.5

    def test_detects_no_need(self):
        r = self.b.extract("We're happy with our current solution. Not a priority and no real need.")
        assert not r.need.qualified

    def test_detects_timeline(self):
        r = self.b.extract("We need this implemented by Q3. It's urgent — board review in July.")
        assert r.timeline.qualified

    def test_detects_no_timeline(self):
        r = self.b.extract("No rush. Eventually maybe. No timeline on this.")
        assert not r.timeline.qualified

    def test_strong_prospect_qualifies(self):
        r = self.b.extract(
            "Budget is $200K allocated this fiscal year. I make the final decision. "
            "We're losing 6 hours per week on manual process. Need this by Q3."
        )
        assert r.qualified
        assert r.overall_score > 0.5

    def test_weak_prospect_not_qualified(self):
        r = self.b.extract(
            "No budget. Need committee approval. Happy with current solution. No timeline."
        )
        assert not r.qualified

    def test_empty_transcript(self):
        r = self.b.extract("")
        assert not r.qualified
        assert r.overall_score == 0.0


class TestObjectionClassifier:
    def setup_method(self):
        self.o = ObjectionClassifier()

    def test_pricing_objection(self):
        objs = self.o.classify("This is too expensive for what it offers.")
        assert any(o.category == "pricing" for o in objs)

    def test_competitor_objection(self):
        objs = self.o.classify("We're currently using a competitor's solution.")
        assert any(o.category == "competition" for o in objs)

    def test_timing_objection(self):
        objs = self.o.classify("Not the right time for us. We're locked in with our current vendor.")
        assert any(o.category == "timing" for o in objs)

    def test_authority_objection(self):
        objs = self.o.classify("I need to check with my manager before proceeding.")
        assert any(o.category == "authority" for o in objs)

    def test_has_rebuttals(self):
        objs = self.o.classify("Too expensive and we need committee approval.")
        for o in objs:
            assert len(o.suggested_rebuttal) > 10

    def test_no_objections(self):
        objs = self.o.classify("Everything sounds great. Let's move forward.")
        assert len(objs) == 0
