"""Tests for core parsing utilities."""
import pytest
from src.core.parsing import (
    conf_to_numeric,
    get_claim_conf,
    parse_json_response,
    extract_claim_ids,
    CONFIDENCE_MAP,
)


class TestConfToNumeric:
    def test_all_labels(self):
        for label, expected in CONFIDENCE_MAP.items():
            assert conf_to_numeric(label) == expected

    def test_case_insensitive(self):
        assert conf_to_numeric("Very_High") == 0.95
        assert conf_to_numeric("LOW") == 0.35

    def test_unknown_defaults_moderate(self):
        assert conf_to_numeric("unknown") == 0.55

    def test_fuzzy_truncated(self):
        """Models sometimes truncate labels."""
        assert conf_to_numeric("ve") == 0.95
        assert conf_to_numeric("very high") == 0.95
        assert conf_to_numeric("VH") == 0.95
        assert conf_to_numeric("mod") == 0.55
        assert conf_to_numeric("medium") == 0.55
        assert conf_to_numeric("vl") == 0.15
        assert conf_to_numeric("very low") == 0.15

    def test_with_percentages(self):
        """Models sometimes include the scale description."""
        assert conf_to_numeric("high (70-89%)") == 0.80
        assert conf_to_numeric("very_low (0-29%)") == 0.15

    def test_prefix_match(self):
        """Partial prefix should match."""
        assert conf_to_numeric("very_h") == 0.95
        assert conf_to_numeric("very_lo") == 0.15


class TestGetClaimConf:
    def test_by_claim_id(self):
        resp = {"claim_level_assessments": [
            {"claim_id": "C1", "confidence": "high"},
            {"claim_id": "C2", "confidence": "low"},
        ]}
        assert get_claim_conf(resp, claim_id="C1") == 0.80
        assert get_claim_conf(resp, claim_id="C2") == 0.35

    def test_numeric_confidence(self):
        resp = {"claim_level_assessments": [
            {"claim_id": "C1", "confidence": 0.75},
        ]}
        assert get_claim_conf(resp, claim_id="C1") == 0.75

    def test_missing_returns_none(self):
        resp = {"claim_level_assessments": []}
        assert get_claim_conf(resp, claim_id="C1") is None

    def test_empty_response(self):
        assert get_claim_conf({}, claim_id="C1") is None


class TestParseJsonResponse:
    def test_pure_json(self):
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_code_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_embedded_json(self):
        raw = 'Here is my response: {"key": "value"} and some more text'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_malformed_returns_error(self):
        result = parse_json_response("not json at all")
        assert result["_parse_error"] is True

    def test_empty_returns_error(self):
        result = parse_json_response("")
        assert result["_parse_error"] is True


class TestExtractClaimIds:
    def test_extracts_ids(self):
        resp = {"claim_level_assessments": [
            {"claim_id": "C1"}, {"claim_id": "C2"}, {"claim_id": "C3"},
        ]}
        assert extract_claim_ids(resp) == ["C1", "C2", "C3"]

    def test_empty(self):
        assert extract_claim_ids({}) == []
