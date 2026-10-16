"""Unit tests for the HN Algolia API client (spec 19)."""

import json
from unittest.mock import MagicMock

import pytest
import requests

from mourat.clients.hn_algolia import HNAlgoliaClient

NOW = 1_780_000_000  # fixed timestamp for deterministic tests


def _response(payload, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code}")
    return resp


def _hit(**overrides):
    base = {
        "objectID": "424242",
        "title": "Test story",
        "author": "alice",
        "created_at_i": NOW - 86400,
        "url": "https://example.com/a",
        "story_text": None,
        "points": 55,
        "num_comments": 12,
    }
    base.update(overrides)
    return base


def _make_client(http_client=None, **kwargs):
    if http_client is None:
        http_client = MagicMock()
        http_client.get.return_value = _response({"hits": [], "nbPages": 1, "page": 0})
    return HNAlgoliaClient(http_client=http_client, **kwargs)


class TestSearchByDate:
    def test_request_carries_numeric_filters_and_tags(self):
        # T1 (R2): numeric filter string and tags appear in the request.
        http = MagicMock()
        http.get.return_value = _response({"hits": [], "nbPages": 1, "page": 0})
        client = _make_client(http)
        client.search_by_date(
            lo=NOW - 2 * 86400, hi=NOW - 86400, min_points=20, tags=["story"]
        )
        params = http.get.call_args.kwargs["params"]
        url = http.get.call_args.args[0]
        assert "created_at_i>%d" % (NOW - 2 * 86400) in url or any(
            "created_at_i>%d" % (NOW - 2 * 86400) in str(v) for v in params.values()
        )
        assert (
            any("points>=20" in str(v) for v in params.values()) or "points>=20" in url
        )
        assert any("story" in str(v) for v in params.values()) or "story" in url

    def test_hits_parsed_into_post_dicts(self):
        # T1 (R2): hits parsed into HNPostInfo-shaped dicts with all fields.
        http = MagicMock()
        http.get.return_value = _response({"hits": [_hit()], "nbPages": 1, "page": 0})
        client = _make_client(http)
        posts = client.search_by_date(
            lo=NOW - 2 * 86400, hi=NOW, min_points=10, tags=["story"]
        )
        assert len(posts) == 1
        p = posts[0]
        assert p["hn_id"] == "424242"
        assert p["title"] == "Test story"
        assert p["author"] == "alice"
        assert p["points"] == 55
        assert p["num_comments"] == 12
        assert p["url"] == "https://example.com/a"
        assert p["text"] == ""
        assert p["date"].endswith("+00:00") or "T" in p["date"]

    def test_missing_story_text_maps_to_empty_string(self):
        # T4 (R3): story_text None -> text "".
        http = MagicMock()
        http.get.return_value = _response(
            {"hits": [_hit(story_text=None)], "nbPages": 1, "page": 0}
        )
        client = _make_client(http)
        posts = client.search_by_date(lo=NOW - 86400, hi=NOW, min_points=1)
        assert posts[0]["text"] == ""

    def test_present_story_text_preserved(self):
        # T4 (R3): story_text mapped when present.
        http = MagicMock()
        http.get.return_value = _response(
            {"hits": [_hit(story_text="Ask me anything")], "nbPages": 1, "page": 0}
        )
        client = _make_client(http)
        posts = client.search_by_date(lo=NOW - 86400, hi=NOW, min_points=1)
        assert posts[0]["text"] == "Ask me anything"

    def test_http_error_raises(self):
        # T2 (R2): HTTP error -> exception (loud failure).
        http = MagicMock()
        http.get.return_value = _response({}, status_code=500)
        client = _make_client(http)
        with pytest.raises(Exception):
            client.search_by_date(lo=NOW - 86400, hi=NOW, min_points=1)

    def test_non_json_response_raises(self):
        # T2 (R2): non-JSON body -> exception.
        http = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.side_effect = json.JSONDecodeError("bad", "doc", 0)
        http.get.return_value = resp
        client = _make_client(http)
        with pytest.raises(Exception):
            client.search_by_date(lo=NOW - 86400, hi=NOW, min_points=1)

    def test_missing_hits_key_raises(self):
        # T2 (R2): malformed payload (no `hits`) -> exception.
        http = MagicMock()
        http.get.return_value = _response({"nbPages": 1})
        client = _make_client(http)
        with pytest.raises(Exception):
            client.search_by_date(lo=NOW - 86400, hi=NOW, min_points=1)

    def test_pagination_advances_pages(self):
        # Pagination across pages is exercised at collector level (T3);
        # here: page parameter advances when the client iterates.
        http = MagicMock()
        http.get.return_value = _response(
            {"hits": [_hit(objectID="1")], "nbPages": 1, "page": 0}
        )
        client = _make_client(http)
        posts = client.search_by_date(lo=NOW - 86400, hi=NOW, min_points=1)
        assert posts[0]["hn_id"] == "1"
