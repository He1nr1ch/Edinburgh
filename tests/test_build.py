"""Build identity, Dublin's scheme: stamped in the image, 1.dev from a checkout."""

from __future__ import annotations

import importlib
import re

import pytest


def _fresh(monkeypatch, **env):
    for k in ("EDINBURGH_BUILD", "EDINBURGH_GIT_SHA"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import tradingagents.build as b

    return importlib.reload(b)


@pytest.mark.unit
def test_a_stamped_image_names_its_build_and_sha(monkeypatch):
    b = _fresh(monkeypatch, EDINBURGH_BUILD="1.2026.11.09.15", EDINBURGH_GIT_SHA="ca1c4d1")
    assert b.IS_BUILD and b.BUILD_VERSION == "1.2026.11.09.15"
    assert b.describe() == "1.2026.11.09.15 (ca1c4d1)"
    assert b.log_banner("worker start") == "Edinburgh build 1.2026.11.09.15 (ca1c4d1) — worker start"


@pytest.mark.unit
def test_a_checkout_is_not_a_build_and_says_so(monkeypatch):
    b = _fresh(monkeypatch)
    assert not b.IS_BUILD and b.BUILD_VERSION == "1.dev"
    text = b.describe()
    assert text.startswith("1.dev") and text.endswith("[checkout, not a build]")
    assert re.search(r"\(([0-9a-f]{7,})(-dirty)?\)", text)      # the sha from git, flagged if the tree is dirty


@pytest.mark.unit
def test_the_deploy_script_uses_dublins_scheme_with_major_one():
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "deploy.sh"
    text = script.read_text()
    assert 'BUILD_VERSION="1.$(date +%Y.%m.%d.%H)"' in text
    assert '--build-arg BUILD_VERSION="$BUILD_VERSION"' in text and '--build-arg GIT_SHA="$GIT_SHA"' in text
    assert "docker pull $IMAGE:$BUILD_VERSION" in text and "$IMAGE:$BUILD_VERSION worker" in text
    assert "$IMAGE:$GIT_SHA" not in text          # the sha rides inside the image, not as a tag
    docker = (script.parent / "Dockerfile").read_text()
    assert "ENV EDINBURGH_BUILD=$BUILD_VERSION" in docker and "ENV EDINBURGH_GIT_SHA=$GIT_SHA" in docker


@pytest.mark.unit
def test_the_page_and_health_name_the_build(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    _fresh(monkeypatch, EDINBURGH_BUILD="1.2026.11.09.15", EDINBURGH_GIT_SHA="ca1c4d1")
    from tradingagents.web import app as web

    monkeypatch.setattr(web, "_results_dir", lambda: tmp_path)
    monkeypatch.setattr(web.build, "BUILD_VERSION", "1.2026.11.09.15")
    monkeypatch.setattr(web.build, "describe", lambda: "1.2026.11.09.15 (ca1c4d1)")
    client = TestClient(web.app)
    assert client.get("/health").json()["build"] == "1.2026.11.09.15"
    assert "Edinburgh build 1.2026.11.09.15 (ca1c4d1)" in client.get("/days").text
