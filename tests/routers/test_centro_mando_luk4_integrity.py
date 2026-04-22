"""Regression for Phase 8 / Plan 08-05: Centro de Mando / luk4 refactor.

Locks (all are invariants; breaking any of them = regressing D-16):

1. GET `/` renders ``templates/luk4.html`` (Pitfall 10 invariant — the route
   does NOT render a hypothetical ``centro_mando.html``).
2. The 4 Alpine ``pabPage(...)`` roots are intact (one per pabellón) and the
   outer selector ``x-data="{ pabellon: 'p5' }"`` is present.
3. ``{% include '_partials/mapa_pabellon.html' %}`` appears 4 times (one per
   pabellón) — the LOCKED plano editor partial is still wired.
4. The outer chrome (topbar header + pabellón selector) uses only semantic
   tokens (``bg-surface-*`` / ``text-muted`` / ``text-body`` / ``border-subtle``
   / ``shadow-card``). Raw Tailwind state colors inside the ``pabPage()``
   ``zoneClass()`` / ``zoneDotClass()`` / ``zoneBadgeBorder()`` helpers are
   LEGACY-ALLOWED (D-16 LOCKED; feed the partial's dynamic ``:class=``
   bindings) and explicitly permitted.
5. ``showToast`` calls in luk4.html are 3-arg (Plan 08-02 invariant) and
   exactly 5 calls exist (producing / incidence / stopped / alarm / turno).
6. ``templates/_partials/mapa_pabellon.html`` Alpine directive counts are
   unchanged vs. the previous commit (grep-diff guard).

This suite intentionally runs the static checks (4, 5, 6) even when Postgres
is down so that CI without the DB still catches template drift. The runtime
route check (1, 2, 3) is gated behind a ``_postgres_reachable`` skip marker,
matching the pattern from ``tests/routers/test_bienvenida.py``.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, text

from nexo.data.engines import SessionLocalNexo
from nexo.data.models_nexo import (
    NexoDepartment,
    NexoLoginAttempt,
    NexoSession,
    NexoUser,
)
from nexo.services.auth import hash_password

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LUK4 = _REPO_ROOT / "templates" / "luk4.html"
_PARTIAL = _REPO_ROOT / "templates" / "_partials" / "mapa_pabellon.html"


# ── Static checks (no DB) ──────────────────────────────────────────────────


def test_luk4_template_extends_base_html() -> None:
    """luk4.html must extend base.html (Plan 08-02 chrome)."""
    src = _LUK4.read_text(encoding="utf-8")
    assert '{% extends "base.html" %}' in src, (
        "luk4.html must extend base.html so the topbar + drawer + toast "
        "chrome from Plan 08-02 applies."
    )


def test_luk4_template_has_four_pabpage_roots() -> None:
    """4 Alpine pabPage(...) roots (one per pabellón) — D-16 LOCKED."""
    src = _LUK4.read_text(encoding="utf-8")
    count = len(re.findall(r'x-data="pabPage\(', src))
    assert count == 4, (
        f"Expected 4 x-data=pabPage(...) roots (one per pabellón p5/p4/p3/p2), "
        f"found {count}. D-16 LOCKED interaction requires all four."
    )


def test_luk4_template_has_four_partial_includes() -> None:
    """4 includes of _partials/mapa_pabellon.html — one per pabellón."""
    src = _LUK4.read_text(encoding="utf-8")
    count = len(re.findall(r"{%\s*include\s*'_partials/mapa_pabellon\.html'\s*%}", src))
    assert count == 4, (
        f"Expected 4 {{% include '_partials/mapa_pabellon.html' %}}, found "
        f"{count}. The LOCKED plano editor must be wired for every pabellón."
    )


def test_luk4_outer_chrome_uses_semantic_tokens_only() -> None:
    """Outer chrome (topbar header + pabellón selector) uses semantic tokens.

    Raw state colors (``bg-red-###`` / ``bg-green-###`` / ``bg-yellow-###``
    / ``bg-blue-###``) ARE permitted inside the Alpine ``pabPage()``
    helpers (``zoneClass`` / ``zoneDotClass`` / ``zoneBadgeBorder``) because
    those strings feed the LOCKED partial's dynamic ``:class=`` bindings
    (D-16 LOCKED + UI-SPEC §Per-Screen Adaptations row luk4.html:
    "Zone editor CSS classes are LEGACY-ALLOWED during Mark-III").

    The OUTER chrome (HTML markup outside the ``<script>`` block) must be
    free of raw state colors — semantic ``.badge-*`` classes are the only
    permitted state chips.
    """
    src = _LUK4.read_text(encoding="utf-8")
    # Strip the <script>...</script> block — the zoneClass helpers live there.
    outer = re.sub(r"<script>.*?</script>", "", src, flags=re.DOTALL)
    raw_state = re.findall(r"bg-(red|green|blue|yellow)-[0-9]{3}", outer)
    assert not raw_state, (
        "luk4.html outer chrome must not use raw Tailwind state color "
        f"utilities; found: {raw_state}. Use semantic .badge-* classes."
    )


def test_luk4_outer_chrome_has_no_legacy_surface_utilities() -> None:
    """Outer chrome must have migrated off bg-white / text-gray-### / etc.

    The Alpine script block is exempt (it contains inline Chart.js color
    literals like ``#1a3a5c`` and zone-coloring helpers — all LOCKED).
    """
    src = _LUK4.read_text(encoding="utf-8")
    outer = re.sub(r"<script>.*?</script>", "", src, flags=re.DOTALL)
    forbidden = [
        r'class="[^"]*\bbg-white\b[^"]*"',
        r'class="[^"]*\btext-gray-\d{3}\b[^"]*"',
        r'class="[^"]*\btext-brand-\d{3}\b[^"]*"',
        r'class="[^"]*\bborder-surface-\d{3}\b[^"]*"',
    ]
    for pattern in forbidden:
        hits = re.findall(pattern, outer)
        assert not hits, (
            f"luk4.html outer chrome should not use legacy utility "
            f"{pattern!r}. Found: {hits}. Use semantic tokens "
            f"(bg-surface-base / text-muted / border-subtle)."
        )


def test_luk4_showtoast_calls_are_three_arg() -> None:
    """luk4.html uses 3-arg showToast(variant, title, msg) only.

    The legacy 2-arg pattern ``showToast('msg', 'error')`` was eliminated
    in Plan 08-02 (D-30).
    """
    src = _LUK4.read_text(encoding="utf-8")
    legacy = re.compile(
        r"\bshowToast\(\s*(['\"`])([^'\"`]+)\1\s*,\s*"
        r"(['\"])(error|info|success|warn)\3\s*\)"
    )
    match = legacy.search(src)
    assert match is None, (
        f"luk4.html must use 3-arg showToast(variant, title, msg) "
        f"exclusively. Found legacy 2-arg call: {match.group(0) if match else ''}"
    )


def test_luk4_showtoast_has_five_calls() -> None:
    """Exactly 5 showToast calls: producing / incidence / stopped / alarm / turno."""
    src = _LUK4.read_text(encoding="utf-8")
    count = len(re.findall(r"\bshowToast\(", src))
    assert count == 5, (
        f"Expected 5 showToast() calls in luk4.html (one per state-change "
        f"event: producing / incidence / stopped / alarm / turno). Found {count}."
    )


def test_mapa_pabellon_partial_alpine_directives_preserved() -> None:
    """Count of Alpine directives in the partial must match the last commit.

    D-16 LOCKED: the partial's interaction logic (Alpine component, editor
    buttons, zone rendering, drag/drop) is preserved verbatim. This test
    diffs the grep count vs. ``HEAD`` to catch accidental directive drift.
    """
    current_src = _PARTIAL.read_text(encoding="utf-8")
    current_count = len(re.findall(r"x-data|x-show|x-if|x-bind", current_src))

    # Compare against the previous git-committed version.
    try:
        head_blob = subprocess.check_output(
            ["git", "show", "HEAD:templates/_partials/mapa_pabellon.html"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
    except subprocess.CalledProcessError:
        pytest.skip(
            "Cannot run git show (not in a git working tree or partial not "
            "yet committed). Skipping directive drift check."
        )
        return

    head_count = len(re.findall(r"x-data|x-show|x-if|x-bind", head_blob))
    assert current_count == head_count, (
        f"_partials/mapa_pabellon.html Alpine directive count drifted: "
        f"HEAD has {head_count}, working tree has {current_count}. "
        f"D-16 LOCKED: the partial's interaction is preserved verbatim."
    )


def test_mapa_pabellon_partial_keeps_interaction_markers() -> None:
    """Sanity: key Alpine / include markers of the LOCKED partial are present."""
    src = _PARTIAL.read_text(encoding="utf-8")
    assert "x-show" in src, "partial must retain x-show directives"
    assert "openZone" in src, "partial must retain the openZone handler hook"
    assert "editMode" in src, "partial must retain the editor toggle"
    assert "editorSave" in src, "partial must retain the editor save handler"
    assert "zoneClass(hs)" in src, (
        "partial must still consume the zoneClass(hs) helper exported by pabPage() in luk4.html"
    )


# ── Integration (Postgres required) ────────────────────────────────────────


def _postgres_reachable() -> bool:
    try:
        db = SessionLocalNexo()
        try:
            db.execute(text("SELECT 1"))
            return True
        finally:
            db.close()
    except Exception:
        return False


_integration = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres no arriba: docker compose up -d db",
)


TEST_DOMAIN = "@luk4-integrity-test.local"
TEST_PASSWORD = "luk4integritytest12345"  # min 12 chars


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    from api.main import app

    with TestClient(app, follow_redirects=False) as c:
        yield c


@pytest.fixture(autouse=True)
def _cleanup() -> Iterator[None]:
    # Static tests (no-DB) don't need purge / rate-limit reset; skip DB touch
    # when Postgres is not reachable so the static regression still runs in CI.
    if not _postgres_reachable():
        yield
        return
    _purge()
    _reset_rate_limit()
    yield
    _purge()


def _reset_rate_limit() -> None:
    try:
        from api.rate_limit import limiter

        limiter.reset()
    except Exception:
        pass


def _purge() -> None:
    db = SessionLocalNexo()
    try:
        users = (
            db.execute(select(NexoUser).where(NexoUser.email.like(f"%{TEST_DOMAIN}")))
            .scalars()
            .all()
        )
        for u in users:
            db.execute(delete(NexoSession).where(NexoSession.user_id == u.id))
            db.delete(u)
        db.execute(delete(NexoLoginAttempt).where(NexoLoginAttempt.email.like(f"%{TEST_DOMAIN}")))
        db.commit()
    finally:
        db.close()


def _create_propietario(email: str) -> NexoUser:
    db = SessionLocalNexo()
    try:
        user = NexoUser(
            email=email,
            password_hash=hash_password(TEST_PASSWORD),
            role="propietario",
            active=True,
            must_change_password=False,
        )
        db.add(user)
        db.flush()
        depts = db.execute(select(NexoDepartment)).scalars().all()
        user.departments = list(depts)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def _login_propietario(c: TestClient, email: str) -> str:
    r = c.post(
        "/login",
        data={"email": email, "password": TEST_PASSWORD},
        headers={"Accept": "text/html"},
    )
    cookie = r.cookies.get("nexo_session", "")
    assert cookie, f"login failed: {r.status_code} {r.headers}"
    return cookie


@_integration
def test_root_route_renders_luk4_template(client: TestClient) -> None:
    """GET / as propietario renders luk4.html (Pitfall 10 invariant)."""
    email = f"propietario{TEST_DOMAIN}"
    _create_propietario(email)
    cookie = _login_propietario(client, email)

    resp = client.get("/", cookies={"nexo_session": cookie})
    assert resp.status_code == 200, f"GET / → {resp.status_code}"
    body = resp.text

    # Centro de Mando marker
    assert "Centro de Mando" in body, "response must contain the screen title"
    # 4 pabPage roots (one per pabellón) survived render
    assert body.count('x-data="pabPage(') == 4, (
        "all four pabPage() Alpine roots must be rendered in the response"
    )
    # pabellón selector outer root
    assert "x-data=\"{ pabellon: 'p5' }\"" in body, (
        "the pabellón selector's outer x-data must be rendered"
    )


@_integration
def test_root_route_includes_mapa_pabellon_markup(client: TestClient) -> None:
    """The LOCKED mapa_pabellon partial markup appears in the rendered HTML."""
    email = f"propietario{TEST_DOMAIN}"
    _create_propietario(email)
    cookie = _login_propietario(client, email)

    resp = client.get("/", cookies={"nexo_session": cookie})
    assert resp.status_code == 200
    body = resp.text
    # Markers unique to the partial
    assert "openZone(" in body, "partial must render — openZone handler wiring is missing"
    assert "editorSave" in body, "partial must render — editor save handler wiring is missing"
    assert "zoneClass(hs)" in body, "partial must bind :class=zoneClass(hs) on the zone overlays"
