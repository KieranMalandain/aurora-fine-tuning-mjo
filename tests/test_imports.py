"""Guards the microsoft-aurora / aurora_mjo name collision.

microsoft-aurora owns the top-level `aurora` import name. A first-party
package called `aurora` would shadow it, surfacing as an ImportError on
`Batch`. See docs/campaigns/refactor/00_CONTEXT.md section 4.
"""

from __future__ import annotations

import aurora

import aurora_mjo


def test_third_party_aurora_resolves_to_site_packages() -> None:
    assert "site-packages" in aurora.__file__ or "dist-packages" in aurora.__file__


def test_first_party_package_is_not_shadowing() -> None:
    assert aurora_mjo.__file__ != aurora.__file__
    assert "aurora_mjo" in aurora_mjo.__file__


def test_aurora_batch_still_importable() -> None:
    from aurora import Batch, Metadata  # noqa: F401


def test_all_first_party_modules_import() -> None:
    from aurora_mjo import checkpoint, dataset, loss, model, trainer  # noqa: F401
