#!/usr/bin/env python3
"""Extract /World/Anchorpoint frames for every asset in the retrieval pool.

Isaac Sim's modular props ship a single ``/World/Anchorpoint`` Xform whose
local transform is "the pose the next module's origin should take when
chained here". The asset's own origin (``/World``) is the implicit second
anchor.

Output is ``data/asset_anchors.json``.  The schema deliberately stores
the *raw* USD convention plus an empty ``aliases`` slot, so hand-author
overrides can layer on top without re-running this extractor losing them::

    {
      "ConveyorBelt_A14": {
        "anchors": {
          "origin":      {"position": [0, 0, 0],
                          "orient_wxyz": [1, 0, 0, 0]},
          "anchorpoint": {"position": [0, -3.9186, 0],
                          "orient_wxyz": [0, 0, 0, 1]}
        },
        "aliases": {}
      }
    }

This script does NOT need Isaac Sim — it uses ``usd-core`` only.  Run with
the project venv::

    .venv/bin/python scripts/extract_anchors.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from pxr import Usd

NUCLEUS_ASSET_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/"
)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
POOL_PATH = DATA_DIR / "retrieval_pool.json"
TAX_PATH = DATA_DIR / "asset_taxonomy.json"
OUT_PATH = DATA_DIR / "asset_anchors.json"
CACHE_DIR = REPO / ".usd_cache"


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    downloading {url}")
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        f.write(r.read())
    return dest


def _round(v: float, ndigits: int = 6) -> float:
    return round(float(v), ndigits)


def _read_anchorpoint(stage: Usd.Stage) -> dict | None:
    """Return {position, orient_wxyz} for /World/Anchorpoint if present."""
    prim = stage.GetPrimAtPath("/World/Anchorpoint")
    if not prim or not prim.IsValid():
        return None
    attrs = {a.GetName(): a.Get() for a in prim.GetAttributes()
             if a.HasAuthoredValue()}
    t = attrs.get("xformOp:translate")
    o = attrs.get("xformOp:orient")
    if t is None:
        return None
    pos = [_round(t[0]), _round(t[1]), _round(t[2])]
    if o is not None:
        # Gf.Quatf / Gf.Quatd: w = real, xyz = imaginary.
        w = float(o.GetReal())
        xyz = o.GetImaginary()
        wxyz = [_round(w), _round(xyz[0]), _round(xyz[1]), _round(xyz[2])]
    else:
        wxyz = [1.0, 0.0, 0.0, 0.0]
    return {"position": pos, "orient_wxyz": wxyz}


def main() -> None:
    pool_ids: list[str] = json.loads(POOL_PATH.read_text())["asset_ids"]
    tax = json.loads(TAX_PATH.read_text())
    usd_index: dict[str, str] = {
        v["variant_id"]: v["usd_path"]
        for cat in tax["categories"]
        for v in cat["variants"]
        if v.get("usd_path")
    }

    out: dict[str, dict] = {}
    suspect: list[str] = []

    for vid in pool_ids:
        usd_rel = usd_index.get(vid)
        if not usd_rel:
            print(f"  [warn] {vid}: no usd_path in taxonomy")
            continue
        url = NUCLEUS_ASSET_ROOT + usd_rel
        local = CACHE_DIR / Path(usd_rel).name
        try:
            _download(url, local)
        except Exception as e:
            print(f"  [warn] {vid}: download failed: {e}")
            continue

        stage = Usd.Stage.Open(str(local))
        if stage is None:
            print(f"  [warn] {vid}: failed to open USD")
            continue

        anchorpoint = _read_anchorpoint(stage)
        origin = {"position": [0.0, 0.0, 0.0],
                  "orient_wxyz": [1.0, 0.0, 0.0, 0.0]}

        anchors = {"origin": origin}
        if anchorpoint is not None:
            anchors["anchorpoint"] = anchorpoint
            # Heuristic: if anchorpoint sits at origin with identity orient,
            # the authored data is degenerate (same as the implicit origin).
            same_pos = anchorpoint["position"] == [0.0, 0.0, 0.0]
            same_rot = anchorpoint["orient_wxyz"] == [1.0, 0.0, 0.0, 0.0]
            if same_pos and same_rot:
                suspect.append(vid)
        else:
            suspect.append(vid)

        out[vid] = {"anchors": anchors, "aliases": {}}
        ap_str = (
            f"pos={anchorpoint['position']} q_wxyz={anchorpoint['orient_wxyz']}"
            if anchorpoint else "(missing)"
        )
        print(f"  {vid:30s} anchorpoint {ap_str}")

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {len(out)} entries to {OUT_PATH}")
    if suspect:
        print(
            f"\n[review] {len(suspect)} asset(s) need hand-author override "
            "(missing or degenerate anchorpoint):",
        )
        for vid in suspect:
            print(f"  - {vid}")


if __name__ == "__main__":
    main()
