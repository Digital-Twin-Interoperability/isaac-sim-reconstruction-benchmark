"""Composer agent — assemble a PredictionJSON and optionally export USD."""

from __future__ import annotations

import math
from pathlib import Path

from isaacsim_bench.agents.retrieval import AssetMatch
from isaacsim_bench.agents.spatial import LayoutEstimate
from isaacsim_bench.schemas.prediction import (
    PredictedComponent,
    PredictedRelation,
    PredictionJSON,
)
from isaacsim_bench.schemas.taxonomy import AssetTaxonomy

NUCLEUS_ASSET_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/"
)


def _heading_to_quat(deg: float) -> list[float]:
    """Convert a Z-axis heading (degrees) to [x, y, z, w] quaternion."""
    half = math.radians(deg) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


class ComposerAgent:
    """Pure-logic agent that assembles upstream outputs into PredictionJSON."""

    def compose(
        self,
        scene_id: str,
        matches: list[AssetMatch],
        layout: LayoutEstimate,
        latency_seconds: float = 0.0,
    ) -> PredictionJSON:
        pose_map = {p.name: p for p in layout.poses}

        components: list[PredictedComponent] = []
        for m in matches:
            pose = pose_map.get(m.name)
            translate = pose.position if pose else [0.0, 0.0, 0.0]
            heading = pose.heading_deg if pose else 0.0

            components.append(PredictedComponent(
                name=m.name,
                asset_id=m.asset_id,
                family=m.family,
                translate=translate,
                orientation_xyzw=_heading_to_quat(heading),
                confidence=m.confidence,
            ))

        relations = [
            PredictedRelation(
                type=r.type,
                from_node=r.from_node,
                to_node=r.to_node,
                from_anchor=r.from_anchor,
                to_anchor=r.to_anchor,
            )
            for r in layout.relations
        ]

        return PredictionJSON(
            sample_id=scene_id,
            components=components,
            relations=relations,
            latency_seconds=latency_seconds,
        )

    @staticmethod
    def export_usd(
        prediction: PredictionJSON,
        output_path: Path,
        taxonomy: AssetTaxonomy,
        asset_root: str = NUCLEUS_ASSET_ROOT,
    ) -> None:
        """Write the prediction to a USD file with asset references."""
        from pxr import Gf, Usd, UsdGeom

        # Build variant_id → usd_path index
        usd_index: dict[str, str] = {}
        for cat in taxonomy.categories:
            for var in cat.variants:
                if var.usd_path:
                    usd_index[var.variant_id] = var.usd_path

        stage = Usd.Stage.CreateNew(str(output_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        UsdGeom.Xform.Define(stage, "/Root")

        for comp in prediction.components:
            prim_path = f"/Root/{comp.name}"
            UsdGeom.Xform.Define(stage, prim_path)
            prim = stage.GetPrimAtPath(prim_path)

            usd_path = usd_index.get(comp.asset_id, "")
            if usd_path:
                ref = f"{asset_root}{usd_path}" if asset_root else usd_path
                prim.GetReferences().AddReference(ref)

            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()

            tx, ty, tz = comp.translate
            xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))

            qx, qy, qz, qw = comp.orientation_xyzw
            # Request PrecisionDouble so we match the ``quatd`` precision the
            # referenced Isaac Sim assets typically bring in.  ``AddOrientOp``
            # refuses to reconcile a prim that already has an orient op of
            # one precision with a request of another.
            orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            orient_op.Set(Gf.Quatd(qw, qx, qy, qz))

        stage.GetRootLayer().Save()
