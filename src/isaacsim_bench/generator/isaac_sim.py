"""Isaac Sim / Omniverse Replicator implementation of SceneGenerator.

This module has hard dependencies on the Isaac Sim runtime
(omni.usd, omni.replicator.core, etc.).  It is *not* imported by the
rest of isaacsim_bench unless the user explicitly instantiates
``IsaacSimSceneGenerator``.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from isaacsim_bench.generator.base import RenderArtifact, SceneGenerator
from isaacsim_bench.generator.templates import TEMPLATES, TemplateResult
from isaacsim_bench.schemas.scene import SceneJSON
from isaacsim_bench.taxonomy.registry import TaxonomyRegistry

logger = logging.getLogger(__name__)

# Isaac / Omniverse imports are deferred to method bodies so that
# ``import isaacsim_bench.generator.isaac_sim`` only fails at *call*
# time if the runtime is missing, not at import time.  This keeps
# the test suite and CLI usable without Isaac Sim installed.


def _lazy_import_omni():
    """Return (Usd, UsdGeom, Gf, Sdf, rep) from the Omniverse runtime."""
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    return Usd, UsdGeom, Gf, Sdf, UsdLux, omni.usd, rep


class IsaacSimSceneGenerator(SceneGenerator):
    """Concrete generator backed by Isaac Sim / Omniverse Kit.

    Parameters
    ----------
    asset_root : str
        Root path (local or Nucleus) prepended to ``Variant.usd_path``.
    resolution : tuple[int, int]
        (width, height) of rendered images.
    """

    def __init__(
        self,
        asset_root: str = "",
        resolution: tuple[int, int] = (1024, 1024),
    ) -> None:
        self.asset_root = asset_root
        self.resolution = resolution

    # ------------------------------------------------------------------
    # generate_scene — pure data, no stage mutation
    # ------------------------------------------------------------------

    def generate_scene(
        self,
        template_id: str,
        params: dict[str, Any],
        registry: TaxonomyRegistry,
    ) -> SceneJSON:
        builder = TEMPLATES.get(template_id)
        if builder is None:
            raise ValueError(
                f"Unknown template '{template_id}'. "
                f"Available: {sorted(TEMPLATES)}"
            )

        result: TemplateResult = builder(params, registry)

        # Deterministic sample_id from template + a hash of params
        param_tag = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
        sample_id = f"{template_id}_{param_tag}" if param_tag else template_id

        return SceneJSON(
            sample_id=sample_id,
            benchmark_tier="closed_world",
            family=result.family,
            template_id=template_id,
            root_node=result.root_node,
            template_params=params,
            camera=result.camera,
            components=result.components,
            relations=result.relations,
            retrieval_pool_version=registry.taxonomy.version,
            world_pool_version=registry.taxonomy.version,
            taxonomy_version=registry.taxonomy.version,
            generator_version="isaac_sim_v1",
        )

    # ------------------------------------------------------------------
    # render_scene — Replicator pipeline
    # ------------------------------------------------------------------

    def render_scene(
        self,
        scene: SceneJSON,
        output_dir: Path | None = None,
    ) -> RenderArtifact:
        Usd, UsdGeom, Gf, Sdf, UsdLux, omni_usd, rep = _lazy_import_omni()

        stage = omni_usd.get_context().get_stage()

        # ---- clean the /World prim for a fresh scene ----
        world_path = "/World"
        if stage.GetPrimAtPath(world_path):
            stage.RemovePrim(world_path)
        UsdGeom.Xform.Define(stage, world_path)

        registry_for_resolve = None  # set below if we can find it
        # We don't carry the registry into render, so resolve via asset_root
        # directly from usd_path stored in taxonomy.

        # ---- place components ----
        for comp in scene.components:
            prim_path = f"{world_path}/{comp.name}"
            xform = UsdGeom.Xform.Define(stage, prim_path)

            # Reference the USD asset
            usd_rel = f"{self.asset_root}/{comp.asset_id}.usd"
            prim = stage.GetPrimAtPath(prim_path)
            prim.GetReferences().AddReference(usd_rel)

            # Apply transform
            tx, ty, tz = comp.translate
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
            qx, qy, qz, qw = comp.orientation_xyzw
            xformable.AddOrientOp().Set(Gf.Quatd(qw, qx, qy, qz))

            # Semantic label for segmentation
            rep.modify.semantics([("class", comp.family)], prim_path)

        # ---- default dome light ----
        light_path = f"{world_path}/DomeLight"
        dome = UsdLux.DomeLight.Define(stage, light_path)
        dome.GetIntensityAttr().Set(1000.0)

        # ---- camera ----
        cam_path = f"{world_path}/BenchmarkCamera"
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam_xformable = UsdGeom.Xformable(cam.GetPrim())
        cam_xformable.ClearXformOpOrder()

        pos = scene.camera.position
        tgt = scene.camera.target

        # Compute look-at transform
        eye = Gf.Vec3d(*pos)
        center = Gf.Vec3d(*tgt)
        up = Gf.Vec3d(0, 0, 1)
        look_at = Gf.Matrix4d()
        look_at.SetLookAt(eye, center, up)
        # SetLookAt returns view matrix; we need the inverse for world transform
        world_xform = look_at.GetInverse()
        cam_xformable.AddTransformOp().Set(world_xform)

        # Set FOV via focal length approximation
        # fov = 2 * atan(aperture / (2 * focal_length))
        horiz_aperture = cam.GetHorizontalApertureAttr().Get() or 20.955
        focal_length = horiz_aperture / (
            2.0 * math.tan(math.radians(scene.camera.fov_deg) / 2.0)
        )
        cam.GetFocalLengthAttr().Set(focal_length)

        # ---- render with Replicator annotators ----
        width, height = self.resolution
        rp = rep.create.render_product(cam_path, (width, height))

        rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        seg_ann = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")

        rgb_ann.attach([rp])
        depth_ann.attach([rp])
        seg_ann.attach([rp])

        # Step the renderer to produce output
        rep.orchestrator.step()

        rgb_data: np.ndarray = rgb_ann.get_data()[:, :, :3].astype(np.uint8)
        depth_data: np.ndarray = depth_ann.get_data().astype(np.float32)
        seg_raw = seg_ann.get_data()
        seg_data: np.ndarray = seg_raw["data"].astype(np.int32)

        # Detach annotators for cleanup
        rgb_ann.detach([rp])
        depth_ann.detach([rp])
        seg_ann.detach([rp])

        # ---- build artifact ----
        artifact = RenderArtifact(
            sample_id=scene.sample_id,
            rgb=rgb_data,
            depth=depth_data,
            segmentation=seg_data,
        )

        # ---- write to disk if requested ----
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            from PIL import Image

            rgb_path = output_dir / "rgb.png"
            Image.fromarray(rgb_data).save(rgb_path)
            artifact.rgb_path = rgb_path

            depth_path = output_dir / "depth.npy"
            np.save(depth_path, depth_data)
            artifact.depth_path = depth_path

            seg_path = output_dir / "segmentation.png"
            Image.fromarray(seg_data.astype(np.uint16)).save(seg_path)
            artifact.segmentation_path = seg_path

            scene_path = output_dir / "scene.json"
            scene_path.write_text(
                scene.model_dump_json(indent=2, by_alias=True)
            )
            artifact.scene_json_path = scene_path

            logger.info("Saved render artifacts to %s", output_dir)

        return artifact
