"""Isaac Sim rendering session with lazy ``SimulationApp`` lifecycle.

``pxr`` imports are deferred to function scope because Isaac Sim's bundled
``python.sh`` only exposes the ``pxr`` module once ``SimulationApp()`` has
booted.  (In a normal ``uv`` venv ``pxr`` is provided by the ``usd-core``
package and is always importable — but we standardise on the deferred form
so both environments work identically.)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_RESOLUTION: tuple[int, int] = (1024, 1024)
DEFAULT_WARM_UP_FRAMES = 60
DEFAULT_SETTLE_FRAMES = 40
DEFAULT_STAGE_LOAD_FRAMES = 30
DEFAULT_POST_ATTACH_FRAMES = 20
DEFAULT_POST_STEP_FRAMES = 10
DEFAULT_CLOSE_FRAMES = 5

DEFAULT_CAM_PATH = "/RenderCam"
DEFAULT_DOME_INTENSITY = 500.0
DEFAULT_DISTANT_INTENSITY = 3000.0

# (name, azimuth_deg, elevation_deg)
_DEFAULT_ORBITS: list[tuple[str, float, float]] = [
    ("front_high", 0, 40),
    ("front_right", 45, 35),
    ("right", 90, 30),
    ("back_right", 135, 35),
    ("back", 180, 40),
    ("back_left", 225, 35),
    ("left", 270, 30),
    ("front_left", 315, 35),
    ("top_down", 0, 80),
    ("low_angle", 30, 10),
]


@dataclass
class Viewpoint:
    name: str
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_deg: float = 60.0


@dataclass
class RenderResult:
    rendered: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    output_paths: dict[str, Path] = field(default_factory=dict)
    output_dir: Path | None = None
    resolution: tuple[int, int] = DEFAULT_RESOLUTION


# ----------------------------------------------------------------------------
# Pure-pxr helpers (safe to import / call outside python.sh; require a stage)
# ----------------------------------------------------------------------------

def compute_scene_bbox(stage: Any) -> tuple[Any, Any]:
    """Return (center, size) of the scene bounding box under ``/Root`` or ``/World``.

    Returns ``Gf.Vec3d`` tuples (pxr types).  pxr is imported lazily so this
    function is safely importable in contexts without pxr available.
    """
    from pxr import Gf, Usd, UsdGeom

    root = stage.GetPrimAtPath("/Root")
    if not root:
        root = stage.GetPrimAtPath("/World")
    if not root:
        return Gf.Vec3d(0, 0, 0), Gf.Vec3d(2, 2, 2)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(root)
    box = bbox.ComputeAlignedRange()
    lo = Gf.Vec3d(box.GetMin())
    hi = Gf.Vec3d(box.GetMax())
    return (lo + hi) * 0.5, hi - lo


def auto_orbit_viewpoints(
    center: Any,
    size: Any,
    scene_camera: dict | None = None,
    base_fov_deg: float = 60.0,
) -> list[Viewpoint]:
    """Generate default orbit viewpoints plus an optional scene-defined camera.

    ``center`` and ``size`` are ``Gf.Vec3d`` (or any 3-indexable).
    """
    diag = math.sqrt(size[0] ** 2 + size[1] ** 2 + size[2] ** 2)
    if diag < 0.1:
        diag = 2.0
    half_fov = math.radians(base_fov_deg) / 2.0
    dist = max((diag * 0.7) / math.tan(half_fov), 2.0)

    viewpoints: list[Viewpoint] = []

    if scene_camera:
        viewpoints.append(
            Viewpoint(
                name="scene_default",
                position=tuple(scene_camera["position"]),
                target=tuple(scene_camera["target"]),
                fov_deg=scene_camera.get("fov_deg", base_fov_deg),
            )
        )

    for name, azim_deg, elev_deg in _DEFAULT_ORBITS:
        elev_rad = math.radians(elev_deg)
        azim_rad = math.radians(azim_deg)
        dx = dist * math.cos(elev_rad) * math.cos(azim_rad)
        dy = dist * math.cos(elev_rad) * math.sin(azim_rad)
        dz = dist * math.sin(elev_rad)
        viewpoints.append(
            Viewpoint(
                name=name,
                position=(center[0] + dx, center[1] + dy, center[2] + dz),
                target=(center[0], center[1], center[2]),
                fov_deg=base_fov_deg,
            )
        )

    return viewpoints


def create_or_move_camera(
    stage: Any,
    cam_path: str,
    position: tuple[float, float, float],
    target: tuple[float, float, float],
    fov_deg: float,
) -> None:
    """Create (if missing) and point a camera prim at ``target`` from ``position``."""
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(cam_path)
    if not prim:
        UsdGeom.Camera.Define(stage, cam_path)
        prim = stage.GetPrimAtPath(cam_path)

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()

    eye = Gf.Vec3d(*position)
    tgt = Gf.Vec3d(*target)
    up = Gf.Vec3d(0, 0, 1)

    look_at = Gf.Matrix4d()
    look_at.SetLookAt(eye, tgt, up)
    xformable.AddTransformOp().Set(look_at.GetInverse())

    cam = UsdGeom.Camera(prim)
    horiz_aperture = cam.GetHorizontalApertureAttr().Get() or 20.955
    focal_length = horiz_aperture / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cam.GetFocalLengthAttr().Set(focal_length)


def add_scene_lighting(
    stage: Any,
    dome_intensity: float = DEFAULT_DOME_INTENSITY,
    distant_intensity: float = DEFAULT_DISTANT_INTENSITY,
) -> None:
    """Add a dome light + a directional distant light under ``/Lights``."""
    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/Lights/DomeLight")
    dome.GetIntensityAttr().Set(dome_intensity)

    dist_light = UsdLux.DistantLight.Define(stage, "/Lights/DistantLight")
    dist_light.GetIntensityAttr().Set(distant_intensity)
    xf = UsdGeom.Xformable(dist_light.GetPrim())
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))


# ----------------------------------------------------------------------------
# Isaac Sim session (lazy boot)
# ----------------------------------------------------------------------------

ProgressFn = Callable[[int, int, Viewpoint, str], None]
"""``(index, total, viewpoint, status)`` — status is ``"start"``, ``"done"``, or ``"failed"``."""


class IsaacRenderSession:
    """Manage a ``SimulationApp`` and render product across one or more jobs.

    The ``SimulationApp`` is booted lazily on :meth:`boot` (or the first
    rendering call).  Importing this class does not require Isaac Sim to be
    installed; only calling :meth:`boot` does.

    Example — standalone single-scene render::

        with IsaacRenderSession(headless=True) as session:
            session.open_stage(Path("scene.usd"))
            result = session.render_auto_orbit(Path("out/"), scene_camera=None)

    Example — reuse across many renders (expected use from the agent loop)::

        session = IsaacRenderSession(headless=True)
        session.boot()
        for snapshot in agent_iterations:
            session.open_stage(snapshot.usd_path)
            session.render_viewpoints(viewpoints, snapshot.out_dir)
        session.close()
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        resolution: tuple[int, int] = DEFAULT_RESOLUTION,
        warm_up_frames: int = DEFAULT_WARM_UP_FRAMES,
        settle_frames: int = DEFAULT_SETTLE_FRAMES,
        cam_path: str = DEFAULT_CAM_PATH,
    ) -> None:
        self.headless = headless
        self.resolution = resolution
        self.warm_up_frames = warm_up_frames
        self.settle_frames = settle_frames
        self.cam_path = cam_path

        self._simulation_app: Any = None
        self._rep: Any = None
        self._omni_usd: Any = None
        self._render_product: Any = None
        self._rgb_annotator: Any = None
        self._stage: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_booted(self) -> bool:
        return self._simulation_app is not None

    @property
    def stage(self) -> Any:
        """The current ``pxr.Usd.Stage``, or ``None`` if no stage is loaded."""
        return self._stage

    def boot(self) -> None:
        """Start the underlying ``SimulationApp``.  No-op if already booted."""
        if self._simulation_app is not None:
            return

        from isaacsim import SimulationApp

        self._simulation_app = SimulationApp({"headless": self.headless})

        # These imports are only valid once SimulationApp is up.
        import omni.kit.app  # noqa: F401  (imported for side effects)
        import omni.replicator.core as rep
        import omni.usd

        self._omni_usd = omni.usd
        self._rep = rep

    def _tear_down_render_product(self) -> None:
        """Detach + destroy the current render product and annotator, if any.

        Called both at :meth:`close` and at the start of :meth:`open_stage`,
        because the rgb annotator is implicitly bound to the previous stage
        — reusing it after a reopen raises ``AnnotatorRegistryError: Annotator
        rgb is not attached to any render products.``
        """
        try:
            if self._rgb_annotator is not None and self._render_product is not None:
                self._rgb_annotator.detach([self._render_product])
        except Exception:
            logger.debug("Failed to detach RGB annotator", exc_info=True)
        try:
            if self._render_product is not None:
                self._render_product.destroy()
        except Exception:
            logger.debug("Failed to destroy render product", exc_info=True)
        self._render_product = None
        self._rgb_annotator = None

    def close(self) -> None:
        """Detach render product and shut down ``SimulationApp``.  Idempotent."""
        if self._simulation_app is None:
            return

        self._tear_down_render_product()
        try:
            if self._rep is not None:
                self._rep.orchestrator.stop()
        except Exception:
            logger.debug("Failed to stop replicator orchestrator", exc_info=True)

        for _ in range(DEFAULT_CLOSE_FRAMES):
            self._simulation_app.update()
        self._simulation_app.close()

        self._simulation_app = None
        self._rep = None
        self._omni_usd = None
        self._stage = None

    def __enter__(self) -> IsaacRenderSession:
        self.boot()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def open_stage(
        self,
        usd_path: Path,
        *,
        add_lighting: bool = True,
    ) -> Any:
        """Open ``usd_path`` in the Omniverse context.

        Does **not** prepare the render product; call :meth:`prepare_render_product`
        afterwards (or simply call :meth:`render_viewpoints`, which lazy-prepares).

        If a render product was attached to a previously-opened stage, it is
        torn down here — the rgb annotator does not survive a stage reopen,
        and reusing it raises ``AnnotatorRegistryError`` on the next render.
        """
        self._ensure_booted()
        self._tear_down_render_product()
        ctx = self._omni_usd.get_context()
        ctx.open_stage(str(usd_path))

        for _ in range(DEFAULT_STAGE_LOAD_FRAMES):
            self._simulation_app.update()

        stage = ctx.get_stage()
        if not stage:
            raise RuntimeError(f"Failed to open stage: {usd_path}")
        self._stage = stage

        if add_lighting:
            add_scene_lighting(stage)

        return stage

    def prepare_render_product(self) -> None:
        """Create the render camera + RGB annotator and warm up the renderer.

        Idempotent — safe to call multiple times.  The render product, once
        created, is reused across subsequent :meth:`render_viewpoints` calls.
        """
        self._ensure_booted()
        if self._stage is None:
            raise RuntimeError("No stage loaded — call open_stage() first")

        create_or_move_camera(
            self._stage,
            self.cam_path,
            position=(5, 5, 5),
            target=(0, 0, 0),
            fov_deg=60.0,
        )
        for _ in range(self.warm_up_frames):
            self._simulation_app.update()

        if self._render_product is None:
            self._render_product = self._rep.create.render_product(
                self.cam_path, self.resolution,
            )
            self._rgb_annotator = self._rep.AnnotatorRegistry.get_annotator("rgb")
            self._rgb_annotator.attach([self._render_product])
            for _ in range(DEFAULT_POST_ATTACH_FRAMES):
                self._simulation_app.update()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_viewpoints(
        self,
        viewpoints: list[Viewpoint],
        output_dir: Path,
        *,
        progress: ProgressFn | None = None,
    ) -> RenderResult:
        """Render each viewpoint to ``<output_dir>/<viewpoint.name>.png``."""
        self._ensure_booted()
        if self._stage is None:
            raise RuntimeError("No stage loaded — call open_stage() first")
        if self._render_product is None:
            self.prepare_render_product()
        output_dir.mkdir(parents=True, exist_ok=True)

        result = RenderResult(output_dir=output_dir, resolution=self.resolution)
        total = len(viewpoints)

        for idx, vp in enumerate(viewpoints, start=1):
            if progress:
                progress(idx, total, vp, "start")
            try:
                create_or_move_camera(
                    self._stage, self.cam_path, vp.position, vp.target, vp.fov_deg,
                )
                for _ in range(self.settle_frames):
                    self._simulation_app.update()
                self._rep.orchestrator.step()
                for _ in range(DEFAULT_POST_STEP_FRAMES):
                    self._simulation_app.update()

                rgb_data = self._rgb_annotator.get_data()
                if rgb_data is None or rgb_data.ndim < 3:
                    logger.warning("Invalid RGB data for viewpoint %s", vp.name)
                    result.failed.append(vp.name)
                    if progress:
                        progress(idx, total, vp, "failed")
                    continue

                rgb_data = rgb_data[:, :, :3].astype(np.uint8)
                out_path = output_dir / f"{vp.name}.png"
                Image.fromarray(rgb_data).save(out_path)
                result.rendered.append(vp.name)
                result.output_paths[vp.name] = out_path
                if progress:
                    progress(idx, total, vp, "done")

            except Exception:
                logger.exception("Viewpoint %s failed", vp.name)
                result.failed.append(vp.name)
                if progress:
                    progress(idx, total, vp, "failed")

        return result

    def render_auto_orbit(
        self,
        output_dir: Path,
        *,
        scene_camera: dict | None = None,
        progress: ProgressFn | None = None,
    ) -> RenderResult:
        """Auto-generate orbit viewpoints from the current stage's bbox and render."""
        self._ensure_booted()
        if self._stage is None:
            raise RuntimeError("No stage loaded — call open_stage() first")
        center, size = compute_scene_bbox(self._stage)
        logger.info(
            "Scene center=(%.2f, %.2f, %.2f)  size=(%.2f, %.2f, %.2f)",
            center[0], center[1], center[2], size[0], size[1], size[2],
        )
        viewpoints = auto_orbit_viewpoints(center, size, scene_camera)
        return self.render_viewpoints(viewpoints, output_dir, progress=progress)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_booted(self) -> None:
        if self._simulation_app is None:
            raise RuntimeError(
                "IsaacRenderSession not booted — call .boot() or use as a context manager",
            )
