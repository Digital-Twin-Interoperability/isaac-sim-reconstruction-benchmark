from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from isaacsim_bench.schemas.scene import SceneJSON
from isaacsim_bench.taxonomy.registry import TaxonomyRegistry


@dataclass
class RenderArtifact:
    """Output of a render pass — disk paths and optional in-memory arrays."""

    sample_id: str

    # Disk paths (populated by render_scene when output_dir is given)
    rgb_path: Path | None = None
    depth_path: Path | None = None
    segmentation_path: Path | None = None
    scene_json_path: Path | None = None

    # In-memory arrays (always populated so agent loops can consume directly)
    rgb: np.ndarray | None = field(default=None, repr=False)  # (H, W, 3) uint8
    depth: np.ndarray | None = field(default=None, repr=False)  # (H, W) float32
    segmentation: np.ndarray | None = field(default=None, repr=False)  # (H, W) int32


class SceneGenerator(ABC):
    """Abstract interface for scene generation.

    Implementations will use omni.isaac.core, Replicator, etc.
    This base class deliberately avoids any Omniverse/Isaac Sim imports
    so it can be used in environments without those dependencies.
    """

    @abstractmethod
    def generate_scene(
        self,
        template_id: str,
        params: dict[str, Any],
        registry: TaxonomyRegistry,
    ) -> SceneJSON:
        """Compose a scene from a template and parameters.

        Returns a SceneJSON with all component transforms, relations,
        and match regimes populated.  No stage mutation here — pure data.
        """
        ...

    @abstractmethod
    def render_scene(
        self,
        scene: SceneJSON,
        output_dir: Path | None = None,
    ) -> RenderArtifact:
        """Render a composed scene.

        Always populates in-memory arrays (rgb, depth, segmentation) on the
        returned RenderArtifact.  When *output_dir* is given, also writes:
        - rgb.png
        - depth.npy
        - segmentation.png
        - scene.json
        and sets the corresponding path fields.
        """
        ...

    def generate_batch(
        self,
        configs: list[dict[str, Any]],
        output_dir: Path,
        registry: TaxonomyRegistry,
    ) -> list[tuple[SceneJSON, RenderArtifact]]:
        """Generate and render a batch of scenes.

        Each config dict should contain 'template_id' and 'params' keys.
        Returns (scene, artifact) pairs.
        """
        results: list[tuple[SceneJSON, RenderArtifact]] = []
        for cfg in configs:
            scene = self.generate_scene(
                cfg["template_id"], cfg.get("params", {}), registry
            )
            sample_dir = output_dir / scene.sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            artifact = self.render_scene(scene, sample_dir)
            results.append((scene, artifact))
        return results
