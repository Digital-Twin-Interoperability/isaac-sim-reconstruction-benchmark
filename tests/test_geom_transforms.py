"""Unit tests for geom/transforms.py.

The conveyor cases use the actual extracted anchor data so a regression
here means the U-shape reconstruction would silently break.
"""

from __future__ import annotations

import math

import pytest

from isaacsim_bench.geom.transforms import (
    QUAT_RZ_PI,
    align_pose,
    pose_compose,
    pose_invert,
    quat_from_axis_angle,
    quat_invert,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

I_QUAT = (1.0, 0.0, 0.0, 0.0)


def approx_vec(a, b, tol=1e-5):
    return all(abs(x - y) < tol for x, y in zip(a, b))


class TestQuatBasics:
    def test_normalize_zero_returns_identity(self):
        assert quat_normalize((0.0, 0.0, 0.0, 0.0)) == I_QUAT

    def test_multiply_identity_is_noop(self):
        q = (0.5, 0.5, 0.5, 0.5)
        assert quat_multiply(I_QUAT, q) == q

    def test_invert_rotates_back(self):
        q = quat_from_axis_angle((0.0, 0.0, 1.0), math.pi / 3)
        v = (1.0, 0.0, 0.0)
        rot = quat_rotate(q, v)
        back = quat_rotate(quat_invert(q), rot)
        assert approx_vec(back, v)

    def test_rz_pi_flips_x_axis(self):
        v = (1.0, 0.0, 0.0)
        assert approx_vec(quat_rotate(QUAT_RZ_PI, v), (-1.0, 0.0, 0.0))

    def test_xyzw_wxyz_roundtrip(self):
        q = (0.1, 0.2, 0.3, 0.9273)
        assert xyzw_to_wxyz(wxyz_to_xyzw(q)) == q


class TestPoseCompose:
    def test_compose_with_identity_is_noop(self):
        pos, quat = pose_compose(
            (1.0, 2.0, 3.0), I_QUAT,
            (0.0, 0.0, 0.0), I_QUAT,
        )
        assert approx_vec(pos, (1.0, 2.0, 3.0))
        assert approx_vec(quat, I_QUAT)

    def test_translate_then_rotate(self):
        # 90° about Z, then translation (1,0,0) in body frame -> (0,1,0) world.
        rz90 = quat_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)
        pos, _ = pose_compose((0.0, 0.0, 0.0), rz90, (1.0, 0.0, 0.0), I_QUAT)
        assert approx_vec(pos, (0.0, 1.0, 0.0))

    def test_invert_undoes_compose(self):
        a_pos, a_q = (1.0, 2.0, 3.0), quat_from_axis_angle((0, 0, 1), 0.7)
        inv_pos, inv_q = pose_invert(a_pos, a_q)
        ident_pos, ident_q = pose_compose(a_pos, a_q, inv_pos, inv_q)
        assert approx_vec(ident_pos, (0.0, 0.0, 0.0))
        assert approx_vec(ident_q, I_QUAT)


class TestAlignPose:
    """The conveyor chaining contract.

    For each test the *moving* anchor is the moving piece's ``in`` (its
    own origin), so ``moving_anchor_local`` is identity and the chain
    literally puts moving's origin at fixed's anchor frame.
    """

    A05_OUT_POS = (2.000006, 0.0, 0.0)
    A05_OUT_QUAT = I_QUAT
    A14_OUT_POS = (0.0, -3.91878, 0.0)
    A14_OUT_QUAT = (0.0, 0.0, 0.0, 1.0)  # 180° about Z
    ORIGIN_POS = (0.0, 0.0, 0.0)
    ORIGIN_QUAT = I_QUAT

    def test_a05_to_a05_chains_forward(self):
        # Place fixed A05 at world origin; mate next A05.in to fixed.out.
        # Expect moving origin at (2, 0, 0), no rotation accumulated.
        pos, quat = align_pose(
            (0.0, 0.0, 0.0), I_QUAT,
            self.A05_OUT_POS, self.A05_OUT_QUAT,
            self.ORIGIN_POS, self.ORIGIN_QUAT,
        )
        assert approx_vec(pos, (2.000006, 0.0, 0.0))
        assert approx_vec(quat, I_QUAT)

    def test_a14_to_a05_preserves_authored_180(self):
        # Place A14 at world origin; mate A05.in to A14.curve_entry (=out).
        # Expected: moving origin at the U's far end, rotated 180° about Z.
        pos, quat = align_pose(
            (0.0, 0.0, 0.0), I_QUAT,
            self.A14_OUT_POS, self.A14_OUT_QUAT,
            self.ORIGIN_POS, self.ORIGIN_QUAT,
        )
        assert approx_vec(pos, (0.0, -3.91878, 0.0))
        # 180° about Z either as (0,0,0,1) or (0,0,0,-1) — same rotation.
        assert abs(quat[0]) < 1e-5
        assert abs(quat[1]) < 1e-5
        assert abs(quat[2]) < 1e-5
        assert abs(abs(quat[3]) - 1.0) < 1e-5

    def test_a05_then_chain_through_a14(self):
        # Fixed A05 at origin, append A14 at A05.out: A14 origin lands at
        # (2, 0, 0).  Then a second A05 mates to A14.curve_entry: it must
        # come back near the fixed start, rotated 180° (the U closes).
        a14_pos, a14_quat = align_pose(
            (0.0, 0.0, 0.0), I_QUAT,
            self.A05_OUT_POS, self.A05_OUT_QUAT,
            self.ORIGIN_POS, self.ORIGIN_QUAT,
        )
        a05b_pos, a05b_quat = align_pose(
            a14_pos, a14_quat,
            self.A14_OUT_POS, self.A14_OUT_QUAT,
            self.ORIGIN_POS, self.ORIGIN_QUAT,
        )
        # A14 swept -3.92 in its body Y; after 180° about Z applied,
        # the second A05 should sit at x=2 (same as A14 start) and
        # y=-3.92 (the far end), facing back -X.
        assert approx_vec(a05b_pos, (2.000006, -3.91878, 0.0), tol=1e-4)
        # 180° rotation accumulated
        assert abs(a05b_quat[0]) < 1e-5
        assert abs(a05b_quat[3]) > 0.99

    def test_opposed_frame_inserts_180_about_z(self):
        # Same A05->A05 chain but with opposed_frame: the moving piece
        # should land at the same position but flipped 180°.
        pos, quat = align_pose(
            (0.0, 0.0, 0.0), I_QUAT,
            self.A05_OUT_POS, self.A05_OUT_QUAT,
            self.ORIGIN_POS, self.ORIGIN_QUAT,
            facing="opposed_frame",
        )
        assert approx_vec(pos, (2.000006, 0.0, 0.0))
        assert abs(quat[0]) < 1e-5
        assert abs(abs(quat[3]) - 1.0) < 1e-5

    def test_rejects_unknown_facing(self):
        with pytest.raises(ValueError, match="facing"):
            align_pose(
                (0.0, 0.0, 0.0), I_QUAT,
                self.A05_OUT_POS, self.A05_OUT_QUAT,
                self.ORIGIN_POS, self.ORIGIN_QUAT,
                facing="bogus",
            )
