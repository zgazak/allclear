"""Tests for triangle hash matching."""

import numpy as np
import pytest

from allclear.matching import (
    build_triangle_hashes,
    build_hash_index,
    match_triangles,
    match_sources,
)


class TestTriangleHashes:
    def test_scale_invariance(self):
        """Same pattern scaled should produce identical hashes."""
        pos1 = np.array([[0, 0], [1, 0], [0.5, 0.8], [1.5, 0.3]])
        pos2 = pos1 * 3.7  # scaled

        h1, _ = build_triangle_hashes(pos1, n_brightest=4)
        h2, _ = build_triangle_hashes(pos2, n_brightest=4)

        np.testing.assert_allclose(np.sort(h1, axis=0),
                                   np.sort(h2, axis=0), atol=1e-10)

    def test_rotation_invariance(self):
        """Same pattern rotated should produce identical hashes."""
        pos1 = np.array([[0, 0], [1, 0], [0.5, 0.8], [1.5, 0.3]])
        theta = 0.7
        R = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        pos2 = (R @ pos1.T).T

        h1, _ = build_triangle_hashes(pos1, n_brightest=4)
        h2, _ = build_triangle_hashes(pos2, n_brightest=4)

        np.testing.assert_allclose(np.sort(h1, axis=0),
                                   np.sort(h2, axis=0), atol=1e-10)

    def test_hash_values_in_range(self):
        """Hash values (a/c, b/c) should be in [0, 1]."""
        rng = np.random.default_rng(42)
        pos = rng.uniform(0, 100, (20, 2))
        hashes, _ = build_triangle_hashes(pos, n_brightest=20)
        assert np.all(hashes >= 0)
        assert np.all(hashes <= 1.0 + 1e-10)


class TestMatchTriangles:
    def test_match_shifted_pattern(self):
        """Matching should work for a shifted+rotated pattern."""
        rng = np.random.default_rng(42)
        # Create a pattern of 15 "stars"
        base_pos = rng.uniform(100, 900, (15, 2))

        # "Catalog" positions: shifted and rotated
        theta = 0.3
        R = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        cat_pos = (R @ base_pos.T).T + np.array([50, -30])

        matches = match_triangles(
            base_pos, cat_pos,
            n_brightest=15, hash_tol=0.02, min_votes=2,
        )
        assert len(matches) >= 5


class TestMatchSources:
    def test_exact_match(self):
        """Identical positions should all match with zero distance."""
        pos = np.array([[100, 200], [300, 400], [500, 600]])
        matches, dists = match_sources(pos, pos, max_dist=1.0)
        assert len(matches) == 3
        np.testing.assert_allclose(dists, 0.0, atol=1e-10)

    def test_noisy_match(self):
        """Small noise should still match."""
        rng = np.random.default_rng(42)
        pos1 = rng.uniform(100, 900, (20, 2))
        pos2 = pos1 + rng.normal(0, 1, pos1.shape)
        matches, dists = match_sources(pos1, pos2, max_dist=5.0)
        assert len(matches) >= 18
        assert np.all(dists < 5.0)

    def test_no_match_far(self):
        """Distant positions should not match."""
        pos1 = np.array([[0, 0], [1, 1]])
        pos2 = np.array([[1000, 1000], [2000, 2000]])
        matches, dists = match_sources(pos1, pos2, max_dist=10.0)
        assert len(matches) == 0
