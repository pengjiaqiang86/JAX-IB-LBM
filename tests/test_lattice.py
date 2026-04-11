"""
Unit tests for lattice descriptors.
"""

import jax.numpy as jnp
import pytest

from src.core.lattice import D2Q9, D3Q19, D3Q27


def test_d2q9_weights_sum():
    assert jnp.isclose(jnp.sum(D2Q9.w), 1.0, atol=1e-7)


def test_d2q9_opposite():
    for q in range(D2Q9.Q):
        opp = D2Q9.opp[q]
        assert jnp.all(D2Q9.c[opp] == -D2Q9.c[q]), f"opp failed for q={q}"


def test_d3q19_weights_sum():
    assert jnp.isclose(jnp.sum(D3Q19.w), 1.0, atol=1e-7)


def test_d3q19_opposite():
    for q in range(D3Q19.Q):
        opp = D3Q19.opp[q]
        assert jnp.all(D3Q19.c[opp] == -D3Q19.c[q])


def test_d3q27_weights_sum():
    assert jnp.isclose(jnp.sum(D3Q27.w), 1.0, atol=1e-7)


def test_d3q27_opposite():
    for q in range(D3Q27.Q):
        opp = D3Q27.opp[q]
        assert jnp.all(D3Q27.c[opp] == -D3Q27.c[q])
