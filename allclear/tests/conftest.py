"""Shared test fixtures for allclear tests."""

import pathlib

import pytest
from astropy.coordinates import EarthLocation
import astropy.units as u


EXAMPLE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "example_images"


@pytest.fixture
def observer_location():
    """EarthLocation for Haleakala, Hawaii."""
    return EarthLocation(lat=20.7458 * u.deg, lon=-156.4317 * u.deg, height=3055 * u.m)


@pytest.fixture
def example_fits_path():
    """Path to a known-good example FITS file."""
    return EXAMPLE_DIR / "2023_11_19__00_00_11.fits"


@pytest.fixture
def example_dir():
    """Path to the example_images directory."""
    return EXAMPLE_DIR
