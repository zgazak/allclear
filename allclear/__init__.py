"""AllClear — all-sky camera cloud detection via star matching."""

__version__ = "0.2.0"

from .api import get_sky_transmission, get_test_transmission, SkyTransmissionResult
from .transmission import TransmissionMap

__all__ = [
    "get_sky_transmission",
    "get_test_transmission",
    "SkyTransmissionResult",
    "TransmissionMap",
]
