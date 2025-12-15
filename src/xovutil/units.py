"""Small collection of angle and time conversion helpers."""

from typing import Union


def rad2as(x: Union[float, int]) -> float:
    """Convert radians to arcseconds."""

    return x * 206265.


def as2rad(x: Union[float, int]) -> float:
    """Convert arcseconds to radians."""

    return x / 206265.


def deg2as(x: Union[float, int]) -> float:
    """Convert degrees to arcseconds."""

    return x * 3600.


def as2deg(x: Union[float, int]) -> float:
    """Convert arcseconds to degrees."""

    return x / 3600.


def day2sec(x: Union[float, int]) -> float:
    """Convert days to seconds."""

    return x * 86400.


def sec2day(x: Union[float, int]) -> float:
    """Convert seconds to days."""

    return x / 86400.
