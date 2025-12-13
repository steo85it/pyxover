"""Small collection of angle and time conversion helpers."""


def rad2as(x: float | int) -> float:
    """Convert radians to arcseconds."""

    return x * 206265.


def as2rad(x: float | int) -> float:
    """Convert arcseconds to radians."""

    return x / 206265.


def deg2as(x: float | int) -> float:
    """Convert degrees to arcseconds."""

    return x * 3600.


def as2deg(x: float | int) -> float:
    """Convert arcseconds to degrees."""

    return x / 3600.


def day2sec(x: float | int) -> float:
    """Convert days to seconds."""

    return x * 86400.


def sec2day(x: float | int) -> float:
    """Convert seconds to days."""

    return x / 86400.
