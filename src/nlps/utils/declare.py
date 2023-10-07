from .design_patterns import singleton


@singleton
class _UNKNOWN_CLASS:
    pass


UNKNOWN = _UNKNOWN_CLASS()

