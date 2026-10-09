"""memetrics: measure memetic claims on social-media data."""

__version__ = "0.1.0"

from .result import Result  # noqa: E402
from .schema import SchemaError, read_items, validate, write_items  # noqa: E402

__all__ = ["Result", "SchemaError", "read_items", "validate", "write_items", "__version__"]
