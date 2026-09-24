class BackendError(Exception):
    """Base application error."""


class ValidationError(BackendError):
    """Raised when request or file validation fails."""


class StorageError(BackendError):
    """Raised when persistence fails."""


class RetrievalError(BackendError):
    """Raised when retrieval or graph lookup fails."""


class IngestionError(BackendError):
    """Raised when document processing fails."""


class AgentExecutionError(BackendError):
    """Raised when the control loop fails."""
