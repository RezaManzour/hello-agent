class LLMError(Exception):
    """Base exception for LLM-related errors."""


class LLMAuthenticationError(LLMError):
    """Raised when LLM authentication fails."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider rate-limits the request."""


class LLMProviderError(LLMError):
    """Raised when the LLM provider returns an unexpected error."""


class GuardrailViolationError(Exception):
    """Raised when a prompt violates a registered Agent guardrail."""
