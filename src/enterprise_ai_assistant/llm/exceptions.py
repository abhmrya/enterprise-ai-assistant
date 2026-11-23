
class LLMError(Exception):
    """Base exception for all LLM-related failures."""


class LLMAuthenticationError(LLMError):
    """Raised when provider authentication fails."""


class LLMRateLimitError(LLMError):
    """Raised when the provider rate limit is exceeded."""


class LLMUnavailableError(LLMError):
    """Raised when the provider service is temporarily unavailable."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds its timeout."""


class LLMResponseError(LLMError):
    """Raised when the provider returns an unusable response."""


class LLMProviderError(LLMError):
    """Raised for other provider-specific failures."""
