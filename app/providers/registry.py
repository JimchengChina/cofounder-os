"""Provider registry — manages available providers with optional fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from app.models import Provider
from app.providers.base import BaseProvider, ProviderError
from app.providers.openai_compat import OpenAICompatProvider

if TYPE_CHECKING:
    from app.models import ChatResponse


class ProviderRegistry:
    """Registry of configured AI providers."""

    def __init__(self) -> None:
        self._providers: dict[Provider, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        """Register a provider instance."""
        self._providers[provider.name] = provider

    def get(self, provider: Provider) -> BaseProvider | None:
        """Return a provider by enum, or None if not registered."""
        return self._providers.get(provider)

    def all(self) -> Sequence[BaseProvider]:
        """Return all registered providers."""
        return list(self._providers.values())

    def clear(self) -> None:
        """Remove all registered providers (useful for tests)."""
        self._providers.clear()

    async def complete_with_fallback(
        self,
        preferred: Provider,
        **kwargs,
    ) -> tuple[ChatResponse, Provider]:
        """Try preferred provider first, fall back to others on failure."""
        tried: list[Provider] = []

        # Try preferred first
        provider = self.get(preferred)
        if provider is not None:
            try:
                response = await provider.complete(**kwargs)
                return response, provider.name
            except ProviderError:
                tried.append(preferred)

        # Fall back to remaining providers
        for p in self.all():
            if p.name in tried:
                continue
            try:
                response = await p.complete(**kwargs)
                return response, p.name
            except ProviderError:
                tried.append(p.name)
                continue

        raise ProviderError(
            f"All providers failed after trying: {[t.value for t in tried]}"
        )

    async def health_status(self) -> list[dict]:
        """Return health info for all registered providers."""
        results = []
        for provider in self.all():
            status, latency = await provider.health()
            results.append(
                {
                    "provider": provider.name.value,
                    "status": status,
                    "latency_ms": latency,
                }
            )
        return results


# Module-level default registry — tests can replace this via set_registry()
_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Return the default provider registry, creating it if needed."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def set_registry(registry: ProviderRegistry | None) -> None:
    """Replace the global registry (primarily for tests)."""
    global _registry
    _registry = registry
