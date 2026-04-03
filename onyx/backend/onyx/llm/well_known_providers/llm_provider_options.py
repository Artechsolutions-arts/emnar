import json
import pathlib
import threading
import time

from onyx.llm.constants import LlmProviderNames
from onyx.llm.constants import PROVIDER_DISPLAY_NAMES
from onyx.llm.constants import WELL_KNOWN_PROVIDER_NAMES
from onyx.llm.utils import get_max_input_tokens
from onyx.llm.utils import model_supports_image_input
from onyx.llm.well_known_providers.auto_update_models import LLMRecommendations
from onyx.llm.well_known_providers.auto_update_service import (
    fetch_llm_recommendations_from_github,
)
from onyx.llm.well_known_providers.constants import OLLAMA_PROVIDER_NAME, VERTEXAI_PROVIDER_NAME
from onyx.llm.well_known_providers.models import WellKnownLLMProviderDescriptor
from onyx.server.manage.llm.models import ModelConfigurationView
from onyx.utils.logger import setup_logger

logger = setup_logger()

_RECOMMENDATIONS_CACHE_TTL_SECONDS = 300
_recommendations_cache_lock = threading.Lock()
_cached_recommendations: LLMRecommendations | None = None
_cached_recommendations_time: float = 0.0


def _get_provider_to_models_map() -> dict[str, list[str]]:
    """Lazy-load provider model mappings.
    Dynamic providers return empty lists here because their models are fetched from the API.
    """
    return {
        OLLAMA_PROVIDER_NAME: [],  # Dynamic - fetched from Ollama API
    }


def _load_bundled_recommendations() -> LLMRecommendations:
    json_path = pathlib.Path(__file__).parent / "recommended-models.json"
    with open(json_path, "r") as f:
        json_config = json.load(f)
    return LLMRecommendations.model_validate(json_config)


def get_recommendations() -> LLMRecommendations:
    """Get the recommendations, with an in-memory cache to avoid
    hitting GitHub on every API request."""
    global _cached_recommendations, _cached_recommendations_time

    now = time.monotonic()
    if (
        _cached_recommendations is not None
        and (now - _cached_recommendations_time) < _RECOMMENDATIONS_CACHE_TTL_SECONDS
    ):
        return _cached_recommendations

    with _recommendations_cache_lock:
        # Double-check after acquiring lock
        if (
            _cached_recommendations is not None
            and (time.monotonic() - _cached_recommendations_time)
            < _RECOMMENDATIONS_CACHE_TTL_SECONDS
        ):
            return _cached_recommendations

        recommendations_from_github = fetch_llm_recommendations_from_github()
        result = recommendations_from_github or _load_bundled_recommendations()

        _cached_recommendations = result
        _cached_recommendations_time = time.monotonic()
        return result


def is_obsolete_model(model_name: str, provider: str) -> bool:
    """Check if a model is obsolete and should be filtered out.

    Filters models that are 2+ major versions behind or deprecated.
    This is the single source of truth for obsolete model detection.
    """
    model_lower = model_name.lower()

    # OpenAI obsolete models
    if provider == LlmProviderNames.OPENAI:
        # GPT-3 models are obsolete
        if "gpt-3" in model_lower:
            return True
        # Legacy models
        deprecated = {
            "text-davinci-003",
            "text-davinci-002",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
            "davinci",
            "curie",
            "babbage",
            "ada",
        }
        if model_lower in deprecated:
            return True

    # Anthropic obsolete models
    if provider == LlmProviderNames.ANTHROPIC:
        if "claude-2" in model_lower or "claude-instant" in model_lower:
            return True

    # Vertex AI obsolete models
    if provider == LlmProviderNames.VERTEX_AI:
        if "gemini-1.0" in model_lower:
            return True
        if "palm" in model_lower or "bison" in model_lower:
            return True

    return False


# Removed cloud provider model fetching functions to enforce on-premises Ollama only.


def model_configurations_for_provider(
    provider_name: str, llm_recommendations: LLMRecommendations
) -> list[ModelConfigurationView]:
    recommended_visible_models = llm_recommendations.get_visible_models(provider_name)
    recommended_visible_models_names = [m.name for m in recommended_visible_models]

    # Preserve provider-defined ordering while de-duplicating.
    model_names: list[str] = []
    seen_model_names: set[str] = set()
    for model_name in (
        fetch_models_for_provider(provider_name) + recommended_visible_models_names
    ):
        if model_name in seen_model_names:
            continue
        seen_model_names.add(model_name)
        model_names.append(model_name)

    # Vertex model list can be large and mixed-vendor; alphabetical ordering
    # makes model discovery easier in admin selection UIs.
    if provider_name == VERTEXAI_PROVIDER_NAME:
        model_names = sorted(model_names, key=str.lower)

    return [
        ModelConfigurationView(
            name=model_name,
            is_visible=model_name in recommended_visible_models_names,
            max_input_tokens=get_max_input_tokens(model_name, provider_name),
            supports_image_input=model_supports_image_input(model_name, provider_name),
        )
        for model_name in model_names
    ]


def fetch_available_well_known_llms() -> list[WellKnownLLMProviderDescriptor]:
    llm_recommendations = get_recommendations()

    well_known_llms = []
    for provider_name in WELL_KNOWN_PROVIDER_NAMES:
        model_configurations = model_configurations_for_provider(
            provider_name, llm_recommendations
        )
        well_known_llms.append(
            WellKnownLLMProviderDescriptor(
                name=provider_name,
                known_models=model_configurations,
                recommended_default_model=llm_recommendations.get_default_model(
                    provider_name
                ),
            )
        )
    return well_known_llms


def fetch_models_for_provider(provider_name: str) -> list[str]:
    return _get_provider_to_models_map().get(provider_name, [])


def fetch_model_names_for_provider_as_set(provider_name: str) -> set[str] | None:
    model_names = fetch_models_for_provider(provider_name)
    return set(model_names) if model_names else None


def fetch_visible_model_names_for_provider_as_set(
    provider_name: str,
) -> set[str] | None:
    """Get visible model names for a provider.

    Note: Since we no longer maintain separate visible model lists,
    this returns all models (same as fetch_model_names_for_provider_as_set).
    Kept for backwards compatibility with alembic migrations.
    """
    return fetch_model_names_for_provider_as_set(provider_name)


def get_provider_display_name(provider_name: str) -> str:
    """Get human-friendly display name for an Onyx-supported provider.

    First checks Onyx-specific display names, then falls back to
    PROVIDER_DISPLAY_NAMES from constants.
    """
    # Display names for Onyx-supported LLM providers (used in admin UI provider selection).
    # These override PROVIDER_DISPLAY_NAMES for Onyx-specific branding.
    _ONYX_PROVIDER_DISPLAY_NAMES: dict[str, str] = {
        OLLAMA_PROVIDER_NAME: "Ollama",
    }

    if provider_name in _ONYX_PROVIDER_DISPLAY_NAMES:
        return _ONYX_PROVIDER_DISPLAY_NAMES[provider_name]
    return PROVIDER_DISPLAY_NAMES.get(
        provider_name.lower(), provider_name.replace("_", " ").title()
    )


def fetch_default_model_for_provider(provider_name: str) -> str | None:
    """Fetch the default model for a provider.

    First checks the GitHub-hosted recommended-models.json config (via fetch_github_config),
    then falls back to hardcoded defaults if unavailable.
    """
    llm_recommendations = get_recommendations()
    default_model = llm_recommendations.get_default_model(provider_name)
    return default_model.name if default_model else None
