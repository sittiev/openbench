"""Charm Hyper provider implementation."""

import os
from typing import Any

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import GenerateConfig


class HyperAPI(OpenAICompatibleAPI):
    """Charm Hyper provider - OpenAI-compatible inference.

    Uses OpenAI-compatible API with Hyper-specific configuration.
    Official Crush provider. Key format: ``sk-hyper-...``.
    Docs: https://console.charm.land/docs
    Models: GET https://hyper.charm.land/v1/models (public).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        # Extract model name without service prefix
        model_name_clean = model_name.replace("hyper/", "", 1)

        # Set defaults for Hyper
        base_url = base_url or os.environ.get(
            "HYPER_BASE_URL", "https://hyper.charm.land/v1"
        )
        api_key = api_key or os.environ.get("HYPER_API_KEY")

        if not api_key:
            raise ValueError(
                "Hyper API key not found. Set HYPER_API_KEY environment variable."
            )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="hyper",
            service_base_url="https://hyper.charm.land/v1",
            **model_args,
        )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name
