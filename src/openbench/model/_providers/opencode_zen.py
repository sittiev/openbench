"""OpenCode Zen provider implementation."""

import os
from typing import Any

from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model import GenerateConfig


class OpenCodeZenAPI(OpenAICompatibleAPI):
    """OpenCode Zen provider - OpenAI-compatible inference.

    Uses OpenAI-compatible API with OpenCode Zen-specific configuration.
    Only models served via ``/chat/completions`` work here
    (e.g. qwen3.8-max, deepseek-v4-flash, kimi-k2.5, glm-5.3).
    Models served via ``/responses`` or ``/messages`` need native APIs.
    Docs: https://opencode.ai/docs/zen/
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
        model_name_clean = model_name.replace("opencode_zen/", "", 1)

        # Set defaults for OpenCode Zen
        base_url = base_url or os.environ.get(
            "OPENCODE_ZEN_BASE_URL", "https://opencode.ai/zen/v1"
        )
        api_key = (
            api_key
            or os.environ.get("OPENCODE_ZEN_API_KEY")
            or os.environ.get("OPENCODE_API_KEY")
        )

        if not api_key:
            raise ValueError(
                "OpenCode Zen API key not found. "
                "Set OPENCODE_ZEN_API_KEY environment variable."
            )

        super().__init__(
            model_name=model_name_clean,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="opencode_zen",
            service_base_url="https://opencode.ai/zen/v1",
            **model_args,
        )

    def service_model_name(self) -> str:
        """Return model name without service prefix."""
        return self.model_name
