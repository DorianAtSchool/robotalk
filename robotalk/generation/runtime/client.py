"""Provide the Gemini generation client and its usage record."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robotalk.utils import (
    normalize_traffic_type,
    parse_dotenv_value,
    usage_field,
)

GOOGLE_GENAI_SDK = "google-genai"
SUPPORTED_GENERATION_SDKS = (GOOGLE_GENAI_SDK,)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"
DEFAULT_TRAFFIC_TYPE = "ON_DEMAND"
# google-genai reads Vertex routing from environment variables.
GOOGLE_GENAI_VERTEX_ENV_VAR = "GOOGLE_GENAI_USE_VERTEXAI"
class TrajectoryGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationUsage:
    prompt_tokens: int | None = None
    candidates_tokens: int | None = None
    thoughts_tokens: int | None = None
    cached_content_tokens: int | None = None
    tool_use_prompt_tokens: int | None = None
    total_tokens: int | None = None
    traffic_type: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    payload: Any
    usage: GenerationUsage | None = None


class BaseGenerationClient:
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, Any] | None,
        temperature: float,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> Any:
        raise NotImplementedError


def _generation_error_status_code(exc: Exception) -> int | None:
    """Extracts one HTTP status code from SDK exceptions with varying shapes."""

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    response_status_code = getattr(response, "status_code", None)
    if isinstance(response_status_code, int):
        return response_status_code

    return None


def load_dotenv_file(
    path: Path | None = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    dotenv_path = path or DEFAULT_DOTENV_PATH
    loaded_values: dict[str, str] = {}
    if not dotenv_path.exists():
        return loaded_values

    # Mirror common dotenv parsing without adding another runtime dependency.
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = parse_dotenv_value(raw_value)
        loaded_values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value

    return loaded_values


def configure_google_genai_environment(
    *,
    project: str | None,
    location: str | None,
) -> None:
    # Some models are served only by the Gemini Developer API and 404 on every
    # Vertex region -- gemini-robotics-er-2-preview is one. When an API key is
    # present, route to the developer endpoint instead of Vertex. project and
    # location are meaningless there, so leave them unset.
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        os.environ[GOOGLE_GENAI_VERTEX_ENV_VAR] = "False"
        os.environ["GOOGLE_API_KEY"] = api_key
        return
    os.environ[GOOGLE_GENAI_VERTEX_ENV_VAR] = "True"
    if project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project
    if location:
        os.environ["GOOGLE_CLOUD_LOCATION"] = location


def validate_google_auth(project: str | None) -> None:
    configured_project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    try:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError
    except ImportError:
        # Keep local tests working even when google-auth is not installed.
        return

    try:
        _, detected_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except DefaultCredentialsError as exc:
        raise TrajectoryGenerationError(
            "Google ADC credentials were not found. Use either "
            "`gcloud auth application-default login` or set "
            "`GOOGLE_APPLICATION_CREDENTIALS` in your environment or .env."
        ) from exc

    if not configured_project and not detected_project:
        raise TrajectoryGenerationError(
            "Google Cloud project was not found. Set --project or "
            "GOOGLE_CLOUD_PROJECT."
        )


def build_generation_usage_metadata(
    usage_metadata: Any,
    *,
    default_traffic_type: str = DEFAULT_TRAFFIC_TYPE,
) -> GenerationUsage | None:
    if usage_metadata is None:
        return None
    return GenerationUsage(
        prompt_tokens=usage_field(
            usage_metadata, "prompt_token_count", "promptTokenCount"
        ),
        candidates_tokens=usage_field(
            usage_metadata,
            "candidates_token_count",
            "candidatesTokenCount",
        ),
        thoughts_tokens=usage_field(
            usage_metadata, "thoughts_token_count", "thoughtsTokenCount"
        ),
        cached_content_tokens=usage_field(
            usage_metadata,
            "cached_content_token_count",
            "cachedContentTokenCount",
        ),
        tool_use_prompt_tokens=usage_field(
            usage_metadata,
            "tool_use_prompt_token_count",
            "toolUsePromptTokenCount",
        ),
        total_tokens=usage_field(
            usage_metadata, "total_token_count", "totalTokenCount"
        ),
        traffic_type=(
            normalize_traffic_type(
                usage_field(usage_metadata, "traffic_type", "trafficType")
            )
            or default_traffic_type
        ),
    )


DEFAULT_GENERATION_TIMEOUT_SEC = 300
"""Default per-request wall-clock cap for SDK calls.

Without this, a stalled HTTP request hangs the worker forever (silent TCP
drops, server-side hangs, idle timeouts at intermediate proxies). Five
minutes is generous enough for the largest trajectory generations we have
observed and short enough that retries can recover within `--max-retries`.
"""


def build_raw_google_genai_client(
    project: str | None,
    location: str,
    *,
    timeout_sec: int | None = DEFAULT_GENERATION_TIMEOUT_SEC,
) -> Any:
    try:
        from google import genai
        from google.genai.types import HttpOptions
    except ImportError as exc:
        raise TrajectoryGenerationError(
            "google-genai is not installed. Install it with "
            "`uv pip install google-genai`."
        ) from exc

    configure_google_genai_environment(project=project, location=location)
    # ADC is the Vertex credential. In API-key mode it is irrelevant, and
    # requiring it would reject a perfectly good developer-endpoint client.
    if os.environ.get(GOOGLE_GENAI_VERTEX_ENV_VAR) != "False":
        validate_google_auth(project)
    # Preview models are only served on v1beta -- gemini-robotics-er-2-preview
    # 404s on v1 with the very key that lists it. Vertex keeps v1, which is
    # what every existing run used.
    default_api_version = (
        "v1beta"
        if os.environ.get(GOOGLE_GENAI_VERTEX_ENV_VAR) == "False"
        else "v1"
    )
    http_options_kwargs: dict[str, Any] = {
        "api_version": os.environ.get(
            "GOOGLE_GENAI_API_VERSION", default_api_version
        )
    }
    if timeout_sec is not None:
        # HttpOptions.timeout is in milliseconds.
        http_options_kwargs["timeout"] = int(timeout_sec) * 1000
    return genai.Client(
        http_options=HttpOptions(**http_options_kwargs),
    )


DEFAULT_GOOGLE_GENAI_MAX_OUTPUT_TOKENS = 32768


class GoogleGenAIClient(BaseGenerationClient):
    def __init__(
        self,
        project: str | None,
        location: str,
        *,
        timeout_sec: int | None = DEFAULT_GENERATION_TIMEOUT_SEC,
    ):
        self._client = build_raw_google_genai_client(
            project=project,
            location=location,
            timeout_sec=timeout_sec,
        )

    def _build_generation_config(
        self,
        *,
        temperature: float,
        response_schema: dict[str, Any] | None,
        thinking_level: str | None,
        thinking_budget: int | None,
    ) -> dict[str, Any]:
        """Builds one google-genai generation config payload."""

        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": DEFAULT_GOOGLE_GENAI_MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json",
        }
        # Pass `response_schema` only when the caller supplied one. Phase 1
        # (TaskSpec generation) omits the schema because the spec uses dicts
        # with dynamic string keys that Gemini's structured-output mode
        # cannot express — keeping the schema would force Gemini to return
        # `{}` for any unenumerated object.
        if response_schema is not None:
            config["response_schema"] = response_schema
        thinking_config: dict[str, Any] = {}
        if thinking_level is not None:
            thinking_config["thinking_level"] = thinking_level
        if thinking_budget is not None:
            thinking_config["thinking_budget"] = thinking_budget
        if thinking_config:
            config["thinking_config"] = thinking_config
        return config

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, Any] | None,
        temperature: float,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> Any:
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=self._build_generation_config(
                    temperature=temperature,
                    response_schema=response_schema,
                    thinking_level=thinking_level,
                    thinking_budget=thinking_budget,
                ),
            )
        except Exception as exc:
            message = str(exc)
            status_code = _generation_error_status_code(exc)
            if (
                "403 PERMISSION_DENIED" in message
                and "aiplatform.endpoints.predict" in message
            ):
                raise TrajectoryGenerationError(
                    "The configured Google principal does not have permission to call "
                    "Vertex AI `GenerateContent`. Grant a role that includes "
                    "`aiplatform.endpoints.predict` on project "
                    f"`{os.environ.get('GOOGLE_CLOUD_PROJECT', '<unknown-project>')}` "
                    "such as Vertex AI User, then retry."
                ) from exc
            if status_code == 400 and "INVALID_ARGUMENT" in message:
                raise TrajectoryGenerationError(
                    "Vertex AI rejected the generation request with "
                    "`400 INVALID_ARGUMENT`."
                    f" Original error: {message}"
                ) from exc
            raise
        # Usage metadata is optional and field names vary a bit across SDK releases.
        usage = build_generation_usage_metadata(
            getattr(response, "usage_metadata", None)
        )
        return GenerationResult(
            payload=getattr(response, "text", None) or str(response),
            usage=usage,
        )


def build_generation_client(
    sdk: str,
    project: str | None,
    location: str,
    *,
    timeout_sec: int | None = DEFAULT_GENERATION_TIMEOUT_SEC,
) -> BaseGenerationClient:
    if sdk == GOOGLE_GENAI_SDK:
        return GoogleGenAIClient(
            project=project,
            location=location,
            timeout_sec=timeout_sec,
        )
    raise TrajectoryGenerationError(
        f"Unsupported SDK '{sdk}'. Expected one of: "
        f"{', '.join(SUPPORTED_GENERATION_SDKS)}."
    )
