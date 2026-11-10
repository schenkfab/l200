"""Google ADK Blog Post Writer Agent.

Implements an enterprise-grade multi-agent workflow for researching, outlining,
drafting, editing, and publishing blog posts with:
- Explicit JSON/Pydantic schemas & guided LLM error recovery
- History compaction, persistent SQLite session state, and async memory operations
- Strategic model routing (Flash vs. Pro), security guardrails, evaluation plugin, and HITL hooks
- Structured logging, intent capturing, OpenTelemetry distributed tracing, and PII redaction
- Google Cloud Secret Manager integration
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools import FunctionTool, ToolContext, google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

# Optional OpenTelemetry distributed tracing setup with graceful fallback
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    _resource = Resource.create({"service.name": "blog-writer-adk-agent", "service.version": "2.0.0"})
    _provider = TracerProvider(resource=_resource)
    if os.getenv("ENABLE_CONSOLE_TRACING", "false").lower() == "true":
        _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(_provider)
    tracer = trace.get_tracer("blog_writer.agent")
except ImportError:  # pragma: no cover
    from contextlib import nullcontext

    class _NoopTracer:
        def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
            return nullcontext()

    tracer = _NoopTracer()  # type: ignore[assignment]


# ============================================================================
# 1. OBSERVABILITY, PII REDACTION & LOGGING
# ============================================================================

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "API_KEY": re.compile(r"\b(?:AIza[0-9A-Za-z-_]{35}|ya29\.[0-9A-Za-z\-_]+)\b"),
}


def redact_pii(text: str) -> str:
    """Redacts sensitive PII and secrets (emails, phones, SSNs, cards, API keys) from text."""
    if not text:
        return text
    sanitized = text
    for label, pattern in PII_PATTERNS.items():
        sanitized = pattern.sub(f"[REDACTED_{label}]", sanitized)
    return sanitized


class PiiRedactingJsonFormatter(logging.Formatter):
    """Structured JSON log formatter that automatically scrubs PII from all log records."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_pii(record.getMessage()),
            "module": record.module,
            "function": record.funcName,
        }
        if hasattr(record, "intent"):
            payload["intent"] = getattr(record, "intent")
        if hasattr(record, "agent_name"):
            payload["agent_name"] = getattr(record, "agent_name")
        return json.dumps(payload)


logger = logging.getLogger("blog_writer")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(PiiRedactingJsonFormatter())
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ============================================================================
# 2. SECRET MANAGER INTEGRATION
# ============================================================================


def load_secret_from_gcp(secret_id: str, project_id: str | None = None, version: str = "latest") -> str | None:
    """Retrieves a secret from Google Cloud Secret Manager with fallback to environment variables.

    Args:
        secret_id: Name of the secret in GCP Secret Manager (e.g. 'GOOGLE_API_KEY').
        project_id: Optional GCP project ID. Defaults to GOOGLE_CLOUD_PROJECT.
        version: Secret version to access (default: 'latest').

    Returns:
        The resolved secret string or None if unavailable.
    """
    with tracer.start_as_current_span("secret_manager.load_secret") as span:
        if span is not None and hasattr(span, "set_attribute"):
            span.set_attribute("secret.id", secret_id)

        resolved_project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if resolved_project and os.getenv("USE_SECRET_MANAGER", "true").lower() == "true":
            try:
                from google.cloud import secretmanager  # type: ignore[import-untyped]

                client = secretmanager.SecretManagerServiceClient()
                secret_path = f"projects/{resolved_project}/secrets/{secret_id}/versions/{version}"
                response = client.access_secret_version(request={"name": secret_path})
                secret_value = response.payload.data.decode("UTF-8").strip()
                logger.info("Loaded secret '%s' from Google Cloud Secret Manager.", secret_id)
                return secret_value
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Secret Manager lookup for '%s' fell back to environment variable: %s",
                    secret_id,
                    type(exc).__name__,
                )

        return os.getenv(secret_id)


def initialize_runtime_secrets() -> None:
    """Ensures GOOGLE_API_KEY is populated via Secret Manager when running in cloud environments."""
    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CLOUD_PROJECT"):
        api_key = load_secret_from_gcp("GOOGLE_API_KEY")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key


initialize_runtime_secrets()


# ============================================================================
# 3. STRATEGIC MODEL ROUTING CONFIGURATION
# ============================================================================

# Fast, cost-efficient model for routing, web search synthesis, and structured outlining
FAST_MODEL = os.getenv("BLOG_FAST_MODEL", "gemini-2.5-flash")

# High-reasoning model for long-form creative drafting and nuanced editorial/SEO critique
REASONING_MODEL = os.getenv("BLOG_REASONING_MODEL", "gemini-2.5-pro")


# ============================================================================
# 4. EXPLICIT JSON & PYDANTIC SCHEMAS FOR STRICT VALIDATION
# ============================================================================


class BlogMetricsInput(BaseModel):
    """Strict input validation schema for blog metrics and SEO analysis."""

    content: str = Field(
        ...,
        min_length=20,
        description="The full Markdown text of the blog post draft (minimum 20 characters).",
    )
    target_keyword: str = Field(
        default="",
        max_length=120,
        description="Optional primary SEO keyword or keyphrase to evaluate.",
    )


class SaveBlogPostInput(BaseModel):
    """Strict input validation schema for saving a blog post to disk."""

    filename: str = Field(
        ...,
        min_length=3,
        max_length=100,
        pattern=r"^[a-zA-Z0-9._-]+$",
        description="Alphanumeric filename with hyphens/underscores (e.g., 'ai-agents-guide.md').",
    )
    content: str = Field(
        ...,
        min_length=50,
        description="Complete publication-ready Markdown content of the blog post.",
    )
    human_approved: bool = Field(
        default=False,
        description="Whether the human user has explicitly reviewed and approved saving/publishing.",
    )


class MemoryEntryInput(BaseModel):
    """Strict input validation schema for asynchronous memory operations."""

    key: str = Field(
        ...,
        min_length=2,
        max_length=80,
        pattern=r"^[a-zA-Z0-9_.-]+$",
        description="Unique identifier key for the memory entry (e.g., 'brand_voice', 'past_topics').",
    )
    value: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Content to persist in long-term memory.",
    )
    category: str = Field(
        default="preference",
        pattern=r"^(preference|style_guide|research_fact|past_post)$",
        description="Memory category: one of 'preference', 'style_guide', 'research_fact', or 'past_post'.",
    )


class OutlineSection(BaseModel):
    """Schema for an individual blog section in a structured outline."""

    heading: str = Field(..., description="H2 section heading.")
    subheadings: list[str] = Field(default_factory=list, description="Optional H3 subheadings.")
    key_points: list[str] = Field(..., description="Bullet points summarizing arguments and examples.")


class BlogOutlineSchema(BaseModel):
    """Structured output schema for the blog outliner sub-agent."""

    title_options: list[str] = Field(..., description="Three catchy, SEO-friendly title options.")
    target_audience: str = Field(..., description="Intended reader persona.")
    tone: str = Field(..., description="Writing tone (e.g., authoritative, conversational, technical).")
    primary_keyword: str = Field(..., description="Primary SEO keyword or phrase.")
    hook: str = Field(..., description="Opening hook and introduction summary.")
    sections: list[OutlineSection] = Field(..., description="Ordered H2/H3 section breakdown.")
    call_to_action: str = Field(..., description="Closing takeaway and reader call-to-action.")


TOOL_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "analyze_blog_metrics": BlogMetricsInput.model_json_schema(),
    "save_blog_post": SaveBlogPostInput.model_json_schema(),
    "save_writer_memory": MemoryEntryInput.model_json_schema(),
}


# ============================================================================
# 5. SECURITY GUARDRAILS, INTENT CAPTURING, COMPACTION & HITL CALLBACKS
# ============================================================================

PROMPT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|system)\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+prompt|api\s+key|secret)", re.IGNORECASE),
    re.compile(r"<\s*script\b", re.IGNORECASE),
]

INTENT_KEYWORDS: dict[str, list[str]] = {
    "research": ["research", "search", "find", "fact", "trend", "statistics", "source"],
    "outline": ["outline", "structure", "plan", "sections", "brainstorm", "titles"],
    "draft": ["write", "draft", "create", "compose", "generate", "blog post"],
    "edit_seo": ["edit", "polish", "review", "seo", "keyword", "proofread", "metrics"],
    "publish_save": ["save", "export", "publish", "file", "download", "approve"],
}

MAX_HISTORY_TURNS = 12


def classify_user_intent(user_text: str) -> str:
    """Detects primary user intent for observability, routing, and session tracking."""
    lowered = user_text.lower()
    scores: dict[str, int] = {
        intent: sum(1 for kw in keywords if kw in lowered)
        for intent, keywords in INTENT_KEYWORDS.items()
    }
    best_intent = max(scores, key=lambda k: scores[k])
    return best_intent if scores[best_intent] > 0 else "general_inquiry"


def compact_conversation_history(llm_request: LlmRequest, state: dict[str, Any]) -> None:
    """Compacts older conversation turns into a concise state summary when history exceeds threshold."""
    if not llm_request.contents or len(llm_request.contents) <= MAX_HISTORY_TURNS:
        return

    with tracer.start_as_current_span("context.compact_history") as span:
        older_turns = llm_request.contents[:-MAX_HISTORY_TURNS]
        recent_turns = llm_request.contents[-MAX_HISTORY_TURNS:]

        summary_snippets: list[str] = []
        for content in older_turns:
            role = getattr(content, "role", "user")
            parts = getattr(content, "parts", []) or []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    cleaned = redact_pii(text.strip().replace("\n", " "))
                    summary_snippets.append(f"[{role}]: {cleaned[:160]}")

        compacted_summary = " | ".join(summary_snippets[-8:])
        existing_summary = state.get("compacted_history_summary", "")
        combined_summary = f"{existing_summary} || {compacted_summary}".strip(" |")
        state["compacted_history_summary"] = combined_summary[-2000:]

        compaction_notice = types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"[System Context Summary of Earlier Conversation]: {state['compacted_history_summary']}"
                )
            ],
        )
        llm_request.contents = [compaction_notice, *recent_turns]

        if span is not None and hasattr(span, "set_attribute"):
            span.set_attribute("compaction.removed_turns", len(older_turns))
            span.set_attribute("compaction.remaining_turns", len(llm_request.contents))

        logger.info(
            "Compacted %d historical turns into session summary.",
            len(older_turns),
            extra={"agent_name": "history_compactor"},
        )


def security_and_context_before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Executes PII scrubbing, prompt-injection guardrails, intent capture, and history compaction."""
    with tracer.start_as_current_span("callback.before_model") as span:
        agent_name = getattr(callback_context, "agent_name", "unknown_agent")
        state = callback_context.state

        # 1. History compaction to prevent context window bloat
        compact_conversation_history(llm_request, state)

        # 2. Inspect latest user message for intent capture, PII redaction, and security guardrails
        if llm_request.contents:
            latest_content = llm_request.contents[-1]
            parts = getattr(latest_content, "parts", []) or []
            for part in parts:
                raw_text = getattr(part, "text", None)
                if not raw_text:
                    continue

                # Security guardrail: block prompt injection / system prompt extraction
                for pattern in PROMPT_INJECTION_PATTERNS:
                    if pattern.search(raw_text):
                        logger.warning(
                            "Blocked prompt injection attempt on agent '%s'.",
                            agent_name,
                            extra={"agent_name": agent_name, "intent": "blocked_security_violation"},
                        )
                        return LlmResponse(
                            content=types.Content(
                                role="model",
                                parts=[
                                    types.Part.from_text(
                                        text=(
                                            "Security Guardrail Triggered: Your request contains patterns "
                                            "that violate safety policies (such as overriding system "
                                            "instructions or exposing secrets). Please rephrase your "
                                            "blog writing request."
                                        )
                                    )
                                ],
                            )
                        )

                # Capture user intent in session state & observability logs
                detected_intent = classify_user_intent(raw_text)
                state["last_detected_intent"] = detected_intent
                intent_history = list(state.get("intent_history", []))
                intent_history.append(
                    {
                        "intent": detected_intent,
                        "agent": agent_name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                state["intent_history"] = intent_history[-25:]

                # Redact PII in-place before sending to the LLM
                sanitized_text = redact_pii(raw_text)
                if sanitized_text != raw_text:
                    part.text = sanitized_text
                    logger.info(
                        "Redacted PII from incoming prompt before LLM invocation.",
                        extra={"agent_name": agent_name, "intent": detected_intent},
                    )

                if span is not None and hasattr(span, "set_attribute"):
                    span.set_attribute("agent.name", agent_name)
                    span.set_attribute("user.intent", detected_intent)

                logger.info(
                    "Captured user intent '%s' for agent '%s'.",
                    detected_intent,
                    agent_name,
                    extra={"agent_name": agent_name, "intent": detected_intent},
                )

        return None


def pii_redaction_after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Scrubs accidental PII leakage from model responses before returning to the user."""
    with tracer.start_as_current_span("callback.after_model"):
        if llm_response and llm_response.content and llm_response.content.parts:
            for part in llm_response.content.parts:
                if getattr(part, "text", None):
                    part.text = redact_pii(part.text)
        return None


def hitl_and_tracing_before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict[str, Any] | None:
    """Enforces programmatic Human-in-the-Loop (HITL) checks and logs tool invocations."""
    tool_name = getattr(tool, "name", str(tool))
    with tracer.start_as_current_span(f"tool.before.{tool_name}") as span:
        if span is not None and hasattr(span, "set_attribute"):
            span.set_attribute("tool.name", tool_name)

        logger.info(
            "Invoking tool '%s' with keys=%s",
            tool_name,
            list(args.keys()),
            extra={"agent_name": getattr(tool_context, "agent_name", "tool_runner")},
        )

        # Programmatic Human-in-the-Loop gate for file export / publishing tools
        if tool_name == "save_blog_post":
            session_approved = bool(tool_context.state.get("human_approved_publish", False))
            arg_approved = bool(args.get("human_approved", False))
            if not (session_approved or arg_approved):
                logger.warning("HITL gate paused execution of '%s': awaiting human approval.", tool_name)
                return {
                    "status": "awaiting_human_approval",
                    "error_type": "HumanApprovalRequired",
                    "message": (
                        "Human-in-the-Loop check: Saving/publishing a blog post requires explicit user confirmation."
                    ),
                    "recovery_instruction": (
                        "Ask the user if they approve saving the blog post to disk. Once the user confirms, "
                        "call `record_human_approval(approved=True)` or pass `human_approved=True` to `save_blog_post`."
                    ),
                }
        return None


DEFAULT_SAFETY_SETTINGS: list[types.SafetySetting] = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
]


# ============================================================================
# 6. TOOLS WITH STRICT VALIDATION, ASYNC MEMORY & GUIDED ERROR RECOVERY
# ============================================================================

MEMORY_STORE_PATH = Path(os.getenv("BLOG_MEMORY_STORE_PATH", Path.cwd() / "output" / "long_term_memory.json"))
_memory_lock = asyncio.Lock()


def analyze_blog_metrics(
    content: str,
    target_keyword: str = "",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Analyzes a blog post draft for word count, reading time, structure, and SEO keyword usage.

    Validates inputs against `BlogMetricsInput` JSON schema (`TOOL_JSON_SCHEMAS['analyze_blog_metrics']`).

    Args:
        content: The full markdown text of the blog post draft (at least 20 characters).
        target_keyword: Optional primary SEO keyword or phrase to check against the draft.
        tool_context: Optional ADK ToolContext for persisting metrics in session state.

    Returns:
        A dictionary containing word count, estimated reading time in minutes,
        heading counts, paragraph count, and keyword density metrics, or a structured
        error response with `recovery_instruction` if validation fails.
    """
    with tracer.start_as_current_span("tool.analyze_blog_metrics") as span:
        try:
            validated = BlogMetricsInput(content=content, target_keyword=target_keyword)
            clean_content = redact_pii(validated.content)

            words = re.findall(r"\b\w+\b", clean_content)
            word_count = len(words)
            reading_time_minutes = max(1, round(word_count / 200))

            h1_headings = re.findall(r"^#\s+.+", clean_content, flags=re.MULTILINE)
            h2_headings = re.findall(r"^##\s+.+", clean_content, flags=re.MULTILINE)
            h3_headings = re.findall(r"^###\s+.+", clean_content, flags=re.MULTILINE)
            paragraphs = [p.strip() for p in clean_content.split("\n\n") if p.strip()]

            keyword_stats: dict[str, Any] = {}
            if validated.target_keyword.strip():
                kw_lower = validated.target_keyword.strip().lower()
                occurrences = len(re.findall(re.escape(kw_lower), clean_content.lower()))
                in_h1 = any(kw_lower in h.lower() for h in h1_headings)
                in_h2 = any(kw_lower in h.lower() for h in h2_headings)
                density_pct = round(
                    (occurrences * len(kw_lower.split()) / max(1, word_count)) * 100,
                    2,
                )
                keyword_stats = {
                    "target_keyword": validated.target_keyword,
                    "occurrences": occurrences,
                    "density_percent": density_pct,
                    "present_in_h1": in_h1,
                    "present_in_h2": in_h2,
                }

            result: dict[str, Any] = {
                "status": "success",
                "schema": "BlogMetricsInput",
                "word_count": word_count,
                "estimated_reading_time_minutes": reading_time_minutes,
                "heading_counts": {
                    "h1": len(h1_headings),
                    "h2": len(h2_headings),
                    "h3": len(h3_headings),
                },
                "paragraph_count": len(paragraphs),
                "seo_keyword_analysis": keyword_stats,
            }

            if tool_context is not None:
                tool_context.state["latest_blog_metrics"] = result

            if span is not None and hasattr(span, "set_attribute"):
                span.set_attribute("blog.word_count", word_count)

            return result

        except ValidationError as exc:
            logger.error("Input validation failed in analyze_blog_metrics: %s", exc)
            return {
                "status": "error",
                "error_type": "ValidationError",
                "validation_errors": exc.errors(),
                "expected_json_schema": BlogMetricsInput.model_json_schema(),
                "recovery_instruction": (
                    "Provide a valid `content` string of at least 20 characters and a `target_keyword` "
                    "under 120 characters, then retry `analyze_blog_metrics`."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected failure in analyze_blog_metrics")
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": redact_pii(str(exc)),
                "recovery_instruction": (
                    "Verify that `content` is plain Markdown text without malformed characters and retry."
                ),
            }


def record_human_approval(
    approved: bool,
    reviewer_note: str = "",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Records programmatic Human-in-the-Loop (HITL) approval in persistent session state.

    Args:
        approved: True if the user explicitly approved saving/publishing the blog post.
        reviewer_note: Optional feedback or approval note from the human reviewer.
        tool_context: ADK ToolContext used to store approval status in session state.

    Returns:
        A dictionary confirming the updated HITL approval state.
    """
    with tracer.start_as_current_span("tool.record_human_approval"):
        try:
            clean_note = redact_pii(reviewer_note)
            if tool_context is not None:
                tool_context.state["human_approved_publish"] = bool(approved)
                tool_context.state["human_reviewer_note"] = clean_note
                tool_context.state["human_approval_timestamp"] = datetime.now(timezone.utc).isoformat()
            logger.info("Recorded human approval state: approved=%s", approved)
            return {
                "status": "success",
                "human_approved_publish": bool(approved),
                "reviewer_note": clean_note,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": redact_pii(str(exc)),
                "recovery_instruction": "Pass a boolean `approved` value (True or False) and retry.",
            }


def save_blog_post(
    filename: str,
    content: str,
    human_approved: bool = False,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Saves the finished blog post Markdown to a local file in the `output/` directory.

    Validates inputs against `SaveBlogPostInput` JSON schema and enforces HITL approval.

    Args:
        filename: Desired alphanumeric filename (e.g. 'my-blog-post.md').
        content: Complete Markdown content of the blog post (minimum 50 characters).
        human_approved: Set to True once the human user confirms saving the file.
        tool_context: Optional ADK ToolContext for checking session-level HITL state.

    Returns:
        A dictionary with the save status and file path, or a structured error with
        `recovery_instruction` so the LLM can self-correct.
    """
    with tracer.start_as_current_span("tool.save_blog_post") as span:
        try:
            session_approved = bool(
                tool_context.state.get("human_approved_publish", False) if tool_context else False
            )
            effective_approval = human_approved or session_approved

            sanitized_filename = re.sub(r"[^a-zA-Z0-9._-]", "-", filename.strip())
            if sanitized_filename and not sanitized_filename.endswith(".md"):
                sanitized_filename += ".md"

            validated = SaveBlogPostInput(
                filename=sanitized_filename,
                content=content,
                human_approved=effective_approval,
            )

            if not validated.human_approved:
                return {
                    "status": "awaiting_human_approval",
                    "error_type": "HumanApprovalRequired",
                    "recovery_instruction": (
                        "Ask the human user for confirmation before saving. Once approved, call "
                        "`record_human_approval(approved=True)` or invoke `save_blog_post` with `human_approved=True`."
                    ),
                }

            clean_content = redact_pii(validated.content)
            output_dir = Path.cwd() / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / validated.filename
            output_path.write_text(clean_content, encoding="utf-8")

            if tool_context is not None:
                saved_files = list(tool_context.state.get("saved_blog_files", []))
                saved_files.append(str(output_path))
                tool_context.state["saved_blog_files"] = saved_files

            if span is not None and hasattr(span, "set_attribute"):
                span.set_attribute("file.path", str(output_path))

            logger.info("Saved blog post to %s", output_path)
            return {
                "status": "success",
                "file_path": str(output_path),
                "message": f"Blog post saved to {output_path}",
            }

        except ValidationError as exc:
            logger.error("Validation error in save_blog_post: %s", exc)
            return {
                "status": "error",
                "error_type": "ValidationError",
                "validation_errors": exc.errors(),
                "expected_json_schema": SaveBlogPostInput.model_json_schema(),
                "recovery_instruction": (
                    "Ensure `filename` is 3-100 alphanumeric/hyphen characters and `content` is at least "
                    "50 characters long, then retry `save_blog_post`."
                ),
            }
        except OSError as exc:
            logger.exception("Filesystem error in save_blog_post")
            return {
                "status": "error",
                "error_type": "FilesystemError",
                "error_message": redact_pii(str(exc)),
                "recovery_instruction": (
                    "Use a simple filename without directory traversal characters (e.g., 'post.md') and retry."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in save_blog_post")
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": redact_pii(str(exc)),
                "recovery_instruction": "Check arguments against SaveBlogPostInput schema and retry.",
            }


async def save_writer_memory(
    key: str,
    value: str,
    category: str = "preference",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Asynchronously stores author preferences, brand voice, or research facts in persistent memory.

    Args:
        key: Unique identifier key (e.g., 'brand_voice', 'target_persona').
        value: Memory value to persist (PII is automatically redacted).
        category: One of 'preference', 'style_guide', 'research_fact', or 'past_post'.
        tool_context: Optional ADK ToolContext to sync with `user:` scoped persistent state.

    Returns:
        Dictionary confirming asynchronous memory storage or guided error details.
    """
    with tracer.start_as_current_span("tool.async_save_writer_memory"):
        try:
            validated = MemoryEntryInput(key=key, value=value, category=category)
            sanitized_value = redact_pii(validated.value)
            entry = {
                "key": validated.key,
                "value": sanitized_value,
                "category": validated.category,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            async with _memory_lock:
                MEMORY_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
                existing: dict[str, Any] = {}
                if MEMORY_STORE_PATH.exists():
                    raw = await asyncio.to_thread(MEMORY_STORE_PATH.read_text, "utf-8")
                    existing = json.loads(raw) if raw.strip() else {}
                existing[validated.key] = entry
                await asyncio.to_thread(
                    MEMORY_STORE_PATH.write_text,
                    json.dumps(existing, indent=2),
                    "utf-8",
                )

            if tool_context is not None:
                tool_context.state[f"user:memory:{validated.key}"] = entry

            logger.info("Asynchronously persisted memory key='%s' category='%s'", validated.key, validated.category)
            return {
                "status": "success",
                "memory_entry": entry,
            }
        except ValidationError as exc:
            return {
                "status": "error",
                "error_type": "ValidationError",
                "validation_errors": exc.errors(),
                "expected_json_schema": MemoryEntryInput.model_json_schema(),
                "recovery_instruction": (
                    "Ensure `key` is alphanumeric/underscores, `value` is non-empty, and `category` is one of "
                    "'preference', 'style_guide', 'research_fact', or 'past_post'."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in async save_writer_memory")
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": redact_pii(str(exc)),
                "recovery_instruction": "Retry saving the memory with a valid key and plain-text value.",
            }


async def recall_writer_memory(
    query: str = "",
    category: str = "",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Asynchronously retrieves stored author preferences, style guides, and past topics from memory.

    Args:
        query: Optional search substring to filter memory keys or values.
        category: Optional category filter ('preference', 'style_guide', 'research_fact', 'past_post').
        tool_context: Optional ADK ToolContext to inspect session state.

    Returns:
        Dictionary containing matching memory entries.
    """
    with tracer.start_as_current_span("tool.async_recall_writer_memory"):
        try:
            async with _memory_lock:
                if not MEMORY_STORE_PATH.exists():
                    entries: dict[str, Any] = {}
                else:
                    raw = await asyncio.to_thread(MEMORY_STORE_PATH.read_text, "utf-8")
                    entries = json.loads(raw) if raw.strip() else {}

            query_lower = query.strip().lower()
            category_lower = category.strip().lower()
            matches: list[dict[str, Any]] = []
            for item in entries.values():
                if category_lower and item.get("category", "").lower() != category_lower:
                    continue
                if query_lower and (
                    query_lower not in item.get("key", "").lower()
                    and query_lower not in item.get("value", "").lower()
                ):
                    continue
                matches.append(item)

            if tool_context is not None:
                tool_context.state["last_memory_recall_count"] = len(matches)

            return {
                "status": "success",
                "count": len(matches),
                "memories": matches,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in async recall_writer_memory")
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": redact_pii(str(exc)),
                "recovery_instruction": "Pass a simple plain-text `query` string and retry `recall_writer_memory`.",
            }


# Wrap save_blog_post with FunctionTool and programmatic confirmation awareness
save_blog_post_tool = FunctionTool(func=save_blog_post)


# ============================================================================
# 7. MULTI-AGENT DEFINITIONS WITH STRATEGIC MODEL ROUTING
# ============================================================================

research_agent = Agent(
    name="blog_researcher",
    model=FAST_MODEL,
    description="Researches blog topics, current trends, facts, and examples using Google Search (Fast Model).",
    instruction=(
        "You are a specialist research analyst for a blog writing team.\n"
        "Use `google_search` to gather up-to-date facts, statistics, key angles, "
        "and credible examples on the requested topic.\n"
        "Summarize your findings into:\n"
        "1. Key Takeaways & Facts\n"
        "2. Target Audience Pain Points\n"
        "3. Compelling Angles & Hooks\n"
        "4. Suggested SEO Keywords"
    ),
    tools=[google_search],
    output_key="research_notes",
    before_model_callback=security_and_context_before_model_callback,
    after_model_callback=pii_redaction_after_model_callback,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    ),
)

outliner_agent = Agent(
    name="blog_outliner",
    model=FAST_MODEL,
    description="Creates structured blog post outlines with title options, hooks, and section breakdowns (Fast Model).",
    instruction=(
        "You are an expert content strategist and blog outliner.\n"
        "Create a clear, engaging blog post outline tailored to the user's topic, "
        "target audience, and tone.\n"
        "Include:\n"
        "- 3 catchy title options (SEO-friendly and reader-focused)\n"
        "- Target audience, tone, and primary SEO keyword\n"
        "- An engaging hook / introduction summary\n"
        "- Structured H2/H3 sections with bullet points of key arguments or examples\n"
        "- A conclusion with a clear call-to-action (CTA)"
    ),
    tools=[recall_writer_memory],
    output_key="blog_outline",
    before_model_callback=security_and_context_before_model_callback,
    after_model_callback=pii_redaction_after_model_callback,
    before_tool_callback=hitl_and_tracing_before_tool_callback,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.4,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    ),
)

writer_agent = Agent(
    name="blog_drafter",
    model=REASONING_MODEL,
    description="Writes deep, engaging, well-structured Markdown blog posts from an outline (Reasoning Pro Model).",
    instruction=(
        "You are a senior blog writer running on a high-reasoning model.\n"
        "Write a polished, engaging, and well-structured blog post in Markdown based on "
        "the approved outline, stored style preferences, and research context.\n"
        "Guidelines:\n"
        "- Recall any stored brand voice or style preferences with `recall_writer_memory` when relevant.\n"
        "- Start with a compelling H1 title and an attention-grabbing introduction.\n"
        "- Use clear H2 and H3 subheadings, short readable paragraphs, bullet lists, "
        "and concrete examples or code snippets where relevant.\n"
        "- Maintain a consistent voice and tone appropriate for the target audience.\n"
        "- End with a strong takeaway and call-to-action.\n"
        "- Run `analyze_blog_metrics` on your draft to verify word count, structure, and keyword usage."
    ),
    tools=[analyze_blog_metrics, recall_writer_memory, save_writer_memory],
    output_key="blog_draft",
    before_model_callback=security_and_context_before_model_callback,
    after_model_callback=pii_redaction_after_model_callback,
    before_tool_callback=hitl_and_tracing_before_tool_callback,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    ),
)

editor_agent = Agent(
    name="blog_editor",
    model=REASONING_MODEL,
    description="Reviews, polishes, proofreads, and optimizes blog drafts for SEO with HITL publishing (Reasoning Pro Model).",
    instruction=(
        "You are a meticulous managing editor and SEO specialist.\n"
        "Review the blog post draft for clarity, narrative flow, conciseness, grammar, "
        "formatting, and SEO effectiveness.\n"
        "1. Call `analyze_blog_metrics` to inspect the post's structure, reading time, and keyword metrics.\n"
        "2. Provide a brief editorial summary of improvements made.\n"
        "3. Produce the final, publication-ready Markdown blog post (including suggested meta title "
        "and meta description at the top).\n"
        "4. **Human-in-the-Loop (HITL) Requirement**: Before calling `save_blog_post`, ensure the user has "
        "explicitly approved saving/publishing. When the user confirms, call `record_human_approval(approved=True)` "
        "and then call `save_blog_post`."
    ),
    tools=[
        analyze_blog_metrics,
        record_human_approval,
        save_blog_post_tool,
        save_writer_memory,
        recall_writer_memory,
    ],
    output_key="final_blog_post",
    before_model_callback=security_and_context_before_model_callback,
    after_model_callback=pii_redaction_after_model_callback,
    before_tool_callback=hitl_and_tracing_before_tool_callback,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.3,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    ),
)

root_agent = Agent(
    name="blog_writer_coordinator",
    model=FAST_MODEL,
    description="End-to-end blog writing coordinator with strategic model routing, memory, guardrails, and HITL.",
    instruction=(
        "You are an expert Blog Writing Assistant built with Google ADK.\n"
        "Help the user go from an initial idea to a polished, publication-ready blog post.\n\n"
        "How to work with the user:\n"
        "1. **Memory & Preferences**: Use `recall_writer_memory` and `save_writer_memory` to remember the user's "
        "preferred tone, target audience, or style guide across turns.\n"
        "2. **Clarify & Research (`blog_researcher` - Fast Model)**: Understand the topic, target audience, "
        "desired tone, and length. Use the `blog_researcher` tool whenever up-to-date facts, trends, or SEO angles are helpful.\n"
        "3. **Outline (`blog_outliner` - Fast Model)**: Delegate to `blog_outliner` to build a structured outline.\n"
        "4. **Draft (`blog_drafter` - Pro Reasoning Model)**: Delegate to `blog_drafter` for long-form creative writing.\n"
        "5. **Edit & Export (`blog_editor` - Pro Reasoning Model + HITL)**: Delegate to `blog_editor` to polish the draft, "
        "verify SEO metrics with `analyze_blog_metrics`, and request Human-in-the-Loop confirmation via "
        "`record_human_approval` before exporting with `save_blog_post`."
    ),
    tools=[
        AgentTool(agent=research_agent),
        analyze_blog_metrics,
        record_human_approval,
        save_blog_post_tool,
        save_writer_memory,
        recall_writer_memory,
    ],
    sub_agents=[
        outliner_agent,
        writer_agent,
        editor_agent,
    ],
    before_model_callback=security_and_context_before_model_callback,
    after_model_callback=pii_redaction_after_model_callback,
    before_tool_callback=hitl_and_tracing_before_tool_callback,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.3,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    ),
)


# ============================================================================
# 8. PERSISTENT SESSION SERVICE & RUNNER FACTORY
# ============================================================================

DEFAULT_DB_URL = os.getenv("BLOG_SESSION_DB_URL", "sqlite:///./blog_writer_sessions.db")


def create_persistent_runner(
    app_name: str = "blog_writer",
    db_url: str = DEFAULT_DB_URL,
) -> Runner:
    """Creates an ADK Runner backed by a persistent SQLite DatabaseSessionService.

    Args:
        app_name: Application identifier for session partitioning.
        db_url: SQLAlchemy database URL (defaults to local SQLite file).

    Returns:
        Configured ADK `Runner` with persistent session state.
    """
    session_service = DatabaseSessionService(db_url=db_url)
    return Runner(
        agent=root_agent,
        app_name=app_name,
        session_service=session_service,
    )
