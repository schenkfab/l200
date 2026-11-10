# Blog Writer — Enterprise Google ADK Multi-Agent System

A production-grade multi-agent blog writing system built with the **Google Agent Development Kit (ADK)**. It orchestrates research, outlining, long-form drafting, SEO analysis, and Human-in-the-Loop (HITL) publishing with full observability, security guardrails, persistent memory, and Infrastructure as Code (IaC).

---

## Key Capabilities & Architecture

### 1. Tool & Interface Design
- **Explicit JSON & Pydantic Schemas**: Strict validation models ([`BlogMetricsInput`](file:///home/user/Development/l200/blog_writer/agent.py#L145-L157), [`SaveBlogPostInput`](file:///home/user/Development/l200/blog_writer/agent.py#L160-L178), [`MemoryEntryInput`](file:///home/user/Development/l200/blog_writer/agent.py#L181-L200), [`BlogOutlineSchema`](file:///home/user/Development/l200/blog_writer/agent.py#L211-L221)) exported in [`TOOL_JSON_SCHEMAS`](file:///home/user/Development/l200/blog_writer/agent.py#L224-L228).
- **Guided Error Recovery**: Every tool wraps execution in `try/except` blocks handling `ValidationError`, `OSError`, and runtime exceptions, returning structured `error_type`, `expected_json_schema`, and actionable `recovery_instruction` prompts so the LLM can self-correct.

### 2. Context & Memory Management
- **History Compaction**: [`compact_conversation_history`](file:///home/user/Development/l200/blog_writer/agent.py#L260-L303) automatically summarizes older conversation turns into `state["compacted_history_summary"]` when turns exceed `MAX_HISTORY_TURNS`.
- **Persistent Session State**: [`create_persistent_runner`](file:///home/user/Development/l200/blog_writer/agent.py#L774-L793) configures ADK's `DatabaseSessionService` (`sqlite:///./blog_writer_sessions.db`) and tracks `user:` and session-scoped state across turns.
- **Asynchronous Memory Operations**: [`save_writer_memory`](file:///home/user/Development/l200/blog_writer/agent.py#L569-L632) and [`recall_writer_memory`](file:///home/user/Development/l200/blog_writer/agent.py#L635-L686) use `async def`, `asyncio.Lock`, and `asyncio.to_thread` to persist and query author tone, style guides, and past topics without blocking the event loop.

### 3. Orchestration, Strategic Model Routing, Guardrails & HITL
- **Strategic Model Routing**:
  - **Fast Tier (`gemini-2.5-flash`)**: Assigned to [`root_agent`](file:///home/user/Development/l200/blog_writer/agent.py#L728-L765) (`blog_writer_coordinator`), [`research_agent`](file:///home/user/Development/l200/blog_writer/agent.py#L649-L667) (`blog_researcher`), and [`outliner_agent`](file:///home/user/Development/l200/blog_writer/agent.py#L669-L689) (`blog_outliner`) for low-latency routing, search synthesis, and structuring.
  - **Reasoning Tier (`gemini-2.5-pro`)**: Assigned to [`writer_agent`](file:///home/user/Development/l200/blog_writer/agent.py#L691-L716) (`blog_drafter`) and [`editor_agent`](file:///home/user/Development/l200/blog_writer/agent.py#L718-L749) (`blog_editor`) for deep creative writing and nuanced editorial/SEO optimization.
- **Security Guardrails**: [`security_and_context_before_model_callback`](file:///home/user/Development/l200/blog_writer/agent.py#L306-L382) blocks prompt injection and secret-extraction attempts before model invocation, alongside strict Gemini `SafetySetting` thresholds.
- **Programmatic Human-in-the-Loop (HITL)**: [`hitl_and_tracing_before_tool_callback`](file:///home/user/Development/l200/blog_writer/agent.py#L398-L433) and [`record_human_approval`](file:///home/user/Development/l200/blog_writer/agent.py#L469-L501) enforce explicit human review before [`save_blog_post`](file:///home/user/Development/l200/blog_writer/agent.py#L504-L591) writes any file to disk.

### 4. Observability, Distributed Tracing & PII Redaction
- **Structured JSON Logging & PII Redaction**: [`PiiRedactingJsonFormatter`](file:///home/user/Development/l200/blog_writer/agent.py#L70-L85) and [`redact_pii`](file:///home/user/Development/l200/blog_writer/agent.py#L60-L67) scrub emails, phone numbers, SSNs, credit cards, and API keys across logs, incoming prompts, tool outputs, and model responses.
- **Intent Capturing**: [`classify_user_intent`](file:///home/user/Development/l200/blog_writer/agent.py#L250-L257) classifies every turn (`research`, `outline`, `draft`, `edit_seo`, `publish_save`) and records an audit trail in `state["intent_history"]`.
- **OpenTelemetry Tracing**: Spans wrap secret loading, model callbacks, history compaction, and all tool executions.

### 5. Infrastructure as Code, Secret Manager & CI/CD
- **Google Cloud Secret Manager**: [`load_secret_from_gcp`](file:///home/user/Development/l200/blog_writer/agent.py#L97-L129) fetches `GOOGLE_API_KEY` from GCP Secret Manager with local `.env` fallback.
- **Terraform & Docker**: [`terraform/main.tf`](file:///home/user/Development/l200/terraform/main.tf) provisions Secret Manager, least-privilege IAM, and Cloud Run v2; [`Dockerfile`](file:///home/user/Development/l200/Dockerfile) packages the ADK API server.
- **Automated Evaluation & CI/CD**: [`.github/workflows/ci-cd.yml`](file:///home/user/Development/l200/.github/workflows/ci-cd.yml) runs unit tests ([`tests/test_agent_eval.py`](file:///home/user/Development/l200/tests/test_agent_eval.py)), ADK evaluation ([`eval/blog_writer.evalset.json`](file:///home/user/Development/l200/eval/blog_writer.evalset.json)), and Terraform validation.

---

## Setup & Usage

1. **Install dependencies in a virtual environment**:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. **Configure credentials**:
   ```bash
   cp blog_writer/.env.example blog_writer/.env
   ```

3. **Run the Agent**:
   ```bash
   # Interactive Web UI
   .venv/bin/adk web

   # Terminal CLI
   .venv/bin/adk run blog_writer
   ```

4. **Run Automated Tests & ADK Evaluation**:
   ```bash
   # Unit, schema, guardrail, PII, and HITL tests
   .venv/bin/python -m unittest discover -s tests -p "test_*.py" -v

   # ADK evaluation suite
   .venv/bin/adk eval blog_writer eval/blog_writer.evalset.json --config_file_path=eval/test_config.json
   ```
