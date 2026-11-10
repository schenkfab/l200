# Blog Writer — Google ADK Agent

A multi-agent blog writing assistant built with the **Google Agent Development Kit (ADK)**. It helps you brainstorm, research, outline, draft, polish, and save Markdown blog posts.

## Architecture

- **`blog_writer_coordinator` (`root_agent`)**: Orchestrates the blog creation workflow and collaborates with the user.
- **`blog_researcher`** (wrapped via `AgentTool`): Uses `google_search` to gather facts, trends, audience pain points, and SEO keywords.
- **`blog_outliner`** (sub-agent): Produces catchy title options, section structures, and call-to-action plans.
- **`blog_drafter`** (sub-agent): Writes the full Markdown blog post and checks word count/readability metrics.
- **`blog_editor`** (sub-agent): Polishes prose, evaluates SEO keyword density via `analyze_blog_metrics`, adds meta tags, and exports to `output/*.md` via `save_blog_post`.

## Setup

1. Create and activate a virtual environment, then install dependencies:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. Configure credentials by copying `.env.example` to `.env`:
   ```bash
   cp blog_writer/.env.example blog_writer/.env
   ```
   Edit `blog_writer/.env` to set either your `GOOGLE_API_KEY` (Google AI Studio) or `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` (Vertex AI).

## Running the Agent

From `/home/user/Development/l200`:

- **Interactive Web UI**:
  ```bash
  .venv/bin/adk web
  ```
- **Terminal CLI**:
  ```bash
  .venv/bin/adk run blog_writer
  ```
