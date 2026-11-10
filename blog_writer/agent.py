"""Google ADK Blog Post Writer Agent.

Provides a multi-agent workflow for researching, outlining, drafting,
optimizing, and exporting blog posts using the Google Agent Development Kit (ADK).
"""

from pathlib import Path
import re
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool

MODEL_NAME = "gemini-3.8-flash"


def analyze_blog_metrics(content: str, target_keyword: str = "") -> dict[str, Any]:
    """Analyzes a blog post draft for word count, reading time, structure, and SEO keyword usage.

    Args:
        content: The full markdown text of the blog post draft.
        target_keyword: Optional primary SEO keyword or phrase to check against the draft.

    Returns:
        A dictionary containing word count, estimated reading time in minutes,
        heading counts, paragraph count, and keyword density metrics.
    """
    words = re.findall(r"\b\w+\b", content)
    word_count = len(words)
    reading_time_minutes = max(1, round(word_count / 200))

    h1_headings = re.findall(r"^#\s+.+", content, flags=re.MULTILINE)
    h2_headings = re.findall(r"^##\s+.+", content, flags=re.MULTILINE)
    h3_headings = re.findall(r"^###\s+.+", content, flags=re.MULTILINE)
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    keyword_stats: dict[str, Any] = {}
    if target_keyword.strip():
        kw_lower = target_keyword.strip().lower()
        occurrences = len(re.findall(re.escape(kw_lower), content.lower()))
        in_h1 = any(kw_lower in h.lower() for h in h1_headings)
        in_h2 = any(kw_lower in h.lower() for h in h2_headings)
        density_pct = round((occurrences * len(kw_lower.split()) / max(1, word_count)) * 100, 2)
        keyword_stats = {
            "target_keyword": target_keyword,
            "occurrences": occurrences,
            "density_percent": density_pct,
            "present_in_h1": in_h1,
            "present_in_h2": in_h2,
        }

    return {
        "status": "success",
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


def save_blog_post(filename: str, content: str) -> dict[str, str]:
    """Saves the finished blog post markdown to a local file in the `output/` directory.

    Args:
        filename: Desired filename (e.g. 'my-blog-post.md').
        content: The markdown content of the blog post.

    Returns:
        A dictionary with the save status and file path.
    """
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "-", filename.strip())
    if not safe_name.endswith(".md"):
        safe_name += ".md"

    output_dir = Path.cwd() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / safe_name
    output_path.write_text(content, encoding="utf-8")

    return {
        "status": "success",
        "file_path": str(output_path),
        "message": f"Blog post saved to {output_path}",
    }


research_agent = Agent(
    name="blog_researcher",
    model=MODEL_NAME,
    description="Researches blog topics, current trends, facts, and examples using Google Search.",
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
)

outliner_agent = Agent(
    name="blog_outliner",
    model=MODEL_NAME,
    description="Creates structured blog post outlines with title options, hooks, and section breakdowns.",
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
    output_key="blog_outline",
)

writer_agent = Agent(
    name="blog_drafter",
    model=MODEL_NAME,
    description="Writes engaging, well-structured markdown blog posts from an outline or prompt.",
    instruction=(
        "You are a senior blog writer.\n"
        "Write a polished, engaging, and well-structured blog post in Markdown based on "
        "the approved outline and research context.\n"
        "Guidelines:\n"
        "- Start with a compelling H1 title and an attention-grabbing introduction.\n"
        "- Use clear H2 and H3 subheadings, short readable paragraphs, bullet lists, "
        "and concrete examples or code snippets where relevant.\n"
        "- Maintain a consistent voice and tone appropriate for the target audience.\n"
        "- End with a strong takeaway and call-to-action.\n"
        "- Run `analyze_blog_metrics` on your draft to verify word count, structure, and keyword usage."
    ),
    tools=[analyze_blog_metrics],
    output_key="blog_draft",
)

editor_agent = Agent(
    name="blog_editor",
    model=MODEL_NAME,
    description="Reviews, polishes, proofreads, and optimizes blog drafts for clarity, flow, and SEO.",
    instruction=(
        "You are a meticulous managing editor and SEO specialist.\n"
        "Review the blog post draft for clarity, narrative flow, conciseness, grammar, "
        "formatting, and SEO effectiveness.\n"
        "1. Call `analyze_blog_metrics` to inspect the post's structure, reading time, and keyword metrics.\n"
        "2. Provide a brief editorial summary of improvements made.\n"
        "3. Produce the final, publication-ready Markdown blog post (including suggested meta title "
        "and meta description at the top).\n"
        "4. If the user asks to save or export the post, use `save_blog_post`."
    ),
    tools=[analyze_blog_metrics, save_blog_post],
    output_key="final_blog_post",
)

root_agent = Agent(
    name="blog_writer_coordinator",
    model=MODEL_NAME,
    description="End-to-end blog writing assistant that coordinates research, outlining, drafting, and editing.",
    instruction=(
        "You are an expert Blog Writing Assistant built with Google ADK.\n"
        "Help the user go from an initial idea to a polished, publication-ready blog post.\n\n"
        "How to work with the user:\n"
        "1. **Clarify & Research**: Understand the topic, target audience, desired tone, and length. "
        "Use the `blog_researcher` tool whenever up-to-date facts, trends, or SEO angles are helpful.\n"
        "2. **Outline (`blog_outliner`)**: Delegate to `blog_outliner` (or generate a structured outline) "
        "so the user can review the structure or ask you to proceed directly to a full post.\n"
        "3. **Draft (`blog_drafter`)**: Delegate to `blog_drafter` to write the complete Markdown blog post.\n"
        "4. **Edit & Export (`blog_editor`)**: Delegate to `blog_editor` to polish the draft, verify SEO metrics "
        "with `analyze_blog_metrics`, and optionally save the final Markdown file with `save_blog_post`.\n\n"
        "If the user asks for a complete blog post right away, guide it smoothly through research, "
        "outlining, drafting, and editing."
    ),
    tools=[
        AgentTool(agent=research_agent),
        analyze_blog_metrics,
        save_blog_post,
    ],
    sub_agents=[
        outliner_agent,
        writer_agent,
        editor_agent,
    ],
)
