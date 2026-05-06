import os
import json

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.utilities import SerpAPIWrapper


# =========================
# 🔑 CONFIG (SET YOUR KEYS)
# =========================
HUGGINGFACE_API_KEY = "your_hf_key"
SERPAPI_API_KEY = "your_serpapi_key"

os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY


# =========================
# 🤖 LLM SETUP
# =========================
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        temperature=0.1,
        max_new_tokens=512
    )
)


# =========================
# 🔍 TOOL (SEARCH)
# =========================
search = SerpAPIWrapper()


# =========================
# 🧠 STEP FUNCTIONS
# =========================

def generate_outline(topic):
    prompt = f"""
Create a structured outline.

Topic: {topic}

Return STRICT JSON:
{{
  "title": "...",
  "sections": ["...", "...", "..."]
}}
"""
    return llm.invoke(prompt).content


def research_section(section):
    try:
        return search.run(f"{section} detailed explanation 2025")
    except Exception as e:
        return f"Research error: {str(e)}"


def generate_section(topic, section, research):
    prompt = f"""
Write a detailed section.

Topic: {topic}
Section: {section}
Research: {research}

Write clear, structured, non-repetitive content.
"""
    return llm.invoke(prompt).content


def validate_output(step, content):
    prompt = f"""
Check quality of this step.

Step: {step}
Content: {content}

Return:
{{
  "valid": true/false,
  "reason": "..."
}}
"""
    return llm.invoke(prompt).content


def correct_output(step, content):
    prompt = f"""
Fix this content.

Step: {step}
Content: {content}

Return improved version.
"""
    return llm.invoke(prompt).content


def check_coherence(content):
    prompt = f"""
Check if content is:
- coherent
- no repetition
- logical flow

Content:
{content}

Return:
{{
  "coherent": true/false,
  "issues": "..."
}}
"""
    return llm.invoke(prompt).content


def refine_content(content):
    prompt = f"""
Improve this content:
- better flow
- remove repetition
- improve clarity

Content:
{content}
"""
    return llm.invoke(prompt).content


# =========================
# 🔁 MAIN AGENT
# =========================

def content_agent(topic, max_sections=6):
    print("\n=== STEP 1: OUTLINE ===")

    outline_raw = generate_outline(topic)

    try:
        outline = json.loads(outline_raw)
    except:
        print("⚠️ Fixing outline JSON...")
        outline_fixed = correct_output("outline", outline_raw)
        outline = json.loads(outline_fixed)

    title = outline["title"]
    sections = outline["sections"][:max_sections]

    full_content = f"# {title}\n\n"
    section_contents = []

    # =========================
    # STEP 2 + 3 (RESEARCH + DRAFT)
    # =========================
    for sec in sections:
        print(f"\n=== SECTION: {sec} ===")

        research = research_section(sec)

        draft = generate_section(topic, sec, research)

        validation = validate_output("draft", draft)

        if "false" in validation.lower():
            print("⚠️ Fixing draft...")
            draft = correct_output("draft", draft)

        section_contents.append(f"## {sec}\n{draft}")

    # Combine all
    full_content += "\n\n".join(section_contents)

    # =========================
    # STEP 4 (COHERENCE CHECK)
    # =========================
    print("\n=== COHERENCE CHECK ===")

    coherence = check_coherence(full_content)

    if "false" in coherence.lower():
        print("⚠️ Fixing coherence...")
        full_content = refine_content(full_content)

    # =========================
    # STEP 5 (FINAL REFINEMENT)
    # =========================
    print("\n=== FINAL REFINEMENT ===")

    final_output = refine_content(full_content)

    return final_output


# =========================
# ▶️ RUN
# =========================

if __name__ == "__main__":
    topic = "Future of Electric Vehicles in 2025"

    result = content_agent(topic)

    print("\n\n===== FINAL OUTPUT =====\n")
    print(result)
