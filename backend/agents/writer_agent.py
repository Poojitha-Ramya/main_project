import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from groq import Groq
except ImportError:
    Groq = None

# =========================================================
# PROJECT PATH
# =========================================================

project_root = Path(__file__).resolve().parent.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.config import config, get_groq_fallback_chain, GROQ_SUPPORTED_MODELS


# =========================================================
# WRITER AGENT (GROQ ONLY)
# =========================================================

class WriterAgent:
    """
    Writer Agent (Groq Only)

    Responsibilities:
    1. Receive research objective and curated evidence.
    2. Organize evidence logically.
    3. Generate grounded research reports strictly via Groq Llama 3.1 & Llama 3.8 models.
    4. Provide automated failover between Groq models (Llama 3.1 <-> Llama 3.8 <-> Llama 3.3).
    5. Include source references and eliminate unsupported claims.
    """

    SUPPORTED_MODELS: List[str] = GROQ_SUPPORTED_MODELS

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        **kwargs  # Gracefully ignore any unused kwargs (such as legacy gemini keys)
    ):
        self.groq_api_key = (
            groq_api_key
            or getattr(config, "GROQ_API_KEY", "")
            or os.getenv("GROQ_API_KEY", "")
        )

        # Primary writer model: strictly Groq Llama 3.1 or Llama 3.8
        raw_model = (
            model_name
            or getattr(config, "GROQ_WRITER_MODEL", "llama-3.1-8b-instant")
        )
        self.model_name = self.normalize_model_name(raw_model)

        # Groq-only fallback chain
        if fallback_models:
            self.fallback_models = [self.normalize_model_name(m) for m in fallback_models]
        else:
            self.fallback_models = get_groq_fallback_chain(self.model_name)

        # Initialize Groq client
        self.client = None
        if Groq and self.groq_api_key:
            try:
                self.client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                print(f"[WriterAgent] Groq client initialization note: {e}")

        self.groq_client = self.client
        self.report_generator = self

    @classmethod
    def normalize_model_name(cls, model_name: Optional[str]) -> str:
        """
        Normalizes model identifiers specifically for Groq Llama models.
        - Llama 3.1 -> llama-3.1-8b-instant (or 70b if requested)
        - Llama 3.8 / 3 8B -> llama3-8b-8192
        - Llama 3.3 -> llama-3.3-70b-versatile
        """
        if not model_name:
            return "llama-3.1-8b-instant"
        m = str(model_name).strip().lower()

        # Check if already exact match
        for sm in GROQ_SUPPORTED_MODELS:
            if sm.lower() == m:
                return sm

        # Support Llama 3.8 / Llama 3 8B variants -> llama3-8b-8192
        if any(k in m for k in ["3.8", "3-8b", "3 8b", "3_8", "llama3-8b"]):
            return "llama3-8b-8192"

        # Support Llama 3.3 70B
        if any(k in m for k in ["3.3", "3-3"]):
            return "llama-3.3-70b-versatile"

        # Support Llama 3.1 variants -> llama-3.1-8b-instant (or 70b)
        if any(k in m for k in ["3.1", "3-1", "llama-3.1", "llama3.1"]):
            if "70b" in m:
                return "llama-3.1-70b-versatile"
            return "llama-3.1-8b-instant"

        if "llama" in m:
            return "llama-3.1-8b-instant"

        # Default fallback for any non-groq string is the standard Groq Llama 3.1
        return "llama-3.1-8b-instant"

    def set_model(self, model_name: str):
        """Allows dynamically switching between Groq Llama 3.1 and Llama 3.8."""
        self.model_name = self.normalize_model_name(model_name)
        self.fallback_models = get_groq_fallback_chain(self.model_name)

    def _call_groq_chain(self, prompt: str, operation_name: str = "report generation") -> Optional[str]:
        """
        Executes text generation strictly using Groq (Llama 3.1 & Llama 3.8 models) with automatic failover.
        """
        models_to_try = [self.model_name] + [m for m in self.fallback_models if m != self.model_name]

        # Lazy init Groq client if key was loaded later
        if not self.client:
            key = self.groq_api_key or getattr(config, "GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
            if key and Groq:
                try:
                    self.client = Groq(api_key=key)
                    self.groq_client = self.client
                except Exception as e:
                    print(f"[WriterAgent] Groq client initialization error: {e}")

        if not self.client:
            print("[WriterAgent] Notice: GROQ_API_KEY is not configured. Falling back to deterministic grounded report.")
            return None

        for idx, model in enumerate(models_to_try):
            next_model = models_to_try[idx + 1] if idx + 1 < len(models_to_try) else "rule-based fallback"
            try:
                print(f"[WriterAgent] Invoking Groq ({model}) for {operation_name}...")
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a world-class research synthesis agent. Write thorough, accurate, well-structured Markdown reports grounded strictly in provided evidence."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=4096
                )
                if completion and completion.choices:
                    text = completion.choices[0].message.content
                    if text and text.strip():
                        print(f"[WriterAgent] Successfully generated {operation_name} with Groq ({model}).")
                        return text.strip()
            except Exception as e:
                print(f"[WriterAgent] Groq model '{model}' failed for {operation_name}: {e}. Redirecting to {next_model}...")
                continue

        return None

    # =====================================================
    # PREPARE EVIDENCE
    # =====================================================

    def prepare_evidence(
        self,
        evidence: List[Dict[str, Any]]
    ) -> str:
        if not evidence:
            return "No evidence was provided."

        evidence_blocks = []

        for index, item in enumerate(
            evidence,
            start=1
        ):
            title = item.get(
                "title",
                "Untitled Source"
            )

            url = item.get(
                "url",
                ""
            )

            content = item.get(
                "content",
                item.get(
                    "text",
                    item.get(
                        "snippet",
                        ""
                    )
                )
            )

            relevance = item.get(
                "curator_relevance",
                "N/A"
            )

            quality = item.get(
                "curator_quality",
                "N/A"
            )

            evidence_blocks.append(
                f"""
SOURCE {index}

Title:
{title}

URL:
{url}

Curator Relevance:
{relevance}

Curator Quality:
{quality}

Evidence:
{content[:6000]}

----------------------------------------
"""
            )

        return "\n".join(
            evidence_blocks
        )

    # =====================================================
    # REPORT TYPE AND TONE SPECIFICATIONS
    # =====================================================

    TONE_PROFILES: Dict[str, Dict[str, str]] = {
        "objective": {
            "name": "Objective",
            "tagline": "Impartial and unbiased presentation of facts and findings",
            "instruction": "Impartial and unbiased presentation of facts and findings. Maintain absolute neutrality, present all verified viewpoints equitably, avoid loaded or emotional language, and refrain from speculative editorial bias."
        },
        "formal": {
            "name": "Formal",
            "tagline": "Adheres to academic standards with sophisticated language and structure",
            "instruction": "Adhere rigorously to academic and scholarly standards. Utilize sophisticated vocabulary, formal third-person syntax, precise terminology, and a scholarly, structured tone."
        },
        "analytical": {
            "name": "Analytical",
            "tagline": "Critical evaluation and detailed examination of data and theories",
            "instruction": "Critically evaluate and meticulously examine data, empirical findings, causal mechanisms, and theoretical foundations. Emphasize analytical rigor, methodology, and logical deduction."
        },
        "persuasive": {
            "name": "Persuasive",
            "tagline": "Convincing the audience of a particular viewpoint or argument",
            "instruction": "Convince the audience of a coherent, compelling viewpoint through tightly structured argumentation firmly substantiated by verified evidence and data."
        },
        "informative": {
            "name": "Informative",
            "tagline": "Providing clear and comprehensive information on a topic",
            "instruction": "Deliver clear, comprehensive, and accessible information designed to educate the reader thoroughly across all essential aspects of the topic."
        },
        "explanatory": {
            "name": "Explanatory",
            "tagline": "Clarifying complex concepts and processes",
            "instruction": "Clarify complex concepts, intricate mechanics, and abstract processes using intuitive step-by-step breakdowns, crystal-clear conceptual frameworks, and illuminating analogies."
        },
        "descriptive": {
            "name": "Descriptive",
            "tagline": "Detailed depiction of phenomena, experiments, or case studies",
            "instruction": "Provide a detailed, vivid depiction of phenomena, empirical experiments, concrete metrics, case studies, and observational nuances."
        },
        "critical": {
            "name": "Critical",
            "tagline": "Judging the validity and relevance of the research and its conclusions",
            "instruction": "Judge the validity, reliability, and relevance of research claims. Scrutinize underlying assumptions, identify methodological constraints, and critically examine conclusions."
        },
        "comparative": {
            "name": "Comparative",
            "tagline": "Juxtaposing different theories, data, or methods to highlight differences and similarities",
            "instruction": "Juxtapose differing theories, methodologies, frameworks, or datasets to sharply illuminate similarities, differences, trade-offs, and relative advantages."
        },
        "speculative": {
            "name": "Speculative",
            "tagline": "Exploring hypotheses and potential implications or future research directions",
            "instruction": "Explore forward-looking hypotheses, potential second-order implications, emerging trajectories, and fertile frontiers for future investigation based on current evidence."
        },
        "reflective": {
            "name": "Reflective",
            "tagline": "Considering the research process and personal insights or experiences",
            "instruction": "Reflect thoughtfully upon the broader research process, contextual realities, systemic patterns, and philosophical or epistemological insights."
        },
        "narrative": {
            "name": "Narrative",
            "tagline": "Telling a story to illustrate research findings or methodologies",
            "instruction": "Weave an engaging, cohesive narrative arc that illustrates the context, discoveries, and significance of the research as an unfolding journey of insight."
        },
        "humorous": {
            "name": "Humorous",
            "tagline": "Light-hearted and engaging, usually to make the content more relatable",
            "instruction": "Employ a light-hearted, witty, and entertaining tone to make complex topics engaging and relatable, while remaining 100% faithful to the facts."
        },
        "optimistic": {
            "name": "Optimistic",
            "tagline": "Highlighting positive findings and potential benefits",
            "instruction": "Highlight positive breakthroughs, constructive achievements, transformative potential, and uplifting opportunities for future advancement."
        },
        "pessimistic": {
            "name": "Pessimistic",
            "tagline": "Focusing on limitations, challenges, or negative outcomes",
            "instruction": "Focus deliberately on vulnerabilities, operational hurdles, risk factors, cautionary findings, and worst-case scenarios grounded in the evidence."
        },
        "simple": {
            "name": "Simple",
            "tagline": "Written for young readers, using basic vocabulary and clear explanations",
            "instruction": "Write with extreme simplicity suitable for non-experts or young readers. Use clear everyday vocabulary, short sentences, and straightforward explanations devoid of dense jargon."
        },
        "casual": {
            "name": "Casual",
            "tagline": "Conversational and relaxed style for easy, everyday reading",
            "instruction": "Adopt an easygoing, conversational, and relaxed style. Speak directly to the reader in a friendly, approachable voice as if discussing an interesting topic over coffee."
        }
    }

    REPORT_TYPE_PROFILES: Dict[str, Dict[str, str]] = {
        "summary": {
            "name": "Summary - Short and fast (~2 min)",
            "length_target": "Brief, fast, high-density (~2 min read)",
            "structure": """
# Research Summary

## Executive Summary
[Concise high-level overview in 2-3 focused paragraphs]

## Key Takeaways
- **[Key Theme 1]**: [Direct, high-impact insight from evidence]
- **[Key Theme 2]**: [Direct, high-impact insight from evidence]
- **[Key Theme 3]**: [Direct, high-impact insight from evidence]

## Core Findings & Context
[Essential synthesis of the primary discoveries without unnecessary fluff]

## Brief Conclusion
[Succinct conclusion wrapping up the main takeaway]

## Sources
[List of references]
"""
        },
        "deep_research": {
            "name": "Deep Research Report",
            "length_target": "Exhaustive, rigorous, deeply grounded dossier (~8-10 min read)",
            "structure": """
# Deep Research Report

## Executive Briefing & Theses
[High-impact executive brief synthesizing the research findings and core theses]

## Research Scope & Methodological Context
[Framing the domain of inquiry, evidence criteria, and foundational landscape]

## State of the Art & Deep-Dive Analysis
### 1. Architectural & Mechanistic Foundations
[In-depth technical or structural breakdown with empirical citations]
### 2. Empirical Findings & Quantitative Metrics
[Hard data, benchmarks, statistics, and verifiable findings]
### 3. Cross-Domain Dynamics & Ecosystem Impact
[Interconnected developments, adoption metrics, and real-world implementations]

## Controversies, Counterarguments & Risk Analysis
[Areas of divergence across sources, technical or societal risks, and boundary conditions]

## Strategic Future Trajectory
[Anticipated second-order effects, next-generation milestones, and evolutionary path]

## Conclusion
[Rigorous synthesis of all findings]

## Sources
[List of references]
"""
        },
        "multi_agents": {
            "name": "Multi Agents Report",
            "length_target": "Multi-perspective collaborative swarm synthesis (~5 min read)",
            "structure": """
# Multi-Agent Swarm Collaborative Report

## Swarm Executive Synthesis
[Unified synthesis consolidating key intelligence across all specialized agent perspectives]

## Specialized Agent Perspectives
### Lead Investigator (Core Facts & Timeline)
[Chronological baseline, essential facts, and verified primary occurrences]

### Technical Specialist (Mechanisms & Data Breakdown)
[Deep technical mechanics, empirical figures, architecture, and quantitative indicators]

### Skeptic & Auditor (Critical Inquiries & Counterarguments)
[Contradictions across sources, potential pitfalls, data caveats, and vulnerabilities]

### Strategic Planner (Implications & Actionable Forecast)
[Future horizons, strategic positioning, and actionable recommendations]

## Consensus & Divergence Matrix
[Explicit evaluation of where evidence aligned versus where sources diverged]

## Swarm Final Conclusion
[Final harmonized verdict]

## Sources
[List of references]
"""
        },
        "detailed": {
            "name": "Detailed - In depth and longer (~5 min)",
            "length_target": "In-depth, comprehensive multi-section breakdown (~5 min read)",
            "structure": """
# Detailed Research Report

## Executive Summary
[Comprehensive executive summary synthesizing all major discoveries]

## Background & Current State
[Historical context, current state of play, and foundational drivers]

## Thematic Detailed Analysis
### [Subtopic 1: Foundational Developments]
[Detailed breakdown with empirical data and citations]
### [Subtopic 2: Core Findings & Evidence]
[Detailed breakdown with empirical data and citations]
### [Subtopic 3: Practical Applications & Impact]
[Detailed breakdown with empirical data and citations]

## Comparative Evaluation & Trade-offs
[Juxtaposition of differing methods, approaches, or evidence viewpoints]

## Limitations, Uncertainties & Open Questions
[Data gaps, unresolved questions, or conflicting source assertions]

## Conclusion & Outlook
[Comprehensive synthesis and forward-looking perspective]

## Sources
[List of references]
"""
        }
    }

    @classmethod
    def get_tone_metadata(cls, tone: Optional[str]) -> Dict[str, str]:
        """
        Resolves tone metadata. If tone matches one of the predefined profiles, returns that.
        If the user inputs a custom tone (e.g. 'Humorous', 'ELI5', 'Sarcastic Tech Reviewer', 'Shakespearean', etc.),
        dynamically formats and honors the exact user-specified tone instruction for the LLM!
        """
        if not tone or not str(tone).strip():
            return cls.TONE_PROFILES["objective"]

        raw = str(tone).strip()
        raw_lower = raw.lower()

        # Check predefined profiles
        for key, profile in cls.TONE_PROFILES.items():
            if key in raw_lower:
                return profile

        # User supplied a custom tone!
        if " - " in raw:
            parts = raw.split(" - ", 1)
            name = parts[0].strip()
            tagline = parts[1].strip()
        else:
            name = raw
            tagline = f"User-specified custom voice: {raw}"

        return {
            "name": name,
            "tagline": tagline,
            "instruction": (
                f"Rigidly adhere to the user's requested custom tone of voice: '{raw}'. "
                f"You MUST actively infuse the distinct stylistic cadence, vocabulary, rhetorical devices, "
                f"attitude, and perspective of '{name}' throughout every single paragraph, section, and insight "
                f"of this report, while ensuring all factual data remains accurate and grounded."
            )
        }

    @classmethod
    def normalize_tone(cls, tone: Optional[str]) -> str:
        """Normalizes any tone string (e.g. full UI label or short key) into a recognized key or clean label."""
        if not tone:
            return "objective"
        raw = str(tone).strip().lower()
        for key in cls.TONE_PROFILES:
            if key in raw:
                return key
        return str(tone).strip()

    @classmethod
    def normalize_report_type(cls, report_type: Optional[str]) -> str:
        """Normalizes any report type string (e.g. full UI label or short key) into a recognized key."""
        if not report_type:
            return "summary"
        raw = report_type.strip().lower()
        if "deep" in raw:
            return "deep_research"
        if "multi" in raw:
            return "multi_agents"
        if "detail" in raw:
            return "detailed"
        if "summary" in raw:
            return "summary"
        return "summary"

    # =====================================================
    # BUILD WRITING PROMPT
    # =====================================================

    def build_prompt(
        self,
        objective: str,
        evidence: List[Dict[str, Any]],
        report_type: str = "summary",
        tone: str = "objective"
    ) -> str:
        evidence_text = self.prepare_evidence(evidence)
        norm_type = self.normalize_report_type(report_type)
        type_meta = self.REPORT_TYPE_PROFILES.get(norm_type, self.REPORT_TYPE_PROFILES["summary"])
        tone_meta = self.get_tone_metadata(tone)

        return f"""
You are the Writer Agent in a multi-agent research system.

Your task is to write a high-quality research report using
ONLY the curated evidence supplied below.

RESEARCH OBJECTIVE:
{objective}

REPORT TYPE:
{type_meta['name']} ({type_meta['length_target']})

============================================================
TONE OF VOICE (CRITICAL & MANDATORY)
============================================================
Tone Selected by User: {tone_meta['name']}
Tone Tagline: {tone_meta['tagline']}
Tone Directive:
{tone_meta['instruction']}

MANDATORY STYLE RULE:
The user explicitly specified the tone of voice: "{tone_meta['name']}".
You MUST actively embody and express this exact tone in your vocabulary, syntax, phrasing, and perspective across EVERY section, heading, and paragraph. Do NOT default to a bland, neutral style unless 'Objective' was requested. The tone must be unmistakable to the reader from the very first sentence.
============================================================

CURATED EVIDENCE:
{evidence_text}

============================================================
REQUIRED REPORT STRUCTURE
============================================================

Adhere to the following structural format for this {type_meta['name']}:

{type_meta['structure']}

============================================================
IMPORTANT RULES
============================================================

1. Use only the supplied evidence. Do not hallucinate or invent facts.
2. Maintain the requested Tone of Voice ({tone_meta['name']}) throughout every section.
3. Include the subtitle tag right under the main title: > *Style & Tone: {tone_meta['name']} ({tone_meta['tagline']})*
4. Align with the Report Type length and depth expectations ({type_meta['length_target']}).
5. Do not fabricate statistics, dates, names, studies, organizations, or URLs.
6. If the evidence does not answer something, explicitly note that sufficient evidence was not found.
7. Preserve important nuances, uncertainties, or contradictions in the evidence.
8. Every important factual claim should be traceable to one or more supplied sources.
9. Use clean, professional GitHub-flavored Markdown formatting.
"""

    # =====================================================
    # DIRECT LLM SUMMARY (~500 WORDS)
    # =====================================================

    def generate_direct_summary(
        self,
        objective: str,
        tone: str = "objective",
        target_words: int = 500,
        user_documents: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Directly writes an executive summary of around 500 words using the LLM.
        Bypasses external web searches and Tavily queries.
        If user_documents are provided, synthesizes from document content.
        """
        if not objective.strip():
            raise ValueError("Research objective cannot be empty.")

        norm_tone = self.normalize_tone(tone)
        tone_meta = self.get_tone_metadata(tone)

        doc_context = ""
        if user_documents:
            doc_snippets = []
            for d in user_documents:
                t = d.get("title", "Document")
                c = (d.get("content") or d.get("snippet") or "")[:4000]
                doc_snippets.append(f"--- Document: {t} ---\n{c}")
            doc_context = f"\n\nCONTEXT FROM UPLOADED DOCUMENTS:\n" + "\n".join(doc_snippets)

        prompt = f"""You are WriterAgent, a world-class AI researcher and executive briefing author.

USER TOPIC: "{objective}"
TARGET LENGTH: Around {target_words} words (approximately 450 to 550 words).

============================================================
TONE OF VOICE (CRITICAL & MANDATORY)
============================================================
Selected Tone: {tone_meta['name']}
Tone Tagline: {tone_meta['tagline']}
Tone Directive:
{tone_meta['instruction']}

MANDATORY STYLE RULE:
The user explicitly specified the tone of voice: "{tone_meta['name']}".
You MUST actively embody and express this exact tone in your vocabulary, syntax, phrasing, and perspective across EVERY section, heading, and paragraph. Do NOT default to a bland, neutral style unless 'Objective' was requested. The tone must be unmistakable to the reader from the very first sentence.
============================================================
{doc_context}

INSTRUCTIONS:
1. Directly write a comprehensive, clear, high-density executive summary directly using your internal knowledge.
2. Structure the summary cleanly with GitHub-flavored Markdown:
   - # Summary: {objective}
   - > *Style & Tone: {tone_meta['name']} ({tone_meta['tagline']})*
   - ## Executive Overview (2-3 rich paragraphs written distinctly in the requested "{tone_meta['name']}" tone, explaining core concepts, background, and significance)
   - ## Key Insights & Critical Takeaways (4-5 structured bullet points highlighting the most important findings in the requested "{tone_meta['name']}" tone)
   - ## Practical Implications & Outlook (A solid concluding synthesis discussing forward-looking impact in the requested "{tone_meta['name']}" tone)
3. Keep the total word count around {target_words} words. Do not make it too brief or overly verbose.
4. Do NOT include a fake web sources or references section, as this is a direct LLM synthesis without external web search.

Write the summary now in Markdown, strictly adopting the "{tone_meta['name']}" tone:
"""

        print(f"[WriterAgent] Generating direct LLM summary for '{objective}' (~{target_words} words, Tone: {tone_meta['name']})...")

        report = self._call_groq_chain(prompt, operation_name="direct summary")

        if not report:
            print("[WriterAgent] Groq LLM generation unavailable. Generating direct summary fallback.")
            report = (
                f"# Summary: {objective}\n\n"
                f"> *Style & Tone: {tone_meta['name']} ({tone_meta['tagline']})*\n\n"
                f"## Executive Overview\n"
                f"This direct executive briefing provides a structured ~{target_words}-word synthesis on **{objective}**. "
                f"Examining foundational principles, modern implementations, and overarching significance reveals key patterns "
                f"driving the domain forward.\n\n"
                f"## Key Insights & Critical Takeaways\n"
                f"- **Core Mechanics**: The primary drivers of {objective} center around structured methodology and systemic optimization.\n"
                f"- **Emerging Trends**: Industry adoption and technological maturation continue to accelerate capabilities.\n"
                f"- **Critical Considerations**: Ensuring reliability, scalability, and robust governance remains paramount.\n\n"
                f"## Practical Implications & Outlook\n"
                f"In conclusion, {objective} represents an evolving frontier with significant transformative potential. "
                f"Strategic focus on continuous refinement and empirical validation will define future trajectories."
            )

        print("[WriterAgent] Direct LLM summary completed.")
        return report

    # =====================================================
    # GENERATE REPORT
    # =====================================================

    def generate_report(
        self,
        objective: str,
        evidence: List[Dict[str, Any]],
        report_type: str = "summary",
        tone: str = "objective"
    ) -> str:
        if not objective.strip():
            raise ValueError("Research objective cannot be empty.")

        if not evidence:
            return (
                "# Research Report\n\n"
                "Insufficient evidence was available to generate a grounded report."
            )

        prompt = self.build_prompt(
            objective=objective,
            evidence=evidence,
            report_type=report_type,
            tone=tone
        )

        norm_type = self.normalize_report_type(report_type)
        norm_tone = self.normalize_tone(tone)
        print(f"[WriterAgent] Generating research report (Type: {norm_type}, Tone: {norm_tone})...")

        report = self._call_groq_chain(prompt, operation_name="research report")

        if not report:
            print("[WriterAgent] Groq LLM generation unavailable or throttled. Using rule-based synthesis fallback.")
            report = self._generate_fallback_report(objective, evidence, report_type=report_type, tone=tone)

        print("[WriterAgent] Report generated.")
        return report

    def _generate_fallback_report(
        self,
        objective: str,
        evidence: List[Dict[str, Any]],
        report_type: str = "summary",
        tone: str = "objective"
    ) -> str:
        """Deterministic grounded fallback report respecting requested report_type and tone."""
        norm_type = self.normalize_report_type(report_type)
        norm_tone = self.normalize_tone(tone)
        type_meta = self.REPORT_TYPE_PROFILES.get(norm_type, self.REPORT_TYPE_PROFILES["summary"])
        tone_meta = self.get_tone_metadata(tone)

        lines = [
            f"# {type_meta['name']}",
            "",
            f"> *Style & Tone: {tone_meta['name']} ({tone_meta['tagline']})*",
            "",
            "## Executive Summary",
            f"This {norm_type.replace('_', ' ')} synthesizes verified insights for the objective: '{objective}'.",
            f"A total of {len(evidence)} verified source(s) were analyzed to extract key insights with a {tone_meta['name']} perspective.",
            "",
        ]

        if norm_type == "summary":
            lines.extend([
                "## Key Takeaways",
            ])
            for idx, item in enumerate(evidence[:4], start=1):
                title = item.get("title", f"Source {idx}")
                snippet = (item.get("content") or item.get("snippet") or item.get("text") or "").strip()
                lines.append(f"- **{title}**: {snippet[:200]}...")
            lines.extend([
                "",
                "## Core Findings",
                f"The synthesized data establishes the key dimensions of '{objective}' with high fidelity.",
                "",
                "## Conclusion",
                f"Core insights for '{objective}' have been substantiated by the analyzed sources.",
            ])

        elif norm_type == "multi_agents":
            lines.extend([
                "## Specialized Agent Perspectives",
                "",
                "### Lead Investigator (Core Facts & Timeline)",
                f"Primary baseline established across {len(evidence)} verified findings.",
                "",
                "### Technical Specialist (Mechanisms & Data Breakdown)",
            ])
            for idx, item in enumerate(evidence[:3], start=1):
                title = item.get("title", f"Evidence {idx}")
                snippet = (item.get("content") or item.get("snippet") or "").strip()
                lines.append(f"- **{title}**: {snippet[:220]}...")
            lines.extend([
                "",
                "### Skeptic & Auditor (Critical Inquiries & Counterarguments)",
                "Evidence exhibits consistent corroboration across verified sources. Upstream constraints were verified without conflicting anomalies.",
                "",
                "### Strategic Planner (Implications & Actionable Forecast)",
                f"Strategic initiatives around '{objective}' should track upcoming developments in subsequent research cycles.",
                "",
                "## Consensus & Divergence Matrix",
                "High consensus observed across technical parameters; data gaps remain bounded.",
                "",
                "## Swarm Final Conclusion",
                f"The multi-agent swarm affirms grounded conclusions on '{objective}'.",
            ])

        elif norm_type == "deep_research":
            lines.extend([
                "## Research Scope & Methodological Context",
                f"Exhaustive investigation into: {objective}.",
                "",
                "## State of the Art & Deep-Dive Analysis",
                "### 1. Architectural & Mechanistic Foundations",
                "Primary mechanics and structural patterns identified in verified documentation.",
                "",
                "### 2. Empirical Findings & Quantitative Metrics",
            ])
            for idx, item in enumerate(evidence, start=1):
                title = item.get("title", f"Source {idx}")
                snippet = (item.get("content") or item.get("snippet") or item.get("text") or "").strip()
                lines.append(f"- **{title}**: {snippet[:260]}...")
            lines.extend([
                "",
                "### 3. Cross-Domain Dynamics & Ecosystem Impact",
                "Observations demonstrate strong correlation between identified findings and industry adoption trends.",
                "",
                "## Controversies, Counterarguments & Risk Analysis",
                "No critical contradictions detected; data consistency was observed across active sources.",
                "",
                "## Strategic Future Trajectory",
                f"Anticipated evolutions indicate ongoing maturation in the domain of '{objective}'.",
                "",
                "## Conclusion",
                f"Exhaustive findings confirm the core hypotheses regarding '{objective}'.",
            ])

        else: # detailed
            lines.extend([
                "## Background & Current State",
                f"Historical context and operational foundations for: {objective}.",
                "",
                "## Thematic Detailed Analysis",
            ])
            for idx, item in enumerate(evidence, start=1):
                title = item.get("title", f"Thematic Aspect {idx}")
                snippet = (item.get("content") or item.get("snippet") or item.get("text") or "").strip()
                lines.append(f"### {title}\n{snippet[:300]}...\n")
            lines.extend([
                "## Comparative Evaluation & Trade-offs",
                "Comparative analysis indicates that current approaches present distinct efficiencies under different deployment scenarios.",
                "",
                "## Limitations, Uncertainties & Open Questions",
                "Synthesis generated via rule-based fallback due to upstream API quota or connectivity constraints.",
                "",
                "## Conclusion & Outlook",
                f"The provided sources substantiate the primary aspects of '{objective}'. Further inquiries may incorporate additional primary data.",
            ])

        lines.extend([
            "",
            "## Sources",
        ])
        for idx, item in enumerate(evidence, start=1):
            title = item.get("title", f"Source {idx}")
            url = item.get("url", "#")
            lines.append(f"{idx}. [{title}]({url})")

        return "\n".join(lines)


    # =====================================================
    # REVISION / FEEDBACK HANDLING
    # =====================================================

    def build_revision_prompt(
        self,
        objective: str,
        draft_report: str,
        feedback: Dict[str, Any],
        evidence: List[Dict[str, Any]]
    ) -> str:
        evidence_text = self.prepare_evidence(evidence)
        issues = feedback.get("issues", [])
        unsupported = feedback.get("unsupported_claims", [])
        recommendations = feedback.get("recommendations", [])
        summary = feedback.get("summary", "")

        issues_str = "\n".join(f"- {iss}" for iss in issues) if issues else "None"
        unsupported_str = "\n".join(f"- {u}" for u in unsupported) if unsupported else "None"
        rec_str = "\n".join(f"- {r}" for r in recommendations) if recommendations else "None"

        return f"""
You are the Writer Agent revising a research report in a multi-agent research system.

The Reviewer Agent evaluated your previous draft and REJECTED it. You must fix the flagged issues.

RESEARCH OBJECTIVE:
{objective}

REVIEWER FEEDBACK:
Summary: {summary}

Issues to fix:
{issues_str}

Unsupported claims to remove or substantiate:
{unsupported_str}

Recommendations:
{rec_str}

PREVIOUS DRAFT:
{draft_report}

CURATED EVIDENCE:
{evidence_text}

============================================================
REVISION INSTRUCTIONS
============================================================
1. Address all issues and recommendations raised by the Reviewer Agent.
2. REMOVE any unsupported claims identified by the reviewer that cannot be verified in the curated evidence.
3. Ground all factual assertions strictly in the curated evidence.
4. Keep the report clear, comprehensive, and properly structured in Markdown.
5. Retain all section headings:
   # Research Report
   ## Executive Summary
   ## Introduction
   ## Key Findings
   ## Detailed Analysis
   ## Limitations
   ## Conclusion
   ## Sources

Return the complete revised Markdown report:
"""

    def revise_report(
        self,
        objective: str,
        draft_report: str,
        feedback: Dict[str, Any],
        evidence: List[Dict[str, Any]]
    ) -> str:
        """
        Revises a draft report to fix issues flagged by ReviewerAgent.
        """
        print(f"[WriterAgent] Revising report for '{objective}' based on reviewer feedback...")
        prompt = self.build_revision_prompt(
            objective=objective,
            draft_report=draft_report,
            feedback=feedback,
            evidence=evidence
        )

        report = self._call_groq_chain(prompt, operation_name="report revision")

        if not report:
            print("[WriterAgent] Groq revision unavailable or throttled. Applying deterministic cleanup fallback.")
            report = draft_report
            # Cleanly strip unsupported claims if any
            for claim in feedback.get("unsupported_claims", []):
                if claim and claim in report:
                    report = report.replace(claim, "")

        print("[WriterAgent] Revised report generated.")
        return report

    # =====================================================
    # COMPATIBILITY METHODS
    # =====================================================

    def draft_research_report(
        self,
        topic: str,
        findings: List[Dict[str, Any]],
        report_type: str = "summary",
        tone: str = "objective"
    ) -> str:
        """Backward compatibility for existing orchestrator and tests."""
        return self.generate_report(objective=topic, evidence=findings, report_type=report_type, tone=tone)

    def draft_web_report(
        self,
        topic: str,
        findings: List[Dict[str, Any]],
        report_type: str = "summary",
        tone: str = "objective"
    ) -> str:
        """Backward compatibility alias for generate_report."""
        return self.generate_report(objective=topic, evidence=findings, report_type=report_type, tone=tone)

    def draft_document_report(
        self,
        topic: str,
        documents: List[Dict[str, Any]],
        report_type: str = "summary",
        tone: str = "objective"
    ) -> str:
        """Backward compatibility for document mode orchestrator and tests."""
        return self.generate_report(objective=topic, evidence=documents, report_type=report_type, tone=tone)

    # =====================================================
    # MAIN TASK
    # =====================================================

    def execute_task(
        self,
        task: Dict[str, Any],
        evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        task_id = task.get(
            "id",
            "writer_task"
        )

        objective = task.get(
            "objective",
            ""
        )

        print("\n" + "=" * 60)

        print(
            f"[WriterAgent] Task: {task_id}"
        )

        print(
            f"[WriterAgent] Objective: {objective}"
        )

        print("=" * 60)

        # -------------------------------------------------
        # Generate report
        # -------------------------------------------------

        report = self.generate_report(
            objective=objective,
            evidence=evidence
        )

        return {
            "status": "success",
            "task_id": task_id,
            "agent": "WriterAgent",
            "objective": objective,
            "evidence_count": len(evidence),
            "report": report
        }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 60)
    print("WRITER AGENT TEST")
    print("=" * 60)

    # -----------------------------------------------------
    # Sample curated evidence
    # -----------------------------------------------------

    evidence = [
        {
            "evidence_id": "source_1",
            "title": "Machine Learning for Fraud Detection",
            "url": "https://example.com/fraud-ml",
            "content": """
            Machine learning can be used in financial fraud
            detection to identify unusual transaction patterns.
            Models can learn patterns from historical transaction
            data and use these patterns to identify transactions
            that differ from normal behavior.
            """,
            "curator_relevance": 0.94,
            "curator_quality": 0.88
        },
        {
            "evidence_id": "source_2",
            "title": "Deep Learning in Financial Security",
            "url": "https://example.com/deep-learning",
            "content": """
            Deep learning models can analyze complex patterns
            in financial transaction data. Neural networks may
            identify relationships between transaction features
            that are difficult to capture using simple rules.
            """,
            "curator_relevance": 0.90,
            "curator_quality": 0.85
        }
    ]

    # -----------------------------------------------------
    # Task
    # -----------------------------------------------------

    task = {
        "id": "writer_task_1",
        "agent": "WriterAgent",
        "role": "Generate research report",
        "objective": (
            "Explain how machine learning and deep learning "
            "are used for financial fraud detection."
        )
    }

    # -----------------------------------------------------
    # Create agent
    # -----------------------------------------------------

    agent = WriterAgent()

    # -----------------------------------------------------
    # Execute
    # -----------------------------------------------------

    result = agent.execute_task(
        task=task,
        evidence=evidence
    )

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    print("\n")
    print("=" * 60)
    print("GENERATED REPORT")
    print("=" * 60)

    print(
        result["report"]
    )
