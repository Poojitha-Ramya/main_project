import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from google import genai
try:
    from groq import Groq
except ImportError:
    Groq = None

from backend.config import config, get_fallback_chain, get_groq_fallback_chain


class SummaryChatAgent:
    """
    Summary Chat Assistant Agent.
    
    Operates strictly post-research to resolve user doubts, answer questions,
    and clarify details exclusively grounded in the generated research summary.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.gemini_api_key = (
            gemini_api_key
            or config.GEMINI_API_KEY
            or os.getenv("GEMINI_API_KEY", "")
        )
        self.groq_api_key = (
            groq_api_key
            or config.GROQ_API_KEY
            or os.getenv("GROQ_API_KEY", "")
        )
        self.model_name = model_name or getattr(config, "DEFAULT_MODEL", "gemini-3.6-flash")
        self.fallback_models = get_fallback_chain(self.model_name)

        # Initialize Gemini Client if key available
        self.gemini_client = None
        if self.gemini_api_key:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception as e:
                print(f"[SummaryChatAgent] Warning: Failed to init Gemini client: {e}")

        # Initialize Groq Client if key available
        self.groq_client = None
        if Groq and self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                print(f"[SummaryChatAgent] Warning: Failed to init Groq client: {e}")

    def _build_system_instruction(self, topic: str, summary: str, sources: Optional[List[Dict[str, Any]]] = None) -> str:
        sources_str = ""
        if sources:
            source_lines = []
            for s in sources[:10]:
                title = s.get("title", "Source")
                url = s.get("url", "")
                snippet = s.get("snippet", "")[:180]
                source_lines.append(f"- **{title}** ({url}): {snippet}")
            sources_str = "AVAILABLE RESEARCH EVIDENCE & CITATIONS:\n" + "\n".join(source_lines) + "\n\n"

        return f"""You are the specialized "Research Summary Assistant" for Mini Researcher.
Your sole and exclusive purpose is to clarify doubts, explain concepts, unpack findings, and answer user questions strictly based on the provided Research Summary.

============================================================
RESEARCH TOPIC:
{topic}
============================================================

RESEARCH SUMMARY (YOUR EXCLUSIVE SOURCE OF TRUTH):
============================================================
{summary}
============================================================

{sources_str}
STRICT BOUNDARY & OPERATIONAL RULES:
1. ONLY ANSWER ABOUT THE RESEARCH SUMMARY:
   You are strictly scoped to the research summary text provided above. Every answer, clarification, explanation, or synthesis you give must be directly relevant to and grounded in this summary report and topic.

2. STRICT GUARDRAIL FOR OUT-OF-SCOPE QUESTIONS:
   If the user asks a question that is NOT related to this research summary (for example, asking for general coding problems, weather, sports, personal advice, general trivia, unrelated science/history, or anything outside this summary's scope), you MUST POLITELY REFUSE and state clearly:
   "I am your Research Summary Assistant, dedicated strictly to answering doubts and clarifying findings regarding your research summary on '{topic}'. I cannot answer questions outside of this research report. Please feel free to ask about any findings, analysis, data points, or conclusions in this summary!"

3. RESOLVING DOUBTS & QUESTIONS:
   - When the user has doubts about a claim, conclusion, or methodology in the summary, break it down clearly and cite the corresponding parts of the summary.
   - If the user asks about an aspect not covered in the summary, explicitly clarify that the current summary does not contain details on that specific question, while summarizing what the report does provide on related angles.
   - Use clean, structured Markdown formatting (bullet points, bold highlights, concise sections).
   - Be helpful, conversational, precise, and objective.
"""

    def answer_question(
        self,
        question: str,
        summary: str,
        topic: str = "Research Summary",
        history: Optional[List[Dict[str, str]]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answers a user doubt or question strictly based on the research summary.
        Supports multi-turn chat history.
        """
        if not summary or not summary.strip():
            raise ValueError("Research summary is required to answer doubts. Please generate a research report first.")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        system_instruction = self._build_system_instruction(topic=topic, summary=summary, sources=sources)

        # Build prompt messages
        history = history or []
        
        # 1. Try Gemini models first with automatic redirection
        primary_model = model or self.model_name
        models_to_try = get_fallback_chain(primary_model)

        if self.gemini_client:
            # Build conversation context for Gemini
            conversation_history_text = ""
            if history:
                conversation_history_text = "\nPREVIOUS CONVERSATION TURNS:\n"
                for msg in history[-6:]:  # Keep recent 6 turns
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    conversation_history_text += f"{role}: {msg.get('content', '')}\n"
                conversation_history_text += "\n"

            prompt_content = f"{system_instruction}\n{conversation_history_text}USER'S CURRENT DOUBT / QUESTION:\n{question}\n\nASSISTANT'S GROUNDED ANSWER:"

            for candidate in models_to_try:
                try:
                    response = self.gemini_client.models.generate_content(
                        model=candidate,
                        contents=prompt_content
                    )
                    if response and response.text:
                        return {
                            "status": "success",
                            "answer": response.text.strip(),
                            "model_used": candidate,
                            "provider": "Gemini",
                            "topic": topic
                        }
                except Exception as e:
                    print(f"[SummaryChatAgent] Gemini candidate {candidate} failed: {e}. Trying next...")
                    continue

        # 2. Resilient failover to Groq
        if self.groq_client:
            groq_models = get_groq_fallback_chain()
            groq_messages = [
                {"role": "system", "content": system_instruction}
            ]
            for msg in history[-6:]:
                role = "user" if msg.get("role") == "user" else "assistant"
                groq_messages.append({"role": role, "content": msg.get("content", "")})
            groq_messages.append({"role": "user", "content": question})

            for g_model in groq_models:
                try:
                    completion = self.groq_client.chat.completions.create(
                        model=g_model,
                        messages=groq_messages,
                        temperature=0.3,
                        max_tokens=1024
                    )
                    if completion.choices and completion.choices[0].message.content:
                        return {
                            "status": "success",
                            "answer": completion.choices[0].message.content.strip(),
                            "model_used": g_model,
                            "provider": "Groq",
                            "topic": topic
                        }
                except Exception as ge:
                    print(f"[SummaryChatAgent] Groq candidate {g_model} failed: {ge}. Trying next...")
                    continue

        # If all AI providers fail
        raise RuntimeError(
            "Chat Assistant is temporarily unable to reach language model providers. "
            "Please verify your GEMINI_API_KEY or GROQ_API_KEY in .env."
        )
