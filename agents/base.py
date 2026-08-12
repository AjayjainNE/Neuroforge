"""
NeuroForge Base Agent
Abstract base class for all specialized agents.
Handles LLM client initialization, retries, self-critique, and tracing.
"""
from __future__ import annotations
import json
import time
import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from config import settings, LLMProvider
from models.schemas import AgentRole, AgentTrace
from core.memory import memory_graph
from core.prompts import SELF_CRITIQUE_SYSTEM


def _build_client() -> AsyncOpenAI:
    """Build an async OpenAI-compatible client for NVIDIA, Mistral, or Anthropic."""
    return AsyncOpenAI(
        api_key=settings.active_api_key,
        base_url=settings.active_base_url,
        timeout=120.0,
        max_retries=3,
    )


def _build_anthropic_client():
    """Build native Anthropic client (used when provider=anthropic)."""
    import anthropic as _anthropic
    return _anthropic.AsyncAnthropic(timeout=120.0, max_retries=3)


# Shared client instance
_client: Optional[AsyncOpenAI] = None
_anthropic_client = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = _build_anthropic_client()
    return _anthropic_client


class BaseAgent(ABC):
    """
    Abstract base for all NeuroForge agents.
    Provides: LLM calls, retry logic, JSON extraction, self-critique, tracing.
    """

    role: AgentRole
    use_code_model: bool = False

    def __init__(self):
        self.client = get_client()
        self.model = (
            settings.code_model if self.use_code_model else settings.primary_model
        )
        self._use_anthropic = settings.LLM_PROVIDER.value == "anthropic"
        if self._use_anthropic:
            self.anthropic_client = get_anthropic_client()
            self.model = "claude-sonnet-4-20250514"

    # ─── Core LLM Call ───────────────────────────────────────────────────────

    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        model_override: Optional[str] = None,
    ) -> tuple[str, int]:
        if self._use_anthropic:
            return await self._call_anthropic(messages, temperature, max_tokens)
        return await self._call_openai_compat(messages, temperature, max_tokens, model_override)

    async def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int]:
        system_prompt = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                user_messages.append(m)
        if not user_messages:
            user_messages = [{"role": "user", "content": "Begin."}]
        response = await self.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=min(max_tokens, 8000),
            system=system_prompt,
            messages=user_messages,
            temperature=temperature,
        )
        content = response.content[0].text if response.content else ""
        tokens = (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
        return content.strip(), tokens

    async def _call_openai_compat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        model_override: Optional[str] = None,
    ) -> tuple[str, int]:
        """
        Call OpenAI-compatible endpoint (NVIDIA NIM / Mistral).
        Returns (content, tokens_used).
        """
        model = model_override or self.model
        t0 = time.time()

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            return content.strip(), tokens

        except Exception as e:
            # Fallback: try with lower max_tokens
            if max_tokens > 2048:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2048,
                )
                content = response.choices[0].message.content or ""
                tokens = response.usage.total_tokens if response.usage else 0
                return content.strip(), tokens
            raise

    # ─── JSON Extraction ─────────────────────────────────────────────────────

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Robustly extract JSON from LLM output.
        Handles markdown fences, leading text, and trailing comments.
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        # Find JSON object boundaries
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

        # Try line-by-line stripping
        lines = []
        in_json = False
        brace_depth = 0
        for line in cleaned.splitlines():
            if "{" in line:
                in_json = True
            if in_json:
                lines.append(line)
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0 and in_json:
                    break
        if lines:
            try:
                return json.loads("\n".join(lines))
            except json.JSONDecodeError:
                pass

        return {"_raw": text, "_parse_failed": True}

    # ─── Self-Critique ────────────────────────────────────────────────────────

    async def _self_critique(self, output: str) -> Optional[str]:
        """Run a self-critique pass on the agent's own output."""
        if not settings.ENABLE_SELF_CRITIQUE:
            return None
        try:
            messages = [
                {"role": "system", "content": SELF_CRITIQUE_SYSTEM},
                {"role": "user", "content": f"Critique this agent output:\n\n{output[:2000]}"},
            ]
            content, _ = await self._call_llm(
                messages, temperature=0.1, max_tokens=512
            )
            result = self._extract_json(content)
            critique = result.get("critique", "")
            penalty = float(result.get("confidence_penalty", 0.0))
            return critique if critique else None
        except Exception:
            return None

    # ─── Tracing ─────────────────────────────────────────────────────────────

    def _record_trace(
        self,
        session_id: str,
        input_summary: str,
        output_summary: str,
        full_output: str,
        tokens_used: int,
        confidence: float = 1.0,
        critique: Optional[str] = None,
    ) -> AgentTrace:
        trace = AgentTrace(
            agent=self.role,
            input_summary=input_summary,
            output_summary=output_summary,
            full_output=full_output,
            tokens_used=tokens_used,
            confidence=confidence,
            critique=critique,
        )
        memory_graph.record_trace(session_id, trace)
        return trace

    # ─── Abstract Interface ───────────────────────────────────────────────────

    @abstractmethod
    async def run(
        self,
        session_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the agent's primary task."""
        ...
