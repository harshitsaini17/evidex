"""
LLM interface and implementations for the Evidex Q&A system.

Provides an abstract interface for LLM interactions and a mock
implementation for testing without external API calls.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re


@dataclass
class LLMResponse:
    """Raw response from an LLM.
    
    Attributes:
        content: The text content of the response
        raw: Optional raw response data from the provider
    """
    content: str
    raw: dict | None = None


class LLMInterface(ABC):
    """Abstract interface for LLM providers.
    
    All LLM implementations must inherit from this class and implement
    the generate method.
    """
    
    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response for the given prompt.
        
        Args:
            prompt: The full prompt to send to the LLM
            
        Returns:
            LLMResponse containing the generated text
        """
        pass


class MockLLM(LLMInterface):
    """Mock LLM for testing purposes.
    
    This implementation allows configuring predetermined responses
    for testing the Q&A system without external API calls.
    
    The mock can operate in two modes:
    1. Fixed response: Always returns the configured response
    2. Keyword matching: Returns different responses based on question content
    """
    
    def __init__(
        self,
        default_response: str | None = None,
        keyword_responses: dict[str, str] | None = None,
    ):
        """Initialize the mock LLM.
        
        Args:
            default_response: Response to return when no keyword matches
            keyword_responses: Dict mapping keywords to responses
        """
        self.default_response = default_response or self._not_found_response()
        self.keyword_responses = keyword_responses or {}
        self.call_history: list[str] = []
    
    def generate(self, prompt: str) -> LLMResponse:
        """Generate a mock response based on configuration.
        
        Records the prompt in call_history for test assertions.
        
        Args:
            prompt: The prompt (used for keyword matching)
            
        Returns:
            Configured LLMResponse
        """
        self.call_history.append(prompt)
        
        # Check for keyword matches
        prompt_lower = prompt.lower()
        for keyword, response in self.keyword_responses.items():
            if keyword.lower() in prompt_lower:
                return LLMResponse(content=response)
        
        return LLMResponse(content=self.default_response)
    
    @staticmethod
    def _not_found_response() -> str:
        """Default response indicating information not found."""
        return json.dumps({
            "answer": "Not defined in the paper",
            "citations": [],
            "confidence": "high"
        })
    
    @staticmethod
    def create_response(
        answer: str,
        citations: list[str],
        confidence: str = "high"
    ) -> str:
        """Helper to create properly formatted JSON response.
        
        Args:
            answer: The answer text
            citations: List of paragraph IDs
            confidence: "high" or "low"
            
        Returns:
            JSON string in expected format
        """
        return json.dumps({
            "answer": answer,
            "citations": citations,
            "confidence": confidence
        })


def parse_llm_response(response: LLMResponse) -> dict:
    """Parse LLM response into structured format.
    
    Attempts to extract JSON from the response, handling cases where
    the JSON might be embedded in other text.
    
    Args:
        response: The raw LLM response
        
    Returns:
        Dict with answer, citations, and confidence
        
    Raises:
        ValueError: If response cannot be parsed
    """
    content = response.content.strip()
    
    # Try direct JSON parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in the response (handle nested braces for citations array)
    json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Try to find any JSON object with more lenient matching
    json_match = re.search(r'\{.*?"answer".*?"citations".*?"confidence".*?\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # If all parsing fails, raise error
    raise ValueError(f"Could not parse LLM response as JSON: {content[:200]}")


class GroqLLM(LLMInterface):
    """Groq LLM implementation using their API.
    
    Uses the Groq API for fast inference with models like Llama.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
    ):
        """Initialize the Groq LLM.
        
        Args:
            api_key: Groq API key (falls back to GROQ_API_KEY env var)
            model: Model to use (default: llama-3.3-70b-versatile)
            temperature: Sampling temperature (0.0 for deterministic)
        """
        import os
        from groq import Groq
        
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY or pass api_key.")
        
        self.model = model
        self.temperature = temperature
        self.client = Groq(api_key=self.api_key)
        self.call_history: list[str] = []
    
    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response using the Groq API.
        
        Args:
            prompt: The full prompt to send
            
        Returns:
            LLMResponse containing the generated text
        """
        self.call_history.append(prompt)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )
        
        content = response.choices[0].message.content
        
        return LLMResponse(
            content=content,
            raw={
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
        )
