"""
LLM interface and implementations for the Evidex Q&A system.

Provides an abstract interface for LLM interactions and a mock
implementation for testing without external API calls.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import logging
import re

logger = logging.getLogger(__name__)


# =============================================================================
# JSON Extraction Helpers
# =============================================================================

def extract_json_block(text: str) -> str:
    """Extract first balanced JSON object from text using stack-based parsing.
    
    This is more robust than regex for nested JSON structures like
    objects containing arrays.
    
    Args:
        text: Text potentially containing a JSON object
        
    Returns:
        Extracted JSON substring
        
    Raises:
        ValueError: If no valid JSON object found
    """
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found in text")
    
    stack = []
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        ch = text[i]
        
        # Handle escape sequences in strings
        if escape_next:
            escape_next = False
            continue
        
        if ch == '\\':
            escape_next = True
            continue
        
        # Handle string boundaries
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        
        # Only count braces outside strings
        if not in_string:
            if ch == '{':
                stack.append('{')
            elif ch == '}':
                if not stack:
                    raise ValueError("Unbalanced JSON braces - extra closing brace")
                stack.pop()
                if not stack:
                    return text[start:i+1]
    
    raise ValueError("Unbalanced JSON braces - unclosed object")


def safe_parse_json(content: str) -> dict:
    """Parse JSON from LLM response with fallback to extraction.
    
    Tries direct parsing first, then falls back to extracting
    the first balanced JSON block if the content contains extra text.
    
    Args:
        content: LLM response content
        
    Returns:
        Parsed dict
        
    Raises:
        ValueError: If no valid JSON can be extracted
    """
    content = content.strip()
    
    # Try direct parse first (fastest path)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Handle markdown code blocks
    if '```json' in content:
        match = re.search(r'```json\s*\n?(.*?)\n?```', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
    elif '```' in content:
        match = re.search(r'```\s*\n?(.*?)\n?```', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
    
    # Fall back to balanced-braces extraction
    try:
        json_str = extract_json_block(content)
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(
            f"Could not parse LLM response as JSON: {content[:200]}... "
            f"Error: {e}"
        )


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
    
    Uses robust JSON extraction that handles:
    - Direct JSON responses
    - JSON wrapped in markdown code blocks
    - JSON embedded in explanatory text
    - Nested structures (objects with arrays)
    
    Args:
        response: The raw LLM response
        
    Returns:
        Dict with answer, citations, and confidence
        
    Raises:
        ValueError: If response cannot be parsed
    """
    return safe_parse_json(response.content)


class GroqLLM(LLMInterface):
    """Groq LLM implementation using their API.
    
    Uses the Groq API for fast inference with models like Llama.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "moonshotai/kimi-k2-instruct",
        temperature: float = 0.0,
        timeout: float | None = None,
    ):
        """Initialize the Groq LLM.
        
        Args:
            api_key: Groq API key (falls back to GROQ_API_KEY env var)
            model: Model to use (default: llama-3.1-8b-instant)
            temperature: Sampling temperature (0.0 for deterministic)
            timeout: Request timeout in seconds (default: None for no timeout)
        """
        import os
        from groq import Groq
        
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY or pass api_key.")
        
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.client = Groq(api_key=self.api_key, timeout=timeout)
        self.call_history: list[str] = []
    
    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response using the Groq API.
        
        Args:
            prompt: The full prompt to send
            
        Returns:
            LLMResponse containing the generated text
            
        Raises:
            RuntimeError: If API call fails or response is malformed
            TimeoutError: If request times out
        """
        self.call_history.append(prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1024,
            )
        except Exception as e:
            error_msg = str(e).lower()
            # Check for timeout-related errors
            if 'timeout' in error_msg or 'timed out' in error_msg:
                logger.error("Groq API timeout: %s", e)
                raise TimeoutError(f"Groq API request timed out: {e}") from e
            # Check for rate limit errors (let caller handle appropriately)
            if 'rate' in error_msg and 'limit' in error_msg:
                logger.error("Groq API rate limit: %s", e)
                raise RuntimeError(f"Groq API rate limit exceeded: {e}") from e
            # Generic API error
            logger.error("Groq API error: %s", e)
            raise RuntimeError(f"Groq API error: {e}") from e
        
        # Defensive extraction of response content
        if not response.choices:
            raise RuntimeError("Groq API returned empty choices")
        
        choice = response.choices[0]
        if not hasattr(choice, 'message') or choice.message is None:
            raise RuntimeError("Groq API response missing message")
        
        content = getattr(choice.message, 'content', None)
        if content is None:
            raise RuntimeError("Groq API response missing message content")
        
        # Safely extract usage info (may vary by SDK version)
        usage_info = {}
        if hasattr(response, 'usage') and response.usage is not None:
            usage_info = {
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                "total_tokens": getattr(response.usage, 'total_tokens', 0),
            }
            logger.info(f"Token usage - Prompt: {usage_info['prompt_tokens']}, Completion: {usage_info['completion_tokens']}, Total: {usage_info['total_tokens']}")
        
        return LLMResponse(
            content=content,
            raw={
                "model": getattr(response, 'model', self.model),
                "usage": usage_info,
            }
        )
