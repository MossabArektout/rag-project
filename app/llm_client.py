from typing import Optional
from loguru import logger
from config.settings import settings
import google.generativeai as genai


class LLMClient:
    """Client for LLM API (Google Gemini)"""

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize LLM client

        Args:
            api_key: Google API key
            model: Model name to use (default: gemini-pro)
        """
        self.api_key = api_key or settings.google_api_key
        self.model_name = model or settings.llm_model
        self.model = None

        if not self.api_key:
            logger.warning("Google API key not provided. LLM features will not work.")
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                logger.info(f"LLM client initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.model = None

    def generate_answer(
        self,
        question: str,
        context: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> Optional[str]:
        """
        Generate answer using LLM

        Args:
            question: User's question
            context: Retrieved context
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Generated answer or None if failed
        """
        if not self.model:
            logger.error("Cannot generate answer: Gemini model not configured")
            return None

        temp = temperature or settings.llm_temperature
        max_tok = max_tokens or settings.llm_max_tokens

        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question, context)

        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            logger.info(f"Generating answer with {self.model_name}")

            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temp,
                max_output_tokens=max_tok,
            )

            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            # Extract answer
            if response and response.text:
                answer = response.text.strip()
                logger.success(f"Answer generated ({len(answer)} characters)")
                return answer
            else:
                logger.error("No response text generated")
                return None

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return None

    def _build_system_prompt(self) -> str:
        """Build system prompt for the LLM"""
        return """You are a helpful AI assistant that answers questions based on provided context from company documents.

Your responsibilities:
1. Answer questions accurately using ONLY the information in the provided context
2. If the context doesn't contain enough information to answer, say so clearly
3. Cite sources when possible (mention document names and page numbers)
4. Be concise but comprehensive
5. If asked about something not in the context, politely explain you can only answer based on the provided documents

Guidelines:
- Stay factual and grounded in the context
- Don't make up information not present in the context
- Use professional, clear language
- Structure your answers logically
- If multiple sources provide relevant info, synthesize them"""

    def _build_user_prompt(self, question: str, context: str) -> str:
        """
        Build user prompt with question and context

        Args:
            question: User's question
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        prompt = f"""Context from company documents:

{context}

---

Question: {question}

Please provide a comprehensive answer based solely on the context above. If the context doesn't contain enough information, say so clearly."""

        return prompt

    def check_availability(self) -> bool:
        """
        Check if LLM is available and configured

        Returns:
            True if available, False otherwise
        """
        return self.model is not None
