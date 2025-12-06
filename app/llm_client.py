from typing import Optional
from loguru import logger
from config.settings import settings
import openai


class LLMClient:
    """Client for LLM API (OpenAI)"""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize LLM client
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        
        if not self.api_key:
            logger.warning("OpenAI API key not provided. LLM features will not work.")
        else:
            openai.api_key = self.api_key
            logger.info(f"LLM client initialized with model: {self.model}")
    
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
        if not self.api_key:
            logger.error("Cannot generate answer: OpenAI API key not configured")
            return None
        
        temp = temperature or settings.llm_temperature
        max_tok = max_tokens or settings.llm_max_tokens
        
        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question, context)
        
        try:
            logger.info(f"Generating answer with {self.model}")
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=max_tok
            )
            
            answer = response.choices[0].message.content.strip()
            
            logger.success(f"Answer generated ({len(answer)} characters)")
            
            return answer
            
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
        return self.api_key is not None and len(self.api_key) > 0