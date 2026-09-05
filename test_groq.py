"""
Quick sanity check for the Groq API key and client.

Run with: python test_groq.py
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

import groq


def test_groq_connection():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("FAILED: GROQ_API_KEY not found in environment (.env file).")
        sys.exit(1)

    model_name = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

    try:
        client = groq.Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Say 'Hello world' and nothing else."}
            ],
            temperature=0.0,
            max_tokens=100,
        )

        answer = response.choices[0].message.content.strip()
        print(f"SUCCESS: Groq responded with model '{model_name}':")
        print(f"  -> {answer}")

    except groq.AuthenticationError as e:
        print(f"FAILED: Authentication error - check your GROQ_API_KEY. Details: {e}")
        sys.exit(1)
    except groq.RateLimitError as e:
        print(f"FAILED: Rate limit exceeded. Details: {e}")
        sys.exit(1)
    except groq.APIConnectionError as e:
        print(f"FAILED: Network/connection error. Details: {e}")
        sys.exit(1)
    except groq.APIStatusError as e:
        print(f"FAILED: Groq API error (status {e.status_code}). Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FAILED: Unexpected error. Details: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_groq_connection()
