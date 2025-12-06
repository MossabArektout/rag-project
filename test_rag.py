"""
Test RAG system end-to-end
"""
import requests
import json
from loguru import logger

BASE_URL = "http://localhost:8000"


def test_rag_system():
    """Test complete RAG pipeline"""
    
    logger.info("="*60)
    logger.info("TESTING COMPLETE RAG SYSTEM")
    logger.info("="*60)
    
    # Upload document
    logger.info("\n1. Uploading document...")
    with open("uploads/sample.pdf", "rb") as f:
        files = {"file": ("sample.pdf", f, "application/pdf")}
        response = requests.post(f"{BASE_URL}/api/documents/upload", files=files)
    
    if response.status_code != 201:
        logger.error(f"Upload failed: {response.text}")
        return
    
    doc_id = response.json()["document_id"]
    logger.success(f"✓ Document uploaded: {doc_id[:8]}...")
    
    # Test questions
    questions = [
        "What is the company's mission?",
        "What products does the company offer?",
        "What are the company values?",
        "Tell me about Smart Document Processing",
        "What technologies does the company use?"
    ]
    
    logger.info(f"\n2. Testing {len(questions)} questions with RAG...")
    
    for i, question in enumerate(questions, 1):
        logger.info(f"\n--- Question {i} ---")
        logger.info(f"Q: {question}")
        
        payload = {
            "question": question,
            "top_k": 3
        }
        
        response = requests.post(f"{BASE_URL}/api/qa/ask", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"\nA: {data['answer']}\n")
            logger.info(f"Confidence: {data['confidence_score']:.2%}")
            logger.info(f"Time: {data['processing_time_seconds']}s")
            logger.info(f"Sources: {len(data['sources'])}")
        else:
            logger.error(f"Failed: {response.text}")
    
    logger.info("\n" + "="*60)
    logger.success("✓ RAG SYSTEM TEST COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    test_rag_system()