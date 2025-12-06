"""
Test script for API endpoints
"""
import requests
import json
from pathlib import Path
from loguru import logger

BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test health check endpoint"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Health Check")
    logger.info("="*60)
    
    response = requests.get(f"{BASE_URL}/api/health")
    logger.info(f"Status Code: {response.status_code}")
    logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    logger.success("✓ Health check passed")


def test_upload_document(pdf_path: str):
    """Test document upload"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Upload Document")
    logger.info("="*60)
    
    file_path = Path(pdf_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {pdf_path}")
        return None
    
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/pdf")}
        response = requests.post(f"{BASE_URL}/api/documents/upload", files=files)
    
    logger.info(f"Status Code: {response.status_code}")
    
    if response.status_code == 201:
        data = response.json()
        logger.info(f"Response: {json.dumps(data, indent=2)}")
        logger.success("✓ Document uploaded successfully")
        return data["document_id"]
    else:
        logger.error(f"Upload failed: {response.text}")
        return None


def test_list_documents():
    """Test list documents endpoint"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: List Documents")
    logger.info("="*60)
    
    response = requests.get(f"{BASE_URL}/api/documents/")
    logger.info(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        logger.info(f"Total Documents: {data['total_documents']}")
        for doc in data["documents"]:
            logger.info(f"  - {doc['document_name']} (ID: {doc['document_id'][:8]}...)")
        logger.success("✓ List documents passed")
    else:
        logger.error(f"Failed: {response.text}")


def test_get_document(document_id: str):
    """Test get specific document"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Get Document Details")
    logger.info("="*60)
    
    response = requests.get(f"{BASE_URL}/api/documents/{document_id}")
    logger.info(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        logger.info(f"Response: {json.dumps(data, indent=2)}")
        logger.success("✓ Get document passed")
    else:
        logger.error(f"Failed: {response.text}")


def test_ask_question(question: str, document_ids=None):
    """Test question answering"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Ask Question")
    logger.info("="*60)
    
    payload = {
        "question": question,
        "document_ids": document_ids,
        "top_k": 3
    }
    
    logger.info(f"Question: {question}")
    
    response = requests.post(f"{BASE_URL}/api/qa/ask", json=payload)
    logger.info(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        logger.info(f"\nAnswer: {data['answer']}")
        logger.info(f"\nConfidence: {data['confidence_score']}")
        logger.info(f"Processing Time: {data['processing_time_seconds']}s")
        logger.info(f"\nSources ({len(data['sources'])}):")
        for i, source in enumerate(data["sources"], 1):
            logger.info(f"  {i}. {source['document_name']} (Page {source['page_number']}) - Similarity: {source['similarity_score']}")
        logger.success("✓ Ask question passed")
    else:
        logger.error(f"Failed: {response.text}")


def test_delete_document(document_id: str):
    """Test document deletion"""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Delete Document")
    logger.info("="*60)
    
    response = requests.delete(f"{BASE_URL}/api/documents/{document_id}")
    logger.info(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        logger.info(f"Response: {json.dumps(data, indent=2)}")
        logger.success("✓ Delete document passed")
    else:
        logger.error(f"Failed: {response.text}")


def run_all_tests():
    """Run complete API test suite"""
    logger.info("="*60)
    logger.info("SMART INTERNSHIP ASSISTANT - API TEST SUITE")
    logger.info("="*60)
    logger.info("Make sure the API server is running: uvicorn app.main:app --reload")
    logger.info("="*60)
    
    try:
        # Test 1: Health check
        test_health_check()
        
        # Test 2: Upload document
        document_id = test_upload_document("uploads/sample.pdf")
        
        if not document_id:
            logger.error("Upload failed, skipping remaining tests")
            return
        
        # Test 3: List documents
        test_list_documents()
        
        # Test 4: Get document details
        test_get_document(document_id)
        
        # Test 5: Ask questions
        test_ask_question("What does the company do?")
        test_ask_question("What are the company values?")
        test_ask_question("Tell me about the products", document_ids=[document_id])
        
        # Test 6: Delete document (optional - comment out to keep documents)
        # test_delete_document(document_id)
        
        logger.info("\n" + "="*60)
        logger.success("✓ ALL API TESTS COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")


if __name__ == "__main__":
    run_all_tests()