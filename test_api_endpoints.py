import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    print("\n--- 1. Testing Document Upload via /api/upload ---")
    file_content = (
        "Project Mini Researcher Benchmark 2026:\n"
        "Latency decreased by 45% when using semantic sentence chunking and ChromaDB local indexing.\n"
        "Gemini-3.5-flash showed 99.4% grounded accuracy without hallucinations."
    )
    files = {
        "file": ("benchmarks_2026.txt", file_content.encode("utf-8"), "text/plain")
    }
    upload_res = requests.post(f"{BASE_URL}/api/upload", files=files)
    print("Upload status:", upload_res.status_code)
    print("Upload response:", upload_res.json())
    assert upload_res.status_code == 200

    print("\n--- 2. Checking /api/documents list ---")
    docs_res = requests.get(f"{BASE_URL}/api/documents")
    print("Documents:", docs_res.json())
    assert len(docs_res.json()["documents"]) >= 1

    print("\n--- 3. Testing /api/research with mode='document' ---")
    research_payload = {
        "query": "What latency improvements and grounded accuracy were observed?",
        "mode": "document",
        "include_uploaded": True
    }
    res = requests.post(f"{BASE_URL}/api/research", json=research_payload)
    print("Research status:", res.status_code)
    data = res.json()
    print("Mode returned:", data.get("mode"))
    print("Subqueries returned:", data.get("subqueries"))
    print("Uploaded documents:", data.get("uploaded_documents"))
    print("Report length:", len(data.get("report", "")))

    assert data.get("mode") == "document"
    assert data.get("subqueries") == []
    assert len(data.get("report", "")) > 100
    print("\n>>> ALL API ENDPOINT INTEGRATION TESTS PASSED! <<<")

if __name__ == "__main__":
    test_api()
