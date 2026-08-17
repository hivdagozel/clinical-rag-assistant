import pytest
import os
import httpx

@pytest.mark.online
def test_api_connection():
    api_url = os.getenv('MEDICINE_API_URL', 'http://localhost:3000')
    try:
        res = httpx.get(api_url, timeout=3.0)
        assert res.status_code < 500
    except httpx.RequestError:
        pytest.skip('Medicine API not available')

@pytest.mark.online
def test_gemini_connection():
    import socket
    try:
        socket.create_connection(('generativelanguage.googleapis.com', 443), timeout=3.0)
    except Exception:
        pytest.skip('Gemini API not reachable (DNS or network error)')
    assert True
