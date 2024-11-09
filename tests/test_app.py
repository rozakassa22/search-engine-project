# tests/test_app.py
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"My Search Engine" in response.data

def test_search(client):
    response = client.post('/search', data={'query': 'example'})
    assert response.status_code == 200
    assert b"documents found" in response.data
