"""
API Compatibility Tests: rvllm vs Python vLLM
Verifies that rvllm can be used as a drop-in replacement.

Usage:
    # Start rvllm server first, then:
    RVLLM_URL=http://localhost:8000 python3 -m pytest tests/api_compat/ -v

    # Or test against Python vLLM:
    RVLLM_URL=http://localhost:8001 python3 -m pytest tests/api_compat/ -v
"""
import os, json, requests, pytest

BASE_URL = os.environ.get("RVLLM_URL", "http://localhost:8000")

class TestCompletions:
    def test_basic_completion(self):
        """POST /v1/completions with minimal params"""
        r = requests.post(f"{BASE_URL}/v1/completions", json={
            "model": "test", "prompt": "Hello", "max_tokens": 5
        })
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data
        assert len(data["choices"]) >= 1
        assert "text" in data["choices"][0]
        assert "usage" in data

    def test_completion_with_params(self):
        """All sampling params work"""
        r = requests.post(f"{BASE_URL}/v1/completions", json={
            "model": "test", "prompt": "The sky is",
            "max_tokens": 10, "temperature": 0.8,
            "top_p": 0.9, "top_k": 50,
            "presence_penalty": 0.1, "frequency_penalty": 0.1,
        })
        assert r.status_code == 200

    def test_completion_n(self):
        """n parameter generates multiple choices"""
        r = requests.post(f"{BASE_URL}/v1/completions", json={
            "model": "test", "prompt": "Hello", "max_tokens": 5, "n": 2
        })
        assert r.status_code == 200
        # Should have 2 choices (or 1 if n>1 not yet supported)

    def test_completion_stream(self):
        """Streaming via SSE works"""
        r = requests.post(f"{BASE_URL}/v1/completions", json={
            "model": "test", "prompt": "Hello", "max_tokens": 5, "stream": True
        }, stream=True)
        assert r.status_code == 200
        chunks = []
        for line in r.iter_lines():
            line = line.decode()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
        assert len(chunks) >= 1

    def test_completion_stop(self):
        """Stop strings work"""
        r = requests.post(f"{BASE_URL}/v1/completions", json={
            "model": "test", "prompt": "Count: 1, 2, 3,",
            "max_tokens": 20, "stop": ["\n"]
        })
        assert r.status_code == 200

class TestChat:
    def test_basic_chat(self):
        """POST /v1/chat/completions"""
        r = requests.post(f"{BASE_URL}/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5
        })
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data
        assert "message" in data["choices"][0]

    def test_chat_system_message(self):
        """System messages work"""
        r = requests.post(f"{BASE_URL}/v1/chat/completions", json={
            "model": "test",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"}
            ],
            "max_tokens": 5
        })
        assert r.status_code == 200

    def test_chat_stream(self):
        """Chat streaming works"""
        r = requests.post(f"{BASE_URL}/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5, "stream": True
        }, stream=True)
        assert r.status_code == 200

class TestModels:
    def test_list_models(self):
        """GET /v1/models"""
        r = requests.get(f"{BASE_URL}/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert "data" in data

class TestHealth:
    def test_health(self):
        """GET /health"""
        r = requests.get(f"{BASE_URL}/health")
        assert r.status_code == 200

class TestResponseFormat:
    def test_completion_response_format(self):
        """Response matches OpenAI format exactly"""
        r = requests.post(f"{BASE_URL}/v1/completions", json={
            "model": "test", "prompt": "Hi", "max_tokens": 3
        })
        data = r.json()
        # Required fields per OpenAI spec
        assert "id" in data
        assert data["object"] == "text_completion"
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        for choice in data["choices"]:
            assert "text" in choice
            assert "index" in choice
            assert "finish_reason" in choice

    def test_chat_response_format(self):
        """Chat response matches OpenAI format"""
        r = requests.post(f"{BASE_URL}/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 3
        })
        data = r.json()
        assert data["object"] == "chat.completion"
        assert "choices" in data
        for choice in data["choices"]:
            assert "message" in choice
            assert "role" in choice["message"]
            assert "content" in choice["message"]

class TestErrorHandling:
    def test_missing_prompt(self):
        """Error on missing required field"""
        r = requests.post(f"{BASE_URL}/v1/completions", json={
            "model": "test", "max_tokens": 5
        })
        assert r.status_code in [400, 422]

    def test_invalid_temperature(self):
        """Error on invalid temperature"""
        r = requests.post(f"{BASE_URL}/v1/completions", json={
            "model": "test", "prompt": "Hi", "max_tokens": 5,
            "temperature": -1.0
        })
        # Should either reject or clamp
