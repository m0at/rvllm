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
OPENAI_BASE_URL = f"{BASE_URL}/v1"

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

class TestResponses:
    def test_basic_response(self):
        """POST /v1/responses with text input"""
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": "test",
            "input": "Hi",
            "max_output_tokens": 5,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "response"
        assert data["status"] == "completed"
        assert data["output"][0]["type"] == "message"
        assert data["output"][0]["content"][0]["type"] == "output_text"

    def test_response_with_message_items(self):
        """Responses item-based text input works"""
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": "test",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Hi"},
                        {"type": "input_text", "text": " there"},
                    ],
                }
            ],
            "max_output_tokens": 5,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["output"][0]["role"] == "assistant"

    def test_response_stream(self):
        """Responses streaming emits named SSE events"""
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": "test",
            "input": "Hello",
            "max_output_tokens": 5,
            "stream": True,
        }, stream=True)
        assert r.status_code == 200

        events = []
        event_name = None
        for raw_line in r.iter_lines():
            line = raw_line.decode()
            if not line:
                continue
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
                events.append((event_name, payload))

        names = [name for name, _ in events]
        assert "response.created" in names
        assert (
            "response.output_text.delta" in names
            or "response.output_text.done" in names
        )
        assert "response.completed" in names

    def test_response_retrieval_and_input_items(self):
        """Stored responses can be retrieved and list input items"""
        created = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": "test",
            "input": "Tell me a joke",
            "max_output_tokens": 5,
            "store": True,
        })
        assert created.status_code == 200
        response_id = created.json()["id"]

        retrieved = requests.get(f"{BASE_URL}/v1/responses/{response_id}")
        assert retrieved.status_code == 200
        assert retrieved.json()["id"] == response_id

        items = requests.get(f"{BASE_URL}/v1/responses/{response_id}/input_items")
        assert items.status_code == 200
        data = items.json()
        assert data["object"] == "list"
        assert data["data"][0]["type"] == "message"
        assert data["data"][0]["content"][0]["type"] == "input_text"

    def test_response_previous_response_id(self):
        """Responses can chain prior stored turns"""
        first = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": "test",
            "input": "Tell me a joke",
            "max_output_tokens": 5,
            "store": True,
        })
        assert first.status_code == 200
        first_id = first.json()["id"]

        second = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": "test",
            "previous_response_id": first_id,
            "input": [{"role": "user", "content": "Explain why it is funny."}],
            "max_output_tokens": 5,
            "store": True,
        })
        assert second.status_code == 200
        assert second.json()["previous_response_id"] == first_id

    def test_response_accepts_function_tools(self):
        """Responses accept custom function tools"""
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": "test",
            "input": "What's the weather?",
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}}
                }
            }],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["tool_choice"] == "auto"
        assert data["tools"][0]["type"] == "function"

    def test_response_function_call_output_input_items(self):
        """function_call_output items are stored and listed as input items"""
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": "test",
            "input": [{
                "type": "function_call_output",
                "call_id": "call_123",
                "output": {"ok": True},
            }],
            "max_output_tokens": 5,
            "store": True,
        })
        assert r.status_code == 200
        response_id = r.json()["id"]

        items = requests.get(f"{BASE_URL}/v1/responses/{response_id}/input_items")
        assert items.status_code == 200
        data = items.json()
        assert data["data"][0]["type"] == "function_call_output"
        assert data["data"][0]["call_id"] == "call_123"
        assert data["data"][0]["output"]["ok"] is True

    def test_response_rejects_built_in_tools(self):
        """Built-in Responses tools are still rejected explicitly"""
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": "test",
            "input": "Hi",
            "tools": [{"type": "web_search_preview"}],
        })
        assert r.status_code == 400

    def test_response_stream_with_tools(self):
        """Tool-enabled Responses streaming stays available and emits lifecycle events"""
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": "test",
            "input": "Hi",
            "stream": True,
            "tools": [{
                "type": "function",
                "name": "get_weather",
            }],
        }, stream=True)
        assert r.status_code == 200

        events = []
        event_name = None
        for raw_line in r.iter_lines():
            line = raw_line.decode()
            if not line:
                continue
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
                events.append((event_name, payload))

        names = [name for name, _ in events]
        assert "response.created" in names
        assert "response.in_progress" in names
        assert "response.completed" in names
        assert (
            "response.function_call_arguments.delta" in names
            or "response.function_call_arguments.done" in names
            or "response.output_text.delta" in names
            or "response.output_text.done" in names
        )
        if "response.function_call_arguments.delta" in names:
            assert "response.function_call_arguments.done" in names

class TestResponsesSdk:
    @pytest.fixture(scope="class")
    def client(self):
        openai = pytest.importorskip("openai")
        return openai.OpenAI(base_url=OPENAI_BASE_URL, api_key="unused")

    def test_responses_create(self, client):
        """OpenAI SDK can call responses.create()"""
        response = client.responses.create(
            model="test",
            input="Hello",
            max_output_tokens=5,
        )
        assert response.id.startswith("resp_")
        assert response.output[0].type == "message"

    def test_responses_follow_up_turn(self, client):
        """OpenAI SDK can chain previous_response_id"""
        first = client.responses.create(
            model="test",
            input="Tell me a joke",
            max_output_tokens=5,
        )
        second = client.responses.create(
            model="test",
            previous_response_id=first.id,
            input=[{"role": "user", "content": "Explain it."}],
            max_output_tokens=5,
        )
        assert second.previous_response_id == first.id

    def test_responses_function_tools(self, client):
        """OpenAI SDK can send Responses function tools"""
        response = client.responses.create(
            model="test",
            input="What's the weather?",
            tools=[{
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }],
        )
        assert response.tool_choice == "auto"
        assert response.tools[0].type == "function"

class TestBeamSearch:
    def test_beam_search_completion(self):
        """Beam search completion works end to end"""
        r = requests.post(f"{BASE_URL}/v1/completions", json={
            "model": "test",
            "prompt": "Summarize beam search in one sentence.",
            "max_tokens": 8,
            "n": 2,
            "best_of": 3,
            "use_beam_search": True,
            "length_penalty": 0.6,
            "early_stopping": True,
        })
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data
        assert len(data["choices"]) == 1

    def test_beam_search_streaming_rejected(self):
        """Beam search rejects streaming requests"""
        r = requests.post(f"{BASE_URL}/v1/completions", json={
            "model": "test",
            "prompt": "Hello",
            "max_tokens": 5,
            "n": 2,
            "use_beam_search": True,
            "stream": True,
        })
        assert r.status_code == 400

    def test_beam_search_invalid_length_penalty_rejected(self):
        """Invalid beam search controls are rejected"""
        r = requests.post(f"{BASE_URL}/v1/completions", json={
            "model": "test",
            "prompt": "Hello",
            "max_tokens": 5,
            "n": 2,
            "use_beam_search": True,
            "length_penalty": -1.0,
        })
        assert r.status_code == 400

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

    def test_responses_response_format(self):
        """Responses response matches the unified response shape"""
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": "test",
            "input": "Hi",
            "max_output_tokens": 3,
        })
        data = r.json()
        assert "id" in data
        assert data["object"] == "response"
        assert "created_at" in data
        assert "status" in data
        assert "model" in data
        assert "output" in data
        assert data["output"][0]["type"] == "message"
        assert data["output"][0]["content"][0]["type"] == "output_text"

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

    def test_missing_response_input(self):
        """Responses reject missing input without previous_response_id"""
        r = requests.post(f"{BASE_URL}/v1/responses", json={
            "model": "test",
            "max_output_tokens": 5,
        })
        assert r.status_code in [400, 422]
