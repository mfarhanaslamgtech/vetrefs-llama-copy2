import app.app as app_module
import pytest
from app.config.config import Config
from fastapi.testclient import TestClient

from app import app_bckp
from app.core.dependencies import get_chatbot, get_database_handler


class FakeDatabaseHandler:
    def __init__(self):
        self.chat_history = {
            (1, 1): {
                "messages": [
                    {"content": "What is the meaning of life?", "type": "human"},
                    {"content": "A test answer.", "type": "ai"},
                ]
            }
        }

    def generate_chat_session_id(self, user_id):
        return 2

    def generate_chat_title(self, user_id, chat_session_id):
        return "What is the meaning of life?"

    def save_chat_title(self, user_id, chat_session_id, generated_chat_title):
        return None

    def get_chat_history(self, user_id, chat_session_id):
        if (user_id, chat_session_id) not in self.chat_history:
            raise ValueError("Chat history not found")
        return [{"messages": self.chat_history[(user_id, chat_session_id)]["messages"]}]

    def get_chat_sessions(self, user_id):
        if user_id != 1:
            return []
        return [1]

    def get_chat_title(self, user_id, chat_session_id):
        if user_id == 1 and chat_session_id == 1:
            return "What is the meaning of life?"
        return None

    def does_chat_title_exist(self, user_id, chat_session_id, chat_title):
        return user_id == 1 and chat_session_id == 1

    def rename_chat_title(self, user_id, chat_session_id, chat_title):
        return None

    def check_session_existence(self, user_id, chat_session_id):
        return user_id == 1 and chat_session_id == 1

    def delete_chat_session(self, user_id, chat_session_id):
        return None

    def check_user_existence(self, user_id):
        return user_id == 1

    def delete_all_chats(self, user_id):
        return None


class FakeChatbot:
    def answer_question(self, user_id, chat_session_id, question):
        return "A test answer."


class FakeChatbotModelError:
    def answer_question(self, user_id, chat_session_id, question):
        raise ValueError("OpenAI API key is missing or invalid.")


@pytest.fixture
def test_app():
    app_bckp.dependency_overrides[get_database_handler] = lambda: FakeDatabaseHandler()
    app_bckp.dependency_overrides[get_chatbot] = lambda: FakeChatbot()
    with TestClient(app_bckp) as client:
        yield client
    app_bckp.dependency_overrides.clear()


def test_health_request(test_app, monkeypatch):
    monkeypatch.setattr(app_module, "check_openai_health", lambda: (True, {"service": "openai", "status": "up", "model": Config.LLM_NAME, "message": None}))
    response = test_app.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'
    assert data['service'] == 'vetpedia-api'
    assert data['checks']['openai']['status'] == 'up'


def test_get_chat_history_request(test_app):
    response = test_app.get('/v1/chatbot/history/1/1')
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


def test_invalid_get_chat_history_request(test_app):
    response = test_app.get('/v1/chatbot/history/1/1000')
    assert response.status_code == 404


def test_get_chat_list_request(test_app):
    response = test_app.get('/v1/chatbot/chat_list/1')
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert 'chat_sessions' in data


def test_invalid_get_chat_list_request(test_app):
    response = test_app.get('/v1/chatbot/chat_list/1000')
    assert response.status_code == 404


def test_post_request(test_app):
    payload = {
        "user_id": 1,
        "chat_session_id": 1,
        "question": "What is the meaning of life?"
    }
    response = test_app.post('/v1/chatbot', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 'answer' in data
    assert 'chat_session_id' in data


def test_post_request_returns_structured_openai_error():
    app_bckp.dependency_overrides[get_database_handler] = lambda: FakeDatabaseHandler()
    app_bckp.dependency_overrides[get_chatbot] = lambda: FakeChatbotModelError()
    with TestClient(app_bckp) as client:
        response = client.post(
            '/v1/chatbot',
            json={
                "user_id": 1,
                "chat_session_id": 1,
                "question": "What is the meaning of life?"
            },
        )
    app_bckp.dependency_overrides.clear()

    assert response.status_code == 503
    data = response.json()
    assert data["error"]["code"] == "openai_auth_error"
    assert data["error"]["service"] == "openai"
