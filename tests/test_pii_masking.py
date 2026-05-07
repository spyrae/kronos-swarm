import io
import json
import logging

from langchain_core.messages import AIMessage

from kronos.security.pii import mask_pii, mask_pii_object


def test_mask_pii_covers_supported_types():
    text = (
        "Email roman@example.com, RU +7 (916) 123-45-67, INT +1 2025550123, "
        "card 4111 1111 1111 1234, passport 4510 123456, ip 192.168.1.10."
    )

    masked = mask_pii(text)

    assert "roman@example.com" not in masked
    assert "+7 (916) 123-45-67" not in masked
    assert "+1 2025550123" not in masked
    assert "4111 1111 1111 1234" not in masked
    assert "4510 123456" not in masked
    assert "192.168.1.10" not in masked
    assert "***@***.com" in masked
    assert "+7***___**__" in masked
    assert "+***" in masked
    assert "****-****-****-1234" in masked
    assert "**** ******" in masked
    assert "***.***.***.***" in masked


def test_mask_pii_does_not_mask_names():
    assert mask_pii("Позвони Ивану завтра") == "Позвони Ивану завтра"


def test_mask_pii_object_masks_message_content_without_mutating_original():
    message = AIMessage(content="send to roman@example.com")

    masked = mask_pii_object(message)

    assert message.content == "send to roman@example.com"
    assert masked.content == "send to ***@***.com"


def test_pii_logging_filter_masks_msg_and_args():
    from kronos.logging import add_pii_filter

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    add_pii_filter(handler)

    logger = logging.getLogger("tests.pii")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info("Email %s card %s", "roman@example.com", "4111 1111 1111 1234")

    output = stream.getvalue()
    assert "roman@example.com" not in output
    assert "4111 1111 1111 1234" not in output
    assert "***@***.com" in output
    assert "****-****-****-1234" in output


def test_audit_request_previews_mask_pii(tmp_path, monkeypatch):
    from kronos import audit

    db_path = tmp_path / "kaos" / "session.db"
    monkeypatch.setattr(audit.settings, "db_path", str(db_path))
    audit._audit_dir = None

    audit.log_request(
        user_id="u1",
        session_id="s1",
        tier="standard",
        input_text="Contact roman@example.com or +79161234567",
        output_text="Paid with 4111111111111234 from 192.168.1.10",
        duration_ms=12,
    )

    audit_log = db_path.parent / "logs" / "audit.jsonl"
    entry = json.loads(audit_log.read_text(encoding="utf-8").strip())

    serialized = json.dumps(entry, ensure_ascii=False)
    assert "roman@example.com" not in serialized
    assert "+79161234567" not in serialized
    assert "4111111111111234" not in serialized
    assert "192.168.1.10" not in serialized
    assert "***@***.com" in entry["input_preview"]
    assert "+7***___**__" in entry["input_preview"]
    assert "****-****-****-1234" in entry["output_preview"]
    assert "***.***.***.***" in entry["output_preview"]


def test_tool_audit_summaries_mask_pii(tmp_path, monkeypatch):
    from kronos import audit

    db_path = tmp_path / "kaos" / "session.db"
    monkeypatch.setattr(audit.settings, "db_path", str(db_path))
    monkeypatch.setattr(audit.settings, "agent_name", "kaos")
    audit._audit_dir = None

    audit.log_tool_event("tool_call", {
        "name": "send_message",
        "call_id": "call-1",
        "args": {
            "email": "roman@example.com",
            "phone": "+79161234567",
        },
    })

    tool_log = db_path.parent / "logs" / "tool_calls.jsonl"
    entry = json.loads(tool_log.read_text(encoding="utf-8").strip())

    assert "roman@example.com" not in entry["args_summary"]
    assert "+79161234567" not in entry["args_summary"]
    assert "***@***.com" in entry["args_summary"]
    assert "+7***___**__" in entry["args_summary"]


def test_memory_masks_metadata_but_keeps_content(monkeypatch):
    from kronos.memory import store

    class FakeMemory:
        def __init__(self):
            self.messages = None
            self.kwargs = None

        def add(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return {"results": []}

    fake = FakeMemory()
    monkeypatch.setattr(store, "get_memory", lambda: fake)

    store.add_memories(
        [{"role": "user", "content": "My email is roman@example.com"}],
        user_id="u1",
        session_id="+79161234567",
    )

    assert fake.messages[0]["content"] == "My email is roman@example.com"
    assert fake.kwargs["metadata"]["session_id"] == "+7***___**__"


def test_llm_callback_wrapper_masks_langfuse_payloads():
    from kronos.llm import _PIIMaskingCallbackHandler

    class FakeCallback:
        def __init__(self):
            self.prompts = None
            self.error = None

        def on_llm_start(self, serialized, prompts, **kwargs):
            self.prompts = prompts

        def on_llm_error(self, error, **kwargs):
            self.error = error

    fake = FakeCallback()
    wrapper = _PIIMaskingCallbackHandler(fake)

    wrapper.on_llm_start({}, ["Email roman@example.com and phone +79161234567"])
    wrapper.on_llm_error(RuntimeError("Failed for card 4111111111111234"))

    assert fake.prompts == ["Email ***@***.com and phone +7***___**__"]
    assert "4111111111111234" not in str(fake.error)
    assert "****-****-****-1234" in str(fake.error)
