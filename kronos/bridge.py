"""Telegram bridge — adapted from Kronos I bridge.py.

Telethon userbot + webhook server for cron scripts.
Calls KronosAgent for message processing.
"""

import asyncio
import logging
import os
import random
import tempfile
import time

import aiohttp
from aiohttp import web
from telethon import TelegramClient, events
from telethon.tl.types import DocumentAttributeAudio

from kronos.audit import log_request
from kronos.config import settings
from kronos.graph import KronosAgent
from kronos.security.cost_guardian import get_guardian
from kronos.security.output_validator import validate_output
from kronos.swarm_store import get_swarm
from kronos.tts import get_voice_mode, set_voice_mode, should_synthesize, synthesize
from kronos.vision import analyze_image_bytes, is_supported_image_mime, is_vision_configured

log = logging.getLogger("kronos.bridge")

# Rate limiting state
RATE_LIMIT_MIN_DELAY = 2.0
RATE_LIMIT_GLOBAL_DELAY = 1.0
_last_send_per_chat: dict[int, float] = {}
_last_send_global: float = 0.0
_rate_lock = asyncio.Lock()

# Groq Whisper STT
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

# Default chat for cron notifications
DEFAULT_NOTIFY_CHAT = int(os.environ.get("DEFAULT_NOTIFY_CHAT", "0"))

# Agent lock — one request at a time
_agent_lock = asyncio.Lock()

# Agent reference (set in run_bridge)
_agent: KronosAgent | None = None
_client: TelegramClient | None = None
_my_id: int | None = None
_my_username: str | None = None

# Group routing (initialized in run_bridge after login)
_group_router = None  # GroupRouter | None


async def _rate_limit_wait(chat_id: int) -> None:
    """Enforce anti-spam delays before sending."""
    global _last_send_global
    async with _rate_lock:
        now = time.monotonic()
        last_chat = _last_send_per_chat.get(chat_id, 0.0)
        chat_wait = max(0.0, RATE_LIMIT_MIN_DELAY - (now - last_chat))
        global_wait = max(0.0, RATE_LIMIT_GLOBAL_DELAY - (now - _last_send_global))
        wait = max(chat_wait, global_wait)
        if wait > 0:
            await asyncio.sleep(wait)
        now = time.monotonic()
        _last_send_per_chat[chat_id] = now
        _last_send_global = now
        if len(_last_send_per_chat) > 200:
            _last_send_per_chat.clear()


async def _human_typing_delay(chat_id: int, text: str) -> None:
    """Simulate human typing speed."""
    chars = len(text)
    typing_secs = chars / random.uniform(40, 80)
    thinking_secs = random.uniform(0.3, 1.2)
    total = min(typing_secs + thinking_secs, 5.0)
    async with _client.action(chat_id, "typing"):
        await asyncio.sleep(total)


def _is_voice_message(event) -> bool:
    if not event.message.media:
        return False
    doc = getattr(event.message.media, "document", None)
    if not doc:
        return False
    return any(
        isinstance(attr, DocumentAttributeAudio) and attr.voice
        for attr in doc.attributes
    )


def _image_mime_type(event) -> str:
    if getattr(event.message, "photo", None):
        return "image/jpeg"
    doc = getattr(getattr(event.message, "media", None), "document", None)
    return str(getattr(doc, "mime_type", "") or "")


def _is_image_message(event) -> bool:
    if not getattr(event.message, "media", None):
        return False
    return is_supported_image_mime(_image_mime_type(event))


async def _download_image_bytes(event) -> tuple[bytes, str]:
    mime_type = _image_mime_type(event) or "image/jpeg"
    suffix = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(mime_type, ".img")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        await event.message.download_media(file=tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read(), mime_type
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def _analyze_image_message(event, caption: str) -> str:
    if not is_vision_configured():
        return (
            "Я получил изображение, но vision model не настроена. "
            "Нужно установить и авторизовать Codex CLI (`codex login`) "
            "или включить KAOS_VISION_PROVIDER=openai-api."
        )
    image_bytes, mime_type = await _download_image_bytes(event)
    result = await analyze_image_bytes(
        image_bytes,
        mime_type=mime_type,
        context=caption,
    )
    return result.text


def _compose_image_agent_message(caption: str, image_analysis: str) -> str:
    caption = caption.strip()
    user_request = caption or "Пользователь отправил изображение без подписи."
    return (
        f"{user_request}\n\n"
        "[Vision analysis]\n"
        f"{image_analysis}\n\n"
        "Ответь пользователю на основе анализа изображения. Если пользователь просит OCR, "
        "верни извлечённый текст; если это документ/скриншот/чек, кратко классифицируй "
        "и выдели важные детали/action items."
    )


async def _transcribe_voice(file_path: str) -> str:
    """Transcribe audio via Groq Whisper API."""
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        fh = open(file_path, "rb")
        try:
            data.add_field(
                "file", fh,
                filename=os.path.basename(file_path),
                content_type="audio/ogg",
            )
            data.add_field("model", GROQ_WHISPER_MODEL)
            async with session.post(
                GROQ_WHISPER_URL,
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                data=data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Groq STT error {resp.status}: {body}")
                result = await resp.json()
                return result.get("text", "").strip()
        finally:
            fh.close()


def _is_mentioned(event) -> bool:
    if event.is_reply:
        return True
    if _my_username and ("@" + _my_username.lower()) in event.raw_text.lower():
        return True
    if event.message.entities:
        from telethon.tl.types import MessageEntityMention, MessageEntityMentionName
        for ent in event.message.entities:
            if isinstance(ent, MessageEntityMentionName) and ent.user_id == _my_id:
                return True
            if isinstance(ent, MessageEntityMention):
                mentioned = event.raw_text[ent.offset:ent.offset + ent.length].lstrip("@").lower()
                if _my_username and mentioned == _my_username.lower():
                    return True
    return False


def _extract_topic_id(event) -> int | None:
    """Extract forum topic ID from a Telethon message event.

    In forum supergroups, messages belong to topics. The topic ID
    is used to isolate conversation contexts per topic.

    Telethon bot mode: reply_to.reply_to_msg_id = topic root message ID.
    General topic: reply_to_msg_id = 1 (or absent).
    """
    reply_to = getattr(event.message, "reply_to", None)
    if not reply_to:
        # General topic in forum groups may have no reply_to
        # Check if chat itself is a forum
        return None

    # reply_to_top_id = topic root (when replying to a message within topic)
    top_id = getattr(reply_to, "reply_to_top_id", None)
    if top_id:
        return top_id

    # forum_topic flag = direct message in a topic (not a reply)
    if getattr(reply_to, "forum_topic", False):
        return reply_to.reply_to_msg_id

    # Fallback: reply_to_msg_id might be the topic ID in forum groups
    msg_id = getattr(reply_to, "reply_to_msg_id", None)
    if msg_id:
        return msg_id

    return None


def _strip_mention(text: str) -> str:
    if not _my_username:
        return text
    import re
    cleaned = re.sub(r"@" + re.escape(_my_username), "", text, flags=re.IGNORECASE).strip()
    return cleaned if cleaned else text


async def _fetch_root_user_message(event) -> tuple[str, str]:
    """Walk the reply chain up to find the originating user message.

    Returns ``(text, sender_name)``. Both may be empty strings if the root
    cannot be resolved (e.g. the reply chain is broken, or the root is
    a bot message). The router already guarantees for Tier 3 that the
    immediate parent is a whitelisted user; this helper just fetches its
    contents for inclusion as context.
    """
    try:
        replied = await event.get_reply_message()
    except Exception:
        return "", ""
    if replied is None:
        return "", ""
    text = (replied.raw_text or replied.message or "").strip()
    sender_name = ""
    try:
        sender = await replied.get_sender()
        if sender is not None:
            sender_name = getattr(sender, "first_name", "") or getattr(sender, "username", "") or ""
    except Exception:
        pass
    return text, sender_name


async def _typing_loop(chat_id: int, stop_event: asyncio.Event) -> None:
    """Keep typing indicator active until stop_event is set."""
    try:
        while not stop_event.is_set():
            try:
                async with _client.action(chat_id, "typing"):
                    await asyncio.wait_for(stop_event.wait(), timeout=5.0)
                    return
            except TimeoutError:
                continue  # re-send typing every 5s
    except Exception:
        pass  # typing indicator is best-effort


async def _send_bot_api_message(chat_id: int, text: str, topic_id: int) -> None:
    """Send message via Bot API with message_thread_id (for DM Topics)."""
    url = f"https://api.telegram.org/bot{settings.tg_bot_token}/sendMessage"

    chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)] if len(text) > 4000 else [text]

    for chunk in chunks:
        body = {
            "chat_id": chat_id,
            "text": chunk,
            "message_thread_id": topic_id,
            "parse_mode": "Markdown",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        # Markdown parse error → retry without parse_mode
                        body.pop("parse_mode", None)
                        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=15)) as retry:
                            if retry.status != 200:
                                err = await retry.text()
                                if retry.status == 400 and "message_thread" in err:
                                    no_topic_body = dict(body)
                                    no_topic_body.pop("message_thread_id", None)
                                    async with session.post(
                                        url,
                                        json=no_topic_body,
                                        timeout=aiohttp.ClientTimeout(total=15),
                                    ) as no_topic_retry:
                                        if no_topic_retry.status != 200:
                                            fallback_err = await no_topic_retry.text()
                                            log.error(
                                                "Bot API send failed after topic fallback: %s %s",
                                                no_topic_retry.status,
                                                fallback_err[:200],
                                            )
                                else:
                                    log.error("Bot API send failed: %s %s", retry.status, err[:200])
        except Exception as e:
            log.error("Bot API send error: %s", e)
        if len(chunks) > 1:
            await asyncio.sleep(0.5)


async def _clear_context(chat_id: int, topic_id: int | None = None) -> str:
    """Clear conversation history for a chat/topic."""
    thread_id = f"{chat_id}:{topic_id}" if topic_id else str(chat_id)
    return await _agent.clear_context(thread_id)


async def _ask_agent(
    message: str,
    chat_id: int,
    user_id: int,
    topic_id: int | None = None,
    source_kind: str = "user",
    persist_user_turn: bool = True,
    extra_system_context: str = "",
) -> str | None:
    """Send message to KronosAgent and return response text.

    Shows typing indicator while processing. Returns error message
    instead of None on failure (so user always gets feedback).

    When topic_id is provided (forum group), each topic gets its own
    conversation context via separate thread_id.

    source_kind / persist_user_turn / extra_system_context are forwarded
    to KronosAgent.ainvoke — see its docstring. Group transport metadata
    (sender name, "you are in a group chat", peer answer being reacted to)
    must be passed via extra_system_context, NEVER inlined into `message`.
    This is the contract that stops peer text from polluting session
    history and causing verbatim-parrot replies.
    """
    # Topic-aware thread isolation
    thread_id = f"{chat_id}:{topic_id}" if topic_id else str(chat_id)

    # Start typing indicator
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_typing_loop(chat_id, stop_typing))

    start_ms = int(time.monotonic() * 1000)
    reply = None

    try:
        async with _agent_lock:
            reply = await _agent.ainvoke(
                message=message,
                thread_id=thread_id,
                user_id=str(user_id),
                session_id=str(chat_id),
                source_kind=source_kind,
                persist_user_turn=persist_user_turn,
                extra_system_context=extra_system_context,
            )
    except Exception as e:
        log.error("Agent error: %s", e)
        reply = "Произошла ошибка при обработке запроса. Попробуй ещё раз."
    finally:
        stop_typing.set()
        typing_task.cancel()

    if not reply:
        reply = "Не удалось получить ответ от агента. Попробуй переформулировать запрос."

    # Audit log
    duration_ms = int(time.monotonic() * 1000) - start_ms
    from kronos.router import classify_tier
    tier = classify_tier(message).value

    log_request(
        user_id=str(user_id),
        session_id=str(chat_id),
        tier=tier,
        input_text=message,
        output_text=reply,
        duration_ms=duration_ms,
        blocked="заблокирован" in reply,
    )

    return reply


async def _send_to_chat(
    chat_id: int,
    text: str,
    parse_mode: str | None = None,
    topic_id: int | None = None,
) -> None:
    """Send message to Telegram chat with rate limiting and chunking."""
    await _rate_limit_wait(chat_id)
    await _human_typing_delay(chat_id, text)

    kwargs = {}
    if parse_mode:
        kwargs["parse_mode"] = parse_mode
    if topic_id:
        kwargs["reply_to"] = topic_id

    if len(text) > 4000:
        chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]
        for chunk in chunks:
            await _client.send_message(chat_id, chunk, **kwargs)
            await asyncio.sleep(0.5)
    else:
        await _client.send_message(chat_id, text, **kwargs)


# --- Webhook server (for cron scripts, same API as Kronos I) ---


async def _handle_webhook(request: web.Request) -> web.Response:
    secret = request.headers.get("X-Webhook-Secret", "")
    if secret != settings.webhook_secret:
        return web.json_response({"error": "unauthorized"}, status=401)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    text = body.get("text") or body.get("message") or body.get("content", "")
    chat_id = int(body.get("chat_id", DEFAULT_NOTIFY_CHAT))
    parse_mode = body.get("parse_mode")
    topic_id = body.get("topic_id")
    if topic_id:
        topic_id = int(topic_id)

    if not text:
        return web.json_response({"error": "no text"}, status=400)

    log.info("[Webhook] → chat %d: %s", chat_id, text[:100])
    try:
        await _send_to_chat(chat_id, text, parse_mode=parse_mode, topic_id=topic_id)
        return web.json_response({"ok": True})
    except Exception as e:
        log.error("[Webhook] Send failed: %s", e)
        return web.json_response({"error": str(e)}, status=500)


async def _handle_history(request: web.Request) -> web.Response:
    secret = request.headers.get("X-Webhook-Secret", "")
    if secret != settings.webhook_secret:
        return web.json_response({"error": "unauthorized"}, status=401)

    chat_param = request.query.get("chat", "")
    if not chat_param:
        return web.json_response({"error": "missing 'chat' parameter"}, status=400)

    limit = min(int(request.query.get("limit", "200")), 500)
    offset_id = int(request.query.get("offset_id", "0"))

    try:
        entity = await _client.get_entity(chat_param)
    except Exception:
        return web.json_response({"error": f"chat not found: {chat_param}"}, status=404)

    messages = []
    async for msg in _client.iter_messages(entity, limit=limit, offset_id=offset_id):
        if not msg.text:
            continue
        messages.append({
            "id": msg.id,
            "date": msg.date.isoformat(),
            "text": msg.text,
            "is_outgoing": msg.out,
        })

    return web.json_response({
        "messages": messages,
        "chat": {
            "id": entity.id,
            "username": getattr(entity, "username", None),
            "first_name": getattr(entity, "first_name", None),
        },
        "total": len(messages),
        "has_more": len(messages) == limit,
        "oldest_id": messages[-1]["id"] if messages else 0,
    })


async def _handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "agent": "kaos"})


async def _start_webhook_server() -> None:
    app = web.Application()
    app.router.add_post("/webhook", _handle_webhook)
    app.router.add_get("/history", _handle_history)
    app.router.add_get("/health", _handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    webhook_port = int(os.environ.get("WEBHOOK_PORT", "8788"))
    site = web.TCPSite(runner, "0.0.0.0", webhook_port)
    await site.start()
    log.info("Webhook server listening on port %d", webhook_port)


# --- ASO command handler ---


async def _handle_aso_command(text: str) -> str | None:
    """Handle /aso commands. Returns reply text or None if not an ASO command."""
    if not text.startswith("/aso"):
        return None

    parts = text.strip().split(maxsplit=2)
    cmd = parts[1] if len(parts) > 1 else "help"

    from kronos.agents.aso import (
        aso_approve,
        aso_reject,
        aso_resume,
        aso_run,
        aso_skip,
        aso_status,
    )

    if cmd == "run":
        dry_run = "--dry-run" in text
        return await aso_run(dry_run=dry_run)
    elif cmd == "approve":
        return await aso_approve()
    elif cmd == "reject":
        comment = parts[2] if len(parts) > 2 else ""
        return await aso_reject(comment)
    elif cmd == "skip":
        return await aso_skip()
    elif cmd == "resume":
        return await aso_resume()
    elif cmd == "status":
        return await aso_status()
    else:
        return (
            "ASO команды:\n"
            "/aso run [--dry-run] — запустить цикл\n"
            "/aso status — текущий статус\n"
            "/aso approve — одобрить план\n"
            "/aso reject <комментарий> — отклонить\n"
            "/aso skip — пропустить цикл\n"
            "/aso resume — продолжить после ожидания"
        )


# --- Main entry ---


async def run_bridge(agent: KronosAgent) -> None:
    """Start Telethon client + webhook server, listen for messages."""
    global _agent, _client, _my_id, _my_username
    _agent = agent

    session_file = os.environ.get("SESSION_FILE", f"{settings.agent_name}.session")
    _client = TelegramClient(session_file, settings.tg_api_id, settings.tg_api_hash)

    is_bot = bool(settings.tg_bot_token)
    log.info("Starting %s bridge (mode: %s)", settings.agent_name, "bot" if is_bot else "userbot")
    log.info("Allowed users: %s", settings.telegram_access_description)

    @_client.on(events.NewMessage(incoming=True))
    async def handle_message(event):
        # Log ALL incoming events for debugging
        log.info(
            "[EVENT] chat=%s private=%s reply_to=%s text=%s",
            event.chat_id, event.is_private,
            getattr(event.message, "reply_to", None),
            (event.raw_text or "")[:50],
        )

        sender = await event.get_sender()
        user_id = sender.id
        text = event.raw_text

        if user_id == _my_id:
            return

        is_dm = event.is_private
        voice = _is_voice_message(event)
        image = _is_image_message(event)

        if not text and not voice and not image:
            return

        # Swarm ledger ingress: record every observed group message before
        # routing, so other agents (and our post-mortem tools) can see it
        # even if this agent decides to skip. DMs stay out of the swarm
        # ledger — they are 1:1 and already isolated per-agent.
        topic_id_inbound = _extract_topic_id(event) if not is_dm else None
        swarm = get_swarm() if not is_dm else None
        if swarm is not None and text:
            reply_to = getattr(event.message, "reply_to", None)
            reply_to_msg_id = getattr(reply_to, "reply_to_msg_id", None) if reply_to else None
            sender_type = "user"
            agent_name_tag: str | None = None
            if _group_router is not None:
                if _group_router._is_peer(user_id):
                    sender_type = "agent"
                    # Reverse lookup via the router's registry (single source of truth).
                    peer_uname = (getattr(sender, "username", "") or "").lower().lstrip("@")
                    agent_name_tag = (
                        _group_router._username_to_agent.get(peer_uname) if peer_uname else None
                    )
            swarm.record_inbound_message(
                chat_id=event.chat_id,
                topic_id=topic_id_inbound,
                msg_id=event.message.id,
                reply_to_msg_id=reply_to_msg_id,
                sender_id=user_id,
                sender_type=sender_type,
                agent_name=agent_name_tag,
                text=text,
            )

        # Group filtering
        decision = None
        if not is_dm:
            if _group_router:
                # Multi-agent group routing (all group types)
                decision = await _group_router.decide(event, _client)
                if not decision.should_respond:
                    # Count "skipped because another agent was addressed" as
                    # a successful addressing-correctness event, and count
                    # "skipped because another peer already replied" as a
                    # duplicate-prevention event.
                    if decision.addressing and decision.addressing.explicit_to_other:
                        swarm.incr_metric("addressing_respected")
                    return

                log.info(
                    "[GroupRouter] %s: tier=%d delay=%.0fs reason=%s",
                    settings.agent_name, decision.tier, decision.delay, decision.reason,
                )

                # Resolve root user message id for claim bookkeeping. For
                # user-triggered messages, root = the user's message itself.
                # For peer reactions (Tier 3) we look up the reply parent.
                reply_to = getattr(event.message, "reply_to", None)
                parent_msg_id = getattr(reply_to, "reply_to_msg_id", None) if reply_to else None
                root_msg_id = parent_msg_id if decision.tier == 3 and parent_msg_id else event.message.id

                eta_ts = time.time() + max(decision.delay, 0.0)
                swarm.claim_reply(
                    chat_id=event.chat_id,
                    topic_id=topic_id_inbound,
                    root_msg_id=root_msg_id,
                    trigger_msg_id=event.message.id,
                    agent_name=settings.agent_name,
                    tier=decision.tier,
                    eta_ts=eta_ts,
                    reason=decision.reason,
                )

                if decision.delay > 0:
                    await asyncio.sleep(decision.delay)

                # Post-delay recheck (Tier 2/3) — another agent may have
                # answered while we were waiting.
                still_ok = await _group_router.should_still_respond(
                    event, _client, tier=decision.tier,
                )
                if not still_ok:
                    swarm.cancel_claim(
                        chat_id=event.chat_id,
                        topic_id=topic_id_inbound,
                        trigger_msg_id=event.message.id,
                        agent_name=settings.agent_name,
                        reason="post-delay: peer replied first",
                    )
                    swarm.incr_metric("duplicate_replies_avoided")
                    return

                # Atomic arbitration across all agents.
                outcome = swarm.can_send_claim(
                    chat_id=event.chat_id,
                    topic_id=topic_id_inbound,
                    root_msg_id=root_msg_id,
                    agent_name=settings.agent_name,
                    tier=decision.tier,
                )
                if not outcome.won:
                    log.info("[Swarm] %s stands down: %s", settings.agent_name, outcome.reason)
                    swarm.cancel_claim(
                        chat_id=event.chat_id,
                        topic_id=topic_id_inbound,
                        trigger_msg_id=event.message.id,
                        agent_name=settings.agent_name,
                        reason=outcome.reason,
                    )
                    swarm.incr_metric("duplicate_replies_avoided")
                    return

            else:
                # Fallback: no router — only mentions/replies
                if not _is_mentioned(event):
                    return

        # DM: check allowed users
        if is_dm and not settings.is_telegram_user_allowed(user_id):
            log.info("Ignoring DM from unauthorized Telegram user %s", user_id)
            return

        image_analysis = ""

        # Voice transcription / image analysis
        if voice:
            if not settings.groq_api_key:
                return
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
                    tmp_path = tmp.name
                await event.message.download_media(file=tmp_path)
                clean_text = await _transcribe_voice(tmp_path)
                os.unlink(tmp_path)
            except Exception as e:
                log.error("[Voice] Failed: %s", e)
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                return
            if not clean_text:
                return
        elif image:
            clean_text = _strip_mention(text) if not is_dm else text
            try:
                image_analysis = await _analyze_image_message(event, clean_text)
            except Exception as e:
                log.error("[Vision] Failed: %s", e)
                reply = (
                    "Не удалось обработать изображение. "
                    "Проверь, что включён OpenAI/Codex vision provider и формат изображения поддерживается."
                )
                await _send_to_chat(event.chat_id, reply, topic_id=_extract_topic_id(event))
                return
        else:
            clean_text = _strip_mention(text) if not is_dm else text

        # Build transient system context for group chats. This context is
        # passed to ainvoke(extra_system_context=...) and is NEVER persisted
        # into session history — it disappears after the current LLM call.
        # This is what prevents peer text from being echoed back verbatim.
        group_extra_context = ""
        invoke_message = _compose_image_agent_message(clean_text, image_analysis) if image else clean_text
        invoke_source_kind = "user"
        invoke_persist = True

        if not is_dm and _group_router and decision is not None:
            sender_name = sender.first_name or "Unknown"
            is_peer_sender = _group_router._is_peer(user_id)

            if decision.tier == 3 and is_peer_sender:
                # Peer reaction: reframe the call so the agent responds to
                # the *root user question*, treating the peer's answer as
                # context to compare against — not as a new question.
                root_user_text, root_user_name = await _fetch_root_user_message(event)
                invoke_message = root_user_text or clean_text
                invoke_source_kind = "peer_reaction"
                invoke_persist = False  # ephemeral — don't pollute history
                peer_snippet = clean_text[:1500]
                group_extra_context = (
                    f"[Групповой чат] Пользователь{f' ({root_user_name})' if root_user_name else ''} "
                    f"задал вопрос. Другой агент ({sender_name}) уже ответил:\n"
                    f"---\n{peer_snippet}\n---\n"
                    f"Добавь МЕНЯЮЩУЮ смысл дельту только если у тебя есть "
                    f"критически важный иной угол. Если твой ответ по сути "
                    f"повторяет коллегу — промолчи и ответь одним словом 'PASS'. "
                    f"Никогда не цитируй и не перефразируй ответ коллеги целиком. "
                    f"Говори от своего лица, 2-3 коротких абзаца максимум."
                )
            else:
                # Regular user message in group chat: tell the agent it's in
                # a multi-agent room; keep the raw user text as `message`.
                group_extra_context = (
                    f"[Групповой чат] Сообщение от {sender_name}. "
                    f"Отвечай по существу, от своего лица, коротко (2-4 абзаца). "
                    f"Не комментируй ответы других агентов, если они есть. "
                    f"Не описывай свой процесс мышления или фреймворки."
                )

        # Extract forum topic_id (for topic-based context isolation)
        topic_id = _extract_topic_id(event)

        chat_type = "group" if not is_dm else "DM"
        topic_label = f" topic={topic_id}" if topic_id else ""
        log.info("[%s%s] %s (%d): %s", chat_type, topic_label, sender.first_name, user_id, clean_text[:100])

        if is_dm and not is_bot:
            await _rate_limit_wait(event.chat_id)
            try:
                await _client.send_read_acknowledge(event.chat_id, event.message)
            except Exception:
                pass  # Bot API sessions can't call ReadHistoryRequest

        # /clear command — reset conversation context for this chat/topic
        if clean_text.strip().lower() in ("/clear", "/reset"):
            reply = await _clear_context(event.chat_id, topic_id)
        # /voice command — toggle voice mode
        elif clean_text.strip().lower().startswith("/voice"):
            arg = clean_text.strip().lower().removeprefix("/voice").strip()
            if arg == "on":
                set_voice_mode(event.chat_id, True)
                reply = "Голосовой режим включён. Буду отвечать голосом на короткие сообщения."
            elif arg == "off":
                set_voice_mode(event.chat_id, False)
                reply = "Голосовой режим выключен. Голосом отвечаю только на голосовые."
            else:
                current = get_voice_mode(event.chat_id)
                set_voice_mode(event.chat_id, not current)
                if not current:
                    reply = "Голосовой режим включён. Буду отвечать голосом на короткие сообщения."
                else:
                    reply = "Голосовой режим выключен. Голосом отвечаю только на голосовые."
        # Cost guardian check
        else:
            guardian = get_guardian()
            allowed, budget_msg = guardian.check_budget(session_id=str(event.chat_id))
            if not allowed:
                reply = f"⚠️ {budget_msg}"
            # Intercept /aso commands before agent
            elif (aso_reply := await _handle_aso_command(clean_text)) is not None:
                reply = aso_reply
            else:
                # Call agent with new contract: raw user text as message,
                # group metadata as transient extra_system_context only.
                reply = await _ask_agent(
                    invoke_message,
                    event.chat_id,
                    user_id,
                    topic_id=topic_id,
                    source_kind=invoke_source_kind,
                    persist_user_turn=invoke_persist,
                    extra_system_context=group_extra_context,
                )

        # Peer-reaction "PASS" protocol: the agent is instructed to reply
        # with "PASS" when it has nothing meaningfully different to add.
        # Treat that as a no-op — do not send anything to the chat.
        if invoke_source_kind == "peer_reaction" and reply:
            stripped = reply.strip().strip("'\"`.!").upper()
            if stripped == "PASS" or stripped.startswith("PASS"):
                log.info("[%s%s] Peer-reaction PASS from %s", chat_type, topic_label, settings.agent_name)
                if swarm is not None:
                    swarm.cancel_claim(
                        chat_id=event.chat_id,
                        topic_id=topic_id_inbound,
                        trigger_msg_id=event.message.id,
                        agent_name=settings.agent_name,
                        reason="peer-reaction self-pass",
                    )
                return

        # Output validation — redact secrets, log issues
        validation = validate_output(reply)
        if not validation.is_clean:
            reply = validation.redacted_text

        await _rate_limit_wait(event.chat_id)

        # Topic messages: reply_to = topic root so message lands in correct topic
        # Regular DM: no reply_to needed
        # Regular group: reply to the user's message
        if topic_id:
            reply_to = topic_id  # sends into the topic thread
        elif not is_dm:
            reply_to = event.message.id  # reply in group
        else:
            reply_to = None

        # TTS: voice mode always, or mirror user's voice message
        voice_sent = False
        vm = get_voice_mode(event.chat_id)
        if should_synthesize(reply, user_sent_voice=voice, voice_mode=vm):
            voice_path = await synthesize(reply)
            if voice_path:
                try:
                    await _client.send_file(
                        event.chat_id,
                        voice_path,
                        voice_note=True,
                        reply_to=reply_to,
                    )
                    voice_sent = True
                except Exception as e:
                    log.error("Voice send failed: %s", e)
                finally:
                    if os.path.exists(voice_path):
                        os.unlink(voice_path)

        sent_msg = None
        if not voice_sent:
            if topic_id and settings.tg_bot_token:
                # Use Bot API with message_thread_id when a bot token is configured.
                await _send_bot_api_message(event.chat_id, reply, topic_id)
            elif topic_id:
                sent_msg = await _client.send_message(event.chat_id, reply, reply_to=topic_id)
            elif len(reply) > 4000:
                chunks = [reply[i:i + 4000] for i in range(0, len(reply), 4000)]
                for i, chunk in enumerate(chunks):
                    sent = await _client.send_message(
                        event.chat_id, chunk,
                        reply_to=event.message.id if i == 0 and not is_dm else None,
                    )
                    if i == 0:
                        sent_msg = sent
                    await asyncio.sleep(0.5)
            else:
                reply_to_msg = event.message.id if not is_dm else None
                sent_msg = await _client.send_message(event.chat_id, reply, reply_to=reply_to_msg)

        # Swarm ledger: mark claim as sent, record outbound message. DMs
        # and fallback-router paths (no decision) skip this.
        if not is_dm and swarm is not None and decision is not None:
            reply_msg_id = getattr(sent_msg, "id", None) if sent_msg is not None else None
            swarm.mark_sent(
                chat_id=event.chat_id,
                topic_id=topic_id_inbound,
                trigger_msg_id=event.message.id,
                agent_name=settings.agent_name,
                reply_msg_id=reply_msg_id,
            )
            if reply_msg_id is not None:
                swarm.record_outbound_message(
                    chat_id=event.chat_id,
                    topic_id=topic_id_inbound,
                    msg_id=reply_msg_id,
                    reply_to_msg_id=event.message.id,
                    agent_name=settings.agent_name,
                    text=reply,
                )
            # Metrics: tier breakdown of actual replies (count after-send
            # so failed sends don't inflate the denominator).
            swarm.incr_metric(f"replies_tier{decision.tier}")
            swarm.incr_metric("replies_total")

        reply_mode = "voice" if voice_sent else "text"
        log.info("[%s%s] Replied (%s) to %s: %s", chat_type, topic_label, reply_mode, sender.first_name, reply[:100])

    # --- Reaction handler (RL feedback loop) ---

    from telethon.tl.types import UpdateMessageReactions

    @_client.on(events.Raw(types=UpdateMessageReactions))
    async def handle_reaction(event: UpdateMessageReactions):
        """Handle Telegram reactions (👍/👎) for RL feedback."""
        try:
            chat_id = None
            # Extract chat_id from the peer
            peer = event.peer
            if hasattr(peer, "channel_id"):
                chat_id = peer.channel_id
            elif hasattr(peer, "chat_id"):
                chat_id = peer.chat_id
            elif hasattr(peer, "user_id"):
                chat_id = peer.user_id

            if not chat_id:
                return

            msg_id = event.msg_id

            # Get the reactions list
            reactions = event.reactions
            if not reactions or not reactions.results:
                return

            # Find our agent's outbound message in swarm_messages
            swarm = get_swarm()
            rows = swarm._db.read(
                """
                SELECT agent_name FROM swarm_messages
                WHERE msg_id = ? AND (chat_id = ? OR chat_id = ?)
                  AND sender_type = 'agent'
                LIMIT 1
                """,
                (msg_id, chat_id, -chat_id),
            )

            if not rows:
                # Not our message, skip
                return

            agent_name = rows[0]["agent_name"] or settings.agent_name

            # Process each reaction
            for r in reactions.results:
                emoticon = getattr(r.reaction, "emoticon", None)
                if not emoticon:
                    continue

                swarm.add_feedback(
                    agent_name=agent_name,
                    chat_id=chat_id,
                    msg_id=msg_id,
                    emoji=emoticon,
                )
                log.info(
                    "[Feedback] %s on msg %d in chat %d: %s",
                    agent_name, msg_id, chat_id, emoticon,
                )
        except Exception as e:
            log.warning("Reaction handler error (non-fatal): %s", e)

    if is_bot:
        await _client.start(bot_token=settings.tg_bot_token)
    else:
        await _client.start()
    me = await _client.get_me()
    _my_id = me.id
    _my_username = me.username
    log.info("Logged in as: %s (@%s, %d)", me.first_name, me.username, me.id)

    # Initialize group router for multi-agent chats
    global _group_router
    from kronos.group_router import GroupRouter
    _group_router = GroupRouter(
        agent_name=settings.agent_name,
        my_id=_my_id,
        my_username=_my_username,
        allowed_user_ids=settings.allowed_user_ids,
    )
    log.info("Group router initialized for %s", settings.agent_name)

    # Share client for cron jobs and other modules
    from kronos.telegram_client import set_client
    set_client(_client)

    await _start_webhook_server()

    log.info("Listening for messages and webhooks...")
    await _client.run_until_disconnected()
