"""Application settings via Pydantic Settings."""

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = os.environ.get("KAOS_ENV_FILE") or os.environ.get("KRONOS_ENV_FILE") or ".env"
load_dotenv(_ENV_FILE, override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # LLM Providers
    fireworks_api_key: str = ""  # Fireworks AI — Kimi K2.5 Turbo (standard tier)
    deepseek_api_key: str = ""   # DeepSeek V3 (lite tier)
    openai_api_key: str = ""
    kaos_standard_provider_chain: str = "kimi,deepseek"
    kaos_lite_provider_chain: str = "deepseek,kimi"
    kaos_vision_provider: str = "codex-cli"  # codex-cli | openai-api
    kaos_vision_model: str = "gpt-5.2-codex"
    kaos_codex_command: str = "codex"
    kaos_vision_timeout_seconds: int = 120

    # Telegram Bridge
    tg_api_id: int = 0
    tg_api_hash: str = ""
    tg_bot_token: str = ""  # Bot API token (alternative to userbot)
    allowed_users: str = ""  # comma-separated user IDs
    allow_all_users: bool = False  # if false, empty ALLOWED_USERS blocks DMs
    webhook_secret: str = ""

    # Voice STT
    groq_api_key: str = ""

    # Discord
    discord_bot_token: str = ""
    discord_allowed_guilds: str = ""  # comma-separated guild IDs

    # MCP Servers
    brave_api_key: str = ""
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    notion_api_key: str = ""
    exa_api_key: str = ""
    composio_api_key: str = ""

    # Capability gates — public-safe defaults.
    # Risky runtime mutation and infrastructure actions are opt-in for trusted
    # local deployments. Demo/fresh-clone mode should stay read-mostly.
    enable_dynamic_tools: bool = False
    require_dynamic_tool_sandbox: bool = True
    enable_mcp_gateway_management: bool = False
    enable_dynamic_mcp_servers: bool = False
    enable_server_ops: bool = False

    # Memory — per-agent isolation (resolved in model_post_init from agent_name
    # when left empty). Explicit env overrides still win.
    mem0_qdrant_path: str = ""
    # Per-agent data directory for SQLite databases (FTS/KG/session/etc).
    db_dir: str = ""
    # Shared swarm ledger database path (cross-agent). Single file for all
    # agents in the same deployment.
    swarm_db_path: str = "./data/swarm.db"

    # NTFY push notifications
    ntfy_url: str = "https://ntfy.sh"
    ntfy_token: str = ""
    ntfy_topic: str = "persona-alerts"

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""

    # Analytics data sources
    zabbix_url: str = ""
    zabbix_token: str = ""
    grafana_url: str = ""
    grafana_service_account_token: str = ""
    sentry_auth_token: str = ""
    sentry_org: str = ""
    sentry_project: str = ""

    # Product analytics
    posthog_api_key: str = ""
    posthog_host: str = "https://eu.i.posthog.com"
    posthog_project_id: str = ""

    # Supabase (for product stats)
    supabase_url: str = ""
    supabase_service_role_key: str = ""

    # Web analytics
    ym_oauth_token: str = ""
    ym_counter_id: str = ""
    ga4_property_id: str = ""

    # Revenue
    revenuecat_api_key: str = ""
    revenuecat_project_id: str = ""

    # AI costs
    litellm_base_url: str = ""
    litellm_admin_key: str = ""

    # Dev velocity
    linear_api_key: str = ""

    # Agent
    agent_name: str = "kronos"  # kronos | nexus — selects workspaces/<name>/
    workspace_path: str = ""  # override; if empty, resolved from agent_name
    db_path: str = ""  # override; if empty, resolved to ./data/<agent_name>/session.db
    context_strategy: str = "summarize"  # summarize | sliding_window | hybrid

    def model_post_init(self, __context: object) -> None:
        """Resolve empty paths from agent_name after initialization.

        Per-agent storage layout (keeps the 6 processes isolated and prevents
        the historical file-lock contention on a shared Qdrant directory):

            ./data/<agent_name>/session.db       — conversation history
            ./data/<agent_name>/memory_fts.db    — FTS5 fact index
            ./data/<agent_name>/knowledge_graph.db
            ./data/<agent_name>/qdrant/          — Mem0 vector store
            ./data/swarm.db                       — shared cross-agent ledger

        Legacy env overrides (``DB_PATH=./data/<name>.db``) are silently
        rewritten to the new layout so existing ``.env`` files keep working
        after upgrade — the migration in ``app._migrate_legacy_layout`` then
        moves the physical file to match.
        """
        # Detect a legacy flat DB_PATH and rewrite it in place.
        legacy_flat = f"./data/{self.agent_name}.db"
        if self.db_path in ("", legacy_flat, f"data/{self.agent_name}.db"):
            self.db_path = ""  # force re-resolution below

        if not self.db_dir:
            self.db_dir = f"./data/{self.agent_name}"
        if not self.db_path:
            self.db_path = f"{self.db_dir}/session.db"
        if not self.mem0_qdrant_path:
            self.mem0_qdrant_path = f"{self.db_dir}/qdrant"

        # Langfuse uses LANGFUSE_BASEURL in shared env, map to langfuse_host
        if not self.langfuse_host:
            baseurl = os.environ.get("LANGFUSE_BASEURL", "").rstrip("/")
            if baseurl:
                self.langfuse_host = baseurl

    @property
    def allowed_user_ids(self) -> set[int]:
        return {int(uid) for uid in self._allowed_user_tokens()[0]}

    @property
    def invalid_allowed_user_tokens(self) -> tuple[str, ...]:
        return self._allowed_user_tokens()[1]

    def _allowed_user_tokens(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not self.allowed_users:
            return (), ()

        valid: list[str] = []
        invalid: list[str] = []
        for raw_uid in self.allowed_users.split(","):
            uid = raw_uid.strip()
            if not uid or uid.startswith("#"):
                continue
            if uid.isdecimal():
                valid.append(uid)
            else:
                invalid.append(uid)
        return tuple(valid), tuple(invalid)

    @property
    def telegram_access_description(self) -> str:
        allowed_count = len(self.allowed_user_ids)
        if allowed_count:
            suffix = "user" if allowed_count == 1 else "users"
            return f"configured ({allowed_count} {suffix})"
        if self.allow_all_users:
            return "ALL (ALLOW_ALL_USERS=true)"
        return "NONE (set ALLOWED_USERS or ALLOW_ALL_USERS=true)"

    def is_telegram_user_allowed(self, user_id: int) -> bool:
        allowed = self.allowed_user_ids
        if allowed:
            return user_id in allowed
        return self.allow_all_users


settings = Settings()
