from typing import cast
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.auth.oauth_token_manager import OAuthTokenManager
from onyx.chat.emitter import Emitter
from onyx.configs.app_configs import DISABLE_VECTOR_DB
from onyx.configs.model_configs import GEN_AI_TEMPERATURE
from onyx.context.search.models import BaseFilters
from onyx.db.models import Persona
from onyx.db.models import User
from onyx.db.oauth_config import get_oauth_config
from onyx.db.search_settings import get_current_search_settings
from onyx.db.tools import get_builtin_tool
from onyx.document_index.factory import get_default_document_index

from onyx.llm.interfaces import LLM
from onyx.llm.interfaces import LLMConfig
from onyx.tools.built_in_tools import get_built_in_tool_by_id
from onyx.tools.interface import Tool
from onyx.tools.models import DynamicSchemaInfo
from onyx.tools.models import SearchToolUsage
from onyx.tools.tool_implementations.custom.custom_tool import (
    build_custom_tools_from_openapi_schema_and_headers,
)
from onyx.tools.tool_implementations.file_reader.file_reader_tool import FileReaderTool

from onyx.tools.tool_implementations.memory.memory_tool import MemoryTool
from onyx.tools.tool_implementations.open_url.open_url_tool import (
    OpenURLTool,
)
from onyx.tools.tool_implementations.python.python_tool import PythonTool
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool_implementations.web_search.web_search_tool import (
    WebSearchTool,
)
from onyx.utils.headers import header_dict_to_header_list
from onyx.utils.logger import setup_logger

logger = setup_logger()


class SearchToolConfig(BaseModel):
    user_selected_filters: BaseFilters | None = None
    project_id: int | None = None
    persona_id: int | None = None
    bypass_acl: bool = False
    additional_context: str | None = None


class FileReaderToolConfig(BaseModel):
    # IDs from the ``user_file`` table (project / persona-attached files).
    user_file_ids: list[UUID] = []
    # IDs from the ``file_record`` table (chat-attached files).
    chat_file_ids: list[UUID] = []


class CustomToolConfig(BaseModel):
    chat_session_id: UUID | None = None
    message_id: int | None = None
    additional_headers: dict[str, str] | None = None





def construct_tools(
    persona: Persona,
    db_session: Session,
    emitter: Emitter,
    user: User,
    llm: LLM,
    search_tool_config: SearchToolConfig | None = None,
    custom_tool_config: CustomToolConfig | None = None,
    file_reader_tool_config: FileReaderToolConfig | None = None,
    allowed_tool_ids: list[int] | None = None,
    search_usage_forcing_setting: SearchToolUsage = SearchToolUsage.AUTO,
) -> dict[int, list[Tool]]:
    """Constructs tools based on persona configuration and available APIs.

    Will simply skip tools that are not allowed/available."""
    tool_dict: dict[int, list[Tool]] = {}

    # Log which tools are attached to the persona for debugging
    persona_tool_names = [t.name for t in persona.tools]
    logger.debug(
        f"Constructing tools for persona '{persona.name}' (id={persona.id}): {persona_tool_names}"
    )



    search_settings = get_current_search_settings(db_session)
    # This flow is for search so we do not get all indices.
    document_index = get_default_document_index(search_settings, None, db_session)

    added_search_tool = False
    for db_tool_model in persona.tools:
        # If allowed_tool_ids is specified, skip tools not in the allowed list
        if allowed_tool_ids is not None and db_tool_model.id not in allowed_tool_ids:
            continue

        if db_tool_model.in_code_tool_id:
            tool_cls = get_built_in_tool_by_id(db_tool_model.in_code_tool_id)

            try:
                tool_is_available = tool_cls.is_available(db_session)
            except Exception:
                logger.exception(
                    "Failed checking availability for tool %s", tool_cls.__name__
                )
                tool_is_available = False

            if not tool_is_available:
                logger.debug(
                    "Skipping tool %s because it is not available",
                    tool_cls.__name__,
                )
                continue

            # Handle Internal Search Tool
            if tool_cls.__name__ == SearchTool.__name__:
                added_search_tool = True
                if search_usage_forcing_setting == SearchToolUsage.DISABLED:
                    continue

                if not search_tool_config:
                    search_tool_config = SearchToolConfig()

                search_tool = SearchTool(
                    tool_id=db_tool_model.id,
                    emitter=emitter,
                    user=user,
                    persona=persona,
                    llm=llm,
                    document_index=document_index,
                    user_selected_filters=search_tool_config.user_selected_filters,
                    project_id=search_tool_config.project_id,
                    persona_id=search_tool_config.persona_id,
                    bypass_acl=search_tool_config.bypass_acl,
                )

                tool_dict[db_tool_model.id] = [search_tool]



            # Handle Web Search Tool
            elif tool_cls.__name__ == WebSearchTool.__name__:
                try:
                    tool_dict[db_tool_model.id] = [
                        WebSearchTool(tool_id=db_tool_model.id, emitter=emitter)
                    ]
                except ValueError as e:
                    logger.error(f"Failed to initialize Internet Search Tool: {e}")
                    raise ValueError(
                        "Internet search tool requires a search provider API key, please contact your Onyx admin to get it added!"
                    )

            # Handle Open URL Tool
            elif tool_cls.__name__ == OpenURLTool.__name__:
                try:
                    tool_dict[db_tool_model.id] = [
                        OpenURLTool(
                            tool_id=db_tool_model.id,
                            emitter=emitter,
                            document_index=document_index,
                            user=user,
                        )
                    ]
                except RuntimeError as e:
                    logger.error(f"Failed to initialize Open URL Tool: {e}")
                    raise ValueError(
                        "Open URL tool requires a web content provider, please contact your Onyx admin to get it configured!"
                    )

            # Handle Python/Code Interpreter Tool
            elif tool_cls.__name__ == PythonTool.__name__:
                tool_dict[db_tool_model.id] = [
                    PythonTool(tool_id=db_tool_model.id, emitter=emitter)
                ]

            # Handle File Reader Tool
            elif tool_cls.__name__ == FileReaderTool.__name__:
                cfg = file_reader_tool_config or FileReaderToolConfig()
                tool_dict[db_tool_model.id] = [
                    FileReaderTool(
                        tool_id=db_tool_model.id,
                        emitter=emitter,
                        user_file_ids=cfg.user_file_ids,
                        chat_file_ids=cfg.chat_file_ids,
                    )
                ]

            # Handle KG Tool
            # TODO: disabling for now because it's broken in the refactor
            # elif tool_cls.__name__ == KnowledgeGraphTool.__name__:

            #     # skip the knowledge graph tool if KG is not enabled/exposed
            #     kg_config = get_kg_config_settings()
            #     if not kg_config.KG_ENABLED or not kg_config.KG_EXPOSED:
            #         logger.debug("Knowledge Graph Tool is not enabled/exposed")
            #         continue

            #     if persona.name != TMP_DRALPHA_PERSONA_NAME:
            #         # TODO: remove this after the beta period
            #         raise ValueError(
            #             f"The Knowledge Graph Tool should only be used by the '{TMP_DRALPHA_PERSONA_NAME}' Agent."
            #         )
            #     tool_dict[db_tool_model.id] = [
            #         KnowledgeGraphTool(tool_id=db_tool_model.id)
            #     ]

        # Handle custom tools
        elif db_tool_model.openapi_schema:
            if not custom_tool_config:
                custom_tool_config = CustomToolConfig()

            # Determine which OAuth token to use
            oauth_token_for_tool = None

            # Priority 1: OAuth config (per-tool OAuth)
            if db_tool_model.oauth_config_id:
                if user.is_anonymous:
                    logger.warning(
                        f"Anonymous user cannot use OAuth tool {db_tool_model.id}"
                    )
                    continue
                oauth_config = get_oauth_config(
                    db_tool_model.oauth_config_id, db_session
                )
                if oauth_config:
                    token_manager = OAuthTokenManager(oauth_config, user.id, db_session)
                    oauth_token_for_tool = token_manager.get_valid_access_token()
                    if not oauth_token_for_tool:
                        logger.warning(
                            f"No valid OAuth token found for tool {db_tool_model.id} "
                            f"with OAuth config {db_tool_model.oauth_config_id}"
                        )

            elif db_tool_model.passthrough_auth:
                if user.is_anonymous:
                    logger.warning(
                        f"Anonymous user cannot use passthrough auth tool {db_tool_model.id}"
                    )
                    continue
                # user_oauth_token was removed since MCP is gone and this was the only user
                if user.oauth_accounts:
                    oauth_token_for_tool = user.oauth_accounts[0].access_token

            tool_dict[db_tool_model.id] = cast(
                list[Tool],
                build_custom_tools_from_openapi_schema_and_headers(
                    tool_id=db_tool_model.id,
                    openapi_schema=db_tool_model.openapi_schema,
                    emitter=emitter,
                    dynamic_schema_info=DynamicSchemaInfo(
                        chat_session_id=custom_tool_config.chat_session_id,
                        message_id=custom_tool_config.message_id,
                    ),
                    custom_headers=(db_tool_model.custom_headers or [])
                    + (
                        header_dict_to_header_list(
                            custom_tool_config.additional_headers or {}
                        )
                    ),
                    user_oauth_token=oauth_token_for_tool,
                ),
            )



    if (
        not added_search_tool
        and search_usage_forcing_setting == SearchToolUsage.ENABLED
        and not DISABLE_VECTOR_DB
    ):
        # Get the database tool model for SearchTool
        search_tool_db_model = get_builtin_tool(db_session, SearchTool)

        # Use the passed-in config if available, otherwise create a new one
        if not search_tool_config:
            search_tool_config = SearchToolConfig()

        search_tool = SearchTool(
            tool_id=search_tool_db_model.id,
            emitter=emitter,
            user=user,
            persona=persona,
            llm=llm,
            document_index=document_index,
            user_selected_filters=search_tool_config.user_selected_filters,
            project_id=search_tool_config.project_id,
            persona_id=search_tool_config.persona_id,
            bypass_acl=search_tool_config.bypass_acl,
        )

        tool_dict[search_tool_db_model.id] = [search_tool]

    # Always inject MemoryTool when the user has the memory tool enabled,
    # bypassing persona tool associations and allowed_tool_ids filtering
    if user.enable_memory_tool:
        try:
            memory_tool_db_model = get_builtin_tool(db_session, MemoryTool)
            memory_tool = MemoryTool(
                tool_id=memory_tool_db_model.id,
                emitter=emitter,
                llm=llm,
            )
            tool_dict[memory_tool_db_model.id] = [memory_tool]
        except RuntimeError:
            logger.warning(
                "MemoryTool not found in the database. Run the latest alembic migration to seed it."
            )

    tools: list[Tool] = []
    for tool_list in tool_dict.values():
        tools.extend(tool_list)

    return tool_dict
