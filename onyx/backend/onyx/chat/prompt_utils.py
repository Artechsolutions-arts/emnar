from typing import Any, Callable
from sqlalchemy.orm import Session
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType

def get_default_base_system_prompt(db_session: Session) -> str:
    """Returns the default base system prompt."""
    return "You are a helpful AI assistant."

def build_system_prompt(
    base_system_prompt: str,
    datetime_aware: bool,
    user_memory_context: Any,
    tools: list[Any],
    should_cite_documents: bool,
) -> str:
    """Builds the final system prompt with context."""
    prompt = base_system_prompt
    if user_memory_context:
        prompt += f"\n\nContext about the user:\n{user_memory_context}"
    return prompt

def build_reminder_message(reminder_message_text: str | None) -> ChatMessageSimple | None:
    """Builds a reminder system message to be added to the end of the conversation."""
    if not reminder_message_text:
        return None
    return ChatMessageSimple(
        message=reminder_message_text,
        token_count=len(reminder_message_text) // 4,
        message_type=MessageType.USER,
    )

def calculate_reserved_tokens(
    db_session: Session,
    persona_system_prompt: str,
    token_counter: Callable[[str], int],
    files: Any,
    user_memory_context: Any,
) -> int:
    """Calculates tokens reserved for system prompts and context so we don't overflow context window."""
    # Dummy implementation for now
    base = token_counter(persona_system_prompt) if persona_system_prompt else 0
    return base + 1024
