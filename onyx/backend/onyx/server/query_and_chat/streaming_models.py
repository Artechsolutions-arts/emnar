from enum import Enum
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Union

from pydantic import BaseModel
from pydantic import Field

from onyx.context.search.models import SearchDoc
from onyx.server.query_and_chat.placement import Placement


class StreamingType(Enum):
    """Enum defining all streaming packet types. This is the single source of truth for type strings."""

    SECTION_END = "section_end"
    STOP = "stop"
    TOP_LEVEL_BRANCHING = "top_level_branching"
    ERROR = "error"

    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    SEARCH_TOOL_START = "search_tool_start"
    SEARCH_TOOL_QUERIES_DELTA = "search_tool_queries_delta"
    SEARCH_TOOL_DOCUMENTS_DELTA = "search_tool_documents_delta"
    OPEN_URL_START = "open_url_start"
    OPEN_URL_URLS = "open_url_urls"
    OPEN_URL_DOCUMENTS = "open_url_documents"
    PYTHON_TOOL_START = "python_tool_start"
    PYTHON_TOOL_DELTA = "python_tool_delta"
    CUSTOM_TOOL_START = "custom_tool_start"
    CUSTOM_TOOL_ARGS = "custom_tool_args"
    CUSTOM_TOOL_DELTA = "custom_tool_delta"
    FILE_READER_START = "file_reader_start"
    FILE_READER_RESULT = "file_reader_result"
    REASONING_START = "reasoning_start"
    REASONING_DELTA = "reasoning_delta"
    REASONING_DONE = "reasoning_done"
    CITATION_INFO = "citation_info"
    TOOL_CALL_DEBUG = "tool_call_debug"
    TOOL_CALL_ARGUMENT_DELTA = "tool_call_argument_delta"

    MEMORY_TOOL_START = "memory_tool_start"
    MEMORY_TOOL_DELTA = "memory_tool_delta"
    MEMORY_TOOL_NO_ACCESS = "memory_tool_no_access"

    IMAGE_GENERATION_TOOL_START = "image_generation_tool_start"
    IMAGE_GENERATION_TOOL_HEARTBEAT = "image_generation_tool_heartbeat"
    IMAGE_GENERATION_FINAL = "image_generation_final"



class BaseObj(BaseModel):
    type: str = ""


################################################
# Control Packets
################################################
# This one isn't strictly necessary, remove in the future
class SectionEnd(BaseObj):
    type: Literal["section_end"] = StreamingType.SECTION_END.value


class OverallStop(BaseObj):
    type: Literal["stop"] = StreamingType.STOP.value
    stop_reason: str | None = None


class TopLevelBranching(BaseObj):
    # This class is used to give advanced heads up to the frontend that the top level flow is branching
    # This is used to avoid having the frontend render the first call then rerendering the other parallel branches
    type: Literal["top_level_branching"] = StreamingType.TOP_LEVEL_BRANCHING.value

    num_parallel_branches: int


class PacketException(BaseObj):
    type: Literal["error"] = StreamingType.ERROR.value

    exception: Exception = Field(exclude=True)
    model_config = {"arbitrary_types_allowed": True}


################################################
# Reasoning Packets
################################################
# Tells the frontend to display the reasoning block
class ReasoningStart(BaseObj):
    type: Literal["reasoning_start"] = StreamingType.REASONING_START.value


# The stream of tokens for the reasoning
class ReasoningDelta(BaseObj):
    type: Literal["reasoning_delta"] = StreamingType.REASONING_DELTA.value

    reasoning: str


class ReasoningDone(BaseObj):
    type: Literal["reasoning_done"] = StreamingType.REASONING_DONE.value


################################################
# Final Agent Response Packets
################################################
# Start of the final answer
class AgentResponseStart(BaseObj):
    type: Literal["message_start"] = StreamingType.MESSAGE_START.value

    final_documents: list[SearchDoc] | None = None
    pre_answer_processing_seconds: float | None = None


# The stream of tokens for the final response
# There is no end packet for this as the stream is over and a final OverallStop packet is emitted
class AgentResponseDelta(BaseObj):
    type: Literal["message_delta"] = StreamingType.MESSAGE_DELTA.value

    content: str


# Citation info for the sidebar and inline citations
class CitationInfo(BaseObj):
    type: Literal["citation_info"] = StreamingType.CITATION_INFO.value

    # The numerical number of the citation as provided by the LLM
    citation_number: int
    # The document id of the SearchDoc (same as the field stored in the DB)
    # This is the actual document id from the connector, not the int id
    document_id: str


class ToolCallDebug(BaseObj):
    type: Literal["tool_call_debug"] = StreamingType.TOOL_CALL_DEBUG.value

    tool_call_id: str
    tool_name: str
    tool_args: dict[str, Any]


################################################
# Tool Packets
################################################
# Search tool is called and the UI block needs to start
class SearchToolStart(BaseObj):
    type: Literal["search_tool_start"] = StreamingType.SEARCH_TOOL_START.value

    is_internet_search: bool = False


# Queries coming through as the LLM determines what to search
# Mostly for query expansions and advanced search strategies
class SearchToolQueriesDelta(BaseObj):
    type: Literal["search_tool_queries_delta"] = (
        StreamingType.SEARCH_TOOL_QUERIES_DELTA.value
    )

    queries: list[str]


# Documents coming through as the system knows what to add to the context
class SearchToolDocumentsDelta(BaseObj):
    type: Literal["search_tool_documents_delta"] = (
        StreamingType.SEARCH_TOOL_DOCUMENTS_DELTA.value
    )

    # This cannot be the SavedSearchDoc as this is yielded by the SearchTool directly
    # which does not save documents to the DB.
    documents: list[SearchDoc]


# OpenURL tool packets - 3-stage sequence
class OpenUrlStart(BaseObj):
    """Signal that OpenURL tool has started."""

    type: Literal["open_url_start"] = StreamingType.OPEN_URL_START.value


class OpenUrlUrls(BaseObj):
    """URLs to be fetched (sent before crawling begins)."""

    type: Literal["open_url_urls"] = StreamingType.OPEN_URL_URLS.value

    urls: list[str]


class OpenUrlDocuments(BaseObj):
    """Final documents after crawling completes."""

    type: Literal["open_url_documents"] = StreamingType.OPEN_URL_DOCUMENTS.value

    documents: list[SearchDoc]





class PythonToolStart(BaseObj):
    type: Literal["python_tool_start"] = StreamingType.PYTHON_TOOL_START.value
    code: str


class PythonToolDelta(BaseObj):
    type: Literal["python_tool_delta"] = StreamingType.PYTHON_TOOL_DELTA.value

    stdout: str = ""
    stderr: str = ""
    file_ids: list[str] = []


# Custom tool being called, first allocate a placeholder block for it on the UI
class CustomToolStart(BaseObj):
    type: Literal["custom_tool_start"] = StreamingType.CUSTOM_TOOL_START.value

    tool_name: str
    tool_id: int | None = None


class CustomToolArgs(BaseObj):
    type: Literal["custom_tool_args"] = StreamingType.CUSTOM_TOOL_ARGS.value

    tool_name: str
    tool_args: dict[str, Any]


class CustomToolErrorInfo(BaseModel):
    is_auth_error: bool = False
    status_code: int
    message: str


# The allowed streamed packets for a custom tool
class CustomToolDelta(BaseObj):
    type: Literal["custom_tool_delta"] = StreamingType.CUSTOM_TOOL_DELTA.value

    tool_name: str
    tool_id: int | None = None
    response_type: str
    # For non-file responses
    data: dict | list | str | int | float | bool | None = None
    # For file-based responses like image/csv
    file_ids: list[str] | None = None
    error: CustomToolErrorInfo | None = None


class ToolCallArgumentDelta(BaseObj):
    type: Literal["tool_call_argument_delta"] = (
        StreamingType.TOOL_CALL_ARGUMENT_DELTA.value
    )

    tool_type: str
    argument_deltas: dict[str, Any]


################################################
# File Reader Packets
################################################
class FileReaderStart(BaseObj):
    type: Literal["file_reader_start"] = StreamingType.FILE_READER_START.value


class FileReaderResult(BaseObj):
    type: Literal["file_reader_result"] = StreamingType.FILE_READER_RESULT.value

    file_name: str
    file_id: str
    start_char: int
    end_char: int
    total_chars: int
    # Short previews of the retrieved text for the collapsed/expanded UI
    preview_start: str = ""
    preview_end: str = ""


# Memory Tool Packets
################################################
class MemoryToolStart(BaseObj):
    type: Literal["memory_tool_start"] = StreamingType.MEMORY_TOOL_START.value


class MemoryToolDelta(BaseObj):
    type: Literal["memory_tool_delta"] = StreamingType.MEMORY_TOOL_DELTA.value

    memory_text: str
    operation: Literal["add", "update"]
    memory_id: int | None = None
    index: int | None = None


class MemoryToolNoAccess(BaseObj):
    type: Literal["memory_tool_no_access"] = StreamingType.MEMORY_TOOL_NO_ACCESS.value


class GeneratedImage(BaseModel):
    file_id: str
    url: str
    revised_prompt: str
    shape: str


class ImageGenerationToolStart(BaseObj):
    type: Literal["image_generation_tool_start"] = StreamingType.IMAGE_GENERATION_TOOL_START.value


class ImageGenerationToolHeartbeat(BaseObj):
    type: Literal["image_generation_tool_heartbeat"] = StreamingType.IMAGE_GENERATION_TOOL_HEARTBEAT.value


class ImageGenerationFinal(BaseObj):
    type: Literal["image_generation_final"] = StreamingType.IMAGE_GENERATION_FINAL.value
    images: list[GeneratedImage]

################################################
# Packet Object
################################################
# Discriminated union of all possible packet object types
PacketObj = Union[
    # Control Packets
    OverallStop,
    SectionEnd,
    TopLevelBranching,
    PacketException,
    # Agent Response Packets
    AgentResponseStart,
    AgentResponseDelta,
    # Tool Packets
    SearchToolStart,
    SearchToolQueriesDelta,
    SearchToolDocumentsDelta,
    OpenUrlStart,
    OpenUrlUrls,
    OpenUrlDocuments,
    PythonToolStart,
    PythonToolDelta,
    CustomToolStart,
    CustomToolArgs,
    CustomToolDelta,
    FileReaderStart,
    FileReaderResult,
    MemoryToolStart,
    MemoryToolDelta,
    MemoryToolNoAccess,
    ImageGenerationToolStart,
    ImageGenerationToolHeartbeat,
    ImageGenerationFinal,
    # Reasoning Packets
    ReasoningStart,
    ReasoningDelta,
    ReasoningDone,
    # Citation Packets
    CitationInfo,
    ToolCallDebug,
    ToolCallArgumentDelta,
]


class Packet(BaseModel):
    placement: Placement

    obj: Annotated[PacketObj, Field(discriminator="type")]
