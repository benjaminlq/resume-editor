from llama_index.core.schema import Document, MetadataMode
from llama_index.core.prompts import ChatMessage, MessageRole
from typing import List, Tuple

def combine_documents(
    pages: Document
) -> str:
    combined_page_content = ""
    for page in pages:
        combined_page_content += page.get_content(metadata_mode = MetadataMode.LLM)
    return combined_page_content

def convert_llamaindex_messages_to_gradio(
    li_messages: List[ChatMessage]
) -> List[Tuple[str, str]]:
    if li_messages[0].role == MessageRole.SYSTEM:
        li_messages = li_messages[1:]
        
    user_messages = li_messages[::2]
    assistant_messages = li_messages[1::2]
    
    assert len(user_messages) == len(assistant_messages)
    messages = []
    for user_msg, assistant_msg in zip(user_messages, assistant_messages):
        messages.append((user_msg.content, assistant_msg.content))
    return messages