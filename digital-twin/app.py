import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

from modules.ingester import (
    Ingester,
    IngestOptions,
    ChunkingOptions,
    VectorStoreOptions,
    EmbeddingOptions,
)

load_dotenv(override=True)
logger = logging.getLogger(__name__)
client = OpenAI()


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

NAME = "David Inyang-Etoh"
KNOWLEDGE_DIR = Path("me/knowledge")
VECTOR_DB_DIR = Path("db/digital_twin_db")
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

ingester = Ingester(
    IngestOptions(
        knowledge_base_path=str(KNOWLEDGE_DIR),
        chunking=ChunkingOptions(
            chunk_size=1000, chunk_overlap=150, min_chunk_chars=60
        ),
        vector_store=VectorStoreOptions(
            persist_directory=str(VECTOR_DB_DIR),
            delete_existing_collection=False,
        ),
        embedding=EmbeddingOptions(model=EMBED_MODEL),
    )
)

_force_ingest = _truthy_env("DIGITAL_TWIN_FORCE_INGEST")
if _force_ingest:
    logger.info("DIGITAL_TWIN_FORCE_INGEST set — running ingest")
    _, _ = ingester.ingest()
elif not ingester.vector_store_has_documents():
    logger.info("Vector store empty or unreadable — running ingest")
    _, _ = ingester.ingest()
else:
    logger.info("Skipping ingest: vector DB already has data (%s)", VECTOR_DB_DIR)

# ── Prompts ────────────────────────────────────────────────────────────────────

BASE_PROMPT = f"""You are Mr Dee — the digital twin of {NAME}, a Senior Lead Software Engineer specialising in TypeScript, Node.js, AWS, and distributed systems.

Speak in first person, as David. Be concise and direct. Do not pad answers with filler phrases or unsolicited advice.

## Strict grounding rules
- Answer ONLY from the context provided in the user message. Do not infer, extrapolate, or invent details not explicitly stated.
- If a specific detail (name, date, number, company) is not in the context, say: "I don't have that in my notes right now."
- Never fabricate company names, metrics, team sizes, or timelines. If the context has them, use them exactly as stated.
- ALWAYS name companies, projects, roles, and technologies by their exact name when they appear anywhere in the context — even if the user did not ask for the name specifically. For example, if the context mentions "Jiggle" as a company, include it in the answer.
- You may synthesise across multiple context chunks, but every claim must be traceable to the context.
- Never claim to "not have access" to your own experiences — if it's in context, it's your lived experience. If it's not in context, say you don't have that detail handy."""

_EXPAND_SYSTEM = (
    "You are a search query expansion assistant for a personal knowledge base about "
    f"{NAME}. Given a conversation history and the latest user question, output a JSON "
    'object with one key: "queries" — a list of 3 distinct search queries that will '
    "help retrieve the most relevant passages. "
    "The first query should be a standalone, coreference-resolved rewrite of the user's "
    "question (resolve any pronouns or vague references using the conversation history). "
    "The other two should be semantically diverse alternative phrasings. "
    "Output ONLY valid JSON, no markdown, no commentary."
)


# ── Query expansion ────────────────────────────────────────────────────────────

def _extract_history_text(history: list) -> str:
    """Normalise Gradio history to a plain text string.
    Handles both dict format [{role, content}] and legacy list-of-lists [[user, asst]]."""
    lines = []
    for turn in history[-4:]:
        if isinstance(turn, dict):
            lines.append(
                f"{turn.get('role', 'unknown').capitalize()}: {turn.get('content', '')}")
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            if turn[0]:
                lines.append(f"User: {turn[0]}")
            if turn[1]:
                lines.append(f"Assistant: {turn[1]}")
    return "\n".join(lines)


def _last_assistant_message(history: list) -> str | None:
    """Extract the last assistant message from history in any Gradio format."""
    for turn in reversed(history):
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            return turn.get("content", "")
        if isinstance(turn, (list, tuple)) and len(turn) == 2 and turn[1]:
            return turn[1]
    return None


def expand_query(message: str, history: list) -> list[str]:
    """Use a lightweight LLM call to produce 3 search queries from the user message
    and conversation history. Resolves coreferences (e.g. 'the company' → 'Jiggle').
    Always returns at least 2 queries — includig a coreference-anchored fallback."""
    history_text = _extract_history_text(history) if history else ""
    last_asst = _last_assistant_message(history) if history else None

    # Always add a cheap coreference-anchored query as a guaranteed fallback:
    # prepending the last assistant reply to the current question anchors any
    # pronouns/references even if the LLM expansion call fails.
    guaranteed_fallback = (
        f"{last_asst[:400]} {message}" if last_asst else message
    )

    user_content = (
        (f"Conversation so far:\n{history_text}\n\n") if history_text else ""
    ) + f"Latest question: {message}"

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": _EXPAND_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=200,
        )
        data = json.loads(resp.choices[0].message.content)
        queries: list[str] = data.get("queries", [])
        # Always include original message + guaranteed coreference fallback
        if message not in queries:
            queries.insert(0, message)
        if guaranteed_fallback not in queries:
            queries.append(guaranteed_fallback)
        return queries[:5]
    except Exception:
        return [message, guaranteed_fallback]


# ── Retrieval with deduplication ───────────────────────────────────────────────

def retrieve_context(message: str, history: list, k: int = 6) -> str:
    """Expand the query, run similarity search for each variant, and deduplicate."""
    queries = expand_query(message, history)

    seen: set[str] = set()
    chunks = []
    for q in queries:
        for chunk in ingester.similarity_search(q, k=k):
            key = chunk.page_content.strip()
            if key not in seen:
                seen.add(key)
                chunks.append(chunk)

    # print(f"[RAG] Retrieved {len(chunks)} unique chunks:")
    # for i, c in enumerate(chunks):
    #     src = Path(c.metadata.get('source', '')).name
    #     preview = c.page_content[:1000].replace('\n', ' ')
    #     print("="*100)
    #     print(f"  [{i+1}] {src}: {preview}...")

    if not chunks:
        return ""

    return "\n\n---\n\n".join(
        f"[{Path(c.metadata.get('source', '')).name}]\n{c.page_content}"
        for c in chunks
    )


# ── Chat ───────────────────────────────────────────────────────────────────────

def chat(message: str, history: list[dict]) -> str:
    context = retrieve_context(message, history)

    # RAG prompt pattern: system = persona/rules, user turn = question + grounding data
    if context:
        grounded_message = (
            f"{message}\n\n"
            "---\n"
            'Answer using ONLY the context below. '
            'If the answer isn\'t there, say "I don\'t have that in my notes right now."\n\n'
            f"## Context\n\n{context}"
        )
    else:
        grounded_message = (
            f"{message}\n\n"
            "---\n"
            "No relevant context was retrieved for this question. "
            "If you cannot answer confidently, say so."
        )

    msgs = (
        [{"role": "system", "content": BASE_PROMPT}]
        + [{"role": h["role"], "content": h["content"]} for h in history]
        + [{"role": "user", "content": grounded_message}]
    )
    return (
        client.chat.completions.create(model=CHAT_MODEL, messages=msgs)
        .choices[0]
        .message.content
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

CHAT_DESCRIPTION = (
    f"Chat with Mr Dee — a RAG-powered digital twin of {NAME}. "
    "Ask questions grounded in his knowledge base."
)

WELCOME_MESSAGE = (
    f"👋 Hi! I'm Mr Dee, the digital twin of **{NAME}** — "
    "software engineer, AI builder, and technical writer.\n\n"
    "I can answer questions about my background, projects, professional experience, "
    "skills, and the things I write and speak about.\n\n"
    "Feel free to ask me anything!"
)

demo = gr.ChatInterface(
    fn=chat,
    title=f"🤖 {NAME} — Digital Twin",
    description=CHAT_DESCRIPTION,
    chatbot=gr.Chatbot(
        value=[{"role": "assistant", "content": WELCOME_MESSAGE}],

    ),
    examples=[
        "Tell me about your background",
        "What are you building?",
        "What's your view on AI?",
        "What technologies do you specialize in?",
    ],
)

if __name__ == "__main__":
    port = os.environ.get("PORT")
    if port:
        demo.launch(server_name="0.0.0.0", server_port=int(port))
    else:
        demo.launch()
