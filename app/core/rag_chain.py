"""
RAG Chain — HuggingFace Inference API LLM + ChromaDB retriever.

Uses LangChain's LCEL (LangChain Expression Language) pipe syntax:
    retriever | prompt | llm | parser

NOTE: HuggingFace Inference API routes through providers (novita, etc.)
that support "conversational" task only for chat models.
We use HuggingFaceEndpoint with task="conversational" and a supported model.
"""

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from app.core.config import settings
from app.core.logger import get_logger
from app.core.vector_store import VectorStoreManager
from langchain_openai import ChatOpenAI

logger = get_logger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a helpful, accurate assistant. Answer the user's question \
using ONLY the context provided below. If the context does not contain enough information \
to answer, say "I don't have enough information to answer that."

Do NOT fabricate facts. Be concise and direct.

Context:
{context}"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)


# ── LLM factory ──────────────────────────────────────────────────────────────

def build_llm():
    """
    HuggingFace Inference API via OpenAI-compatible router.
    This is the current (2025+) recommended way to call HF models.
    """
    return ChatOpenAI(
        model=settings.LLM_MODEL_ID,
        openai_api_key=settings.HUGGINGFACEHUB_API_TOKEN,
        openai_api_base="https://router.huggingface.co/v1",
        max_tokens=settings.LLM_MAX_NEW_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
    )
# ── Chain factory ─────────────────────────────────────────────────────────────

def build_rag_chain(filter: dict | None = None):
    """
    Build a full RAG chain.

    Returns a callable that accepts {"input": "<question>"} and returns
    {"answer": "...", "context": [<source docs>]}.
    """
    vs_manager = VectorStoreManager()
    retriever = vs_manager.get_retriever(filter=filter)
    llm = build_llm()

    combine_docs_chain = create_stuff_documents_chain(llm, RAG_PROMPT)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    logger.debug(
        "RAG chain built",
        llm=settings.LLM_MODEL_ID,
        search_type=settings.RETRIEVAL_SEARCH_TYPE,
        k=settings.RETRIEVAL_K,
    )
    return rag_chain
