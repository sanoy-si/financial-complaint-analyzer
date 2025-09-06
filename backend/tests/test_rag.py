from app.rag.chunking import chunk_text
from app.rag.pipeline import InMemoryRetriever, RagPipeline, RetrievedChunk
from app.rag.providers import HashEmbedder, MockLLM


def test_chunking_splits_and_overlaps():
    text = "\n\n".join(f"Paragraph number {i} with some filler words." for i in range(20))
    chunks = chunk_text(text, chunk_size=80, overlap=20)
    assert len(chunks) > 1
    assert all(c.content for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunking_empty():
    assert chunk_text("") == []


def test_pipeline_retrieves_and_answers():
    embedder = HashEmbedder(dim=128)
    docs = [
        "Refunds are processed within 5 business days.",
        "Our office is open Monday to Friday.",
        "Shipping is free over fifty dollars.",
    ]
    embeddings = embedder.embed(docs)
    chunks = [RetrievedChunk(content=d, score=0.0, document_id=str(i)) for i, d in enumerate(docs)]
    retriever = InMemoryRetriever(chunks=chunks, embeddings=embeddings)

    pipe = RagPipeline(embedder=embedder, llm=MockLLM(), top_k=2)
    # Query shares the distinctive tokens "refunds"/"business"/"days" with the
    # first doc, so the hash embedder ranks it top.
    ans = pipe.answer("refunds processed business days", retriever)

    assert len(ans.sources) == 2
    # the refunds doc should be the top source
    assert "Refunds" in ans.sources[0].content
    assert "Refunds are processed within 5 business days" in ans.text
