import re
import uuid
import zlib
from html.parser import HTMLParser

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Modifier,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from config import OLLAMA_HOST, QDRANT_HOST, QDRANT_PORT
from edgar_client import EdgarClient

EMBED_MODEL = "nomic-embed-text"
COLLECTION = "filings"
VECTOR_SIZE = 768
CHUNK_SIZE = 800
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"

BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "tr", "table", "section", "article", "dt", "dd", "br",
}
SKIP_TAGS = {"script", "style", "ix:hidden", "ix:header"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self.skip > 0:
            self.skip -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip == 0:
            self.parts.append(data)


def extract_paragraphs(raw):
    extractor = _TextExtractor()
    extractor.feed(raw)
    text = "".join(extractor.parts)
    text = re.sub(r"[ \t]+", " ", text)
    paragraphs = [p.strip() for p in text.split("\n")]
    return [p for p in paragraphs if len(p) > 1]


def _split_sentences(paragraph):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", paragraph)
    return [s.strip() for s in parts if s.strip()]


def pack_chunks(paragraphs, chunk_size=CHUNK_SIZE):
    units = []
    for paragraph in paragraphs:
        units.extend(_split_sentences(paragraph))
    chunks = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > chunk_size:
            chunks.append(current)
            current = unit
        else:
            current = f"{current} {unit}".strip() if current else unit
    if current:
        chunks.append(current)
    if len(chunks) >= 2 and len(chunks[-1]) < 80:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()
    return chunks


def embed(texts):
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _term_counts(text):
    counts = {}
    for token in TOKEN_RE.findall(text.lower()):
        idx = zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF
        counts[idx] = counts.get(idx, 0) + 1
    return counts


def to_sparse_vector(text):
    counts = _term_counts(text)
    indices = sorted(counts)
    return SparseVector(indices=indices, values=[float(counts[i]) for i in indices])


def archive_url(cik, accession_number, filename):
    cik_int = str(int(cik))
    accession = accession_number.replace("-", "")
    return ARCHIVE_URL.format(cik=cik_int, accession=accession, document=filename)


def filing_index_url(cik, accession_number):
    cik_int = str(int(cik))
    accession = accession_number.replace("-", "")
    return INDEX_URL.format(cik=cik_int, accession=accession)


def get_qdrant():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client):
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={DENSE_VECTOR: VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF)},
        )


def latest_filing(submissions, form_type):
    recent = submissions["filings"]["recent"]
    forms = recent["form"]
    for idx, form in enumerate(forms):
        if form == form_type:
            return {
                "accession_number": recent["accessionNumber"][idx],
                "primary_document": recent["primaryDocument"][idx],
                "filing_date": recent["filingDate"][idx],
                "form_type": form,
            }
    raise LookupError(f"no {form_type} filing found in submissions")


def content_documents(edgar, cik, accession_number):
    index = edgar.get_json(filing_index_url(cik, accession_number))
    documents = []
    for item in index["directory"]["item"]:
        name = item["name"]
        lower = name.lower()
        if not lower.endswith((".htm", ".html")):
            continue
        if re.match(r"^r\d+\.", lower):
            continue
        if lower.endswith("_htm.xml"):
            continue
        if "index" in lower:
            continue
        documents.append(name)
    return documents


def ingest_accession(edgar, qdrant, cik, ticker, accession_number, primary, form_type):
    documents = content_documents(edgar, cik, accession_number)
    if primary not in documents:
        documents.insert(0, primary)

    points = []
    total_chunks = 0
    for document in documents:
        raw = edgar.get_text(archive_url(cik, accession_number, document))
        paragraphs = extract_paragraphs(raw)
        chunks = pack_chunks(paragraphs)
        if not chunks:
            continue
        vectors = embed(chunks)
        doc_type = "primary" if document == primary else "exhibit"
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{accession_number}:{document}:{i}",
                )
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR: vector,
                        SPARSE_VECTOR: to_sparse_vector(chunk),
                    },
                    payload={
                        "accession_number": accession_number,
                        "ticker": ticker,
                        "form_type": form_type,
                        "document": document,
                        "doc_type": doc_type,
                        "chunk_index": i,
                        "text": chunk,
                    },
                )
            )
            total_chunks += 1

    if points:
        qdrant.upsert(collection_name=COLLECTION, points=points, wait=True)

    return {
        "ticker": ticker,
        "accession_number": accession_number,
        "form_type": form_type,
        "documents": len(documents),
        "chunks": total_chunks,
    }


def ingest_filing(edgar, qdrant, ticker, form_type="8-K"):
    cik = edgar.resolve_ticker(ticker)
    submissions = edgar.get_submissions(cik)
    filing = latest_filing(submissions, form_type)
    return ingest_accession(
        edgar,
        qdrant,
        cik,
        ticker,
        filing["accession_number"],
        filing["primary_document"],
        form_type,
    )


def _ticker_filter(ticker):
    if not ticker:
        return None
    return Filter(
        must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))]
    )


def _hit_to_dict(hit):
    return {
        "score": hit.score,
        "accession_number": hit.payload["accession_number"],
        "ticker": hit.payload["ticker"],
        "form_type": hit.payload["form_type"],
        "document": hit.payload.get("document"),
        "doc_type": hit.payload.get("doc_type"),
        "chunk_index": hit.payload["chunk_index"],
        "text": hit.payload["text"],
    }


def retrieve(qdrant, query, ticker=None, top_k=5):
    query_vector = embed([query])[0]
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=(DENSE_VECTOR, query_vector),
        limit=top_k,
        query_filter=_ticker_filter(ticker),
    )
    return [_hit_to_dict(hit) for hit in results]


def hybrid_retrieve(qdrant, query, ticker=None, top_k=5, candidates_per=20):
    query_vector = embed([query])[0]
    result = qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=query_vector, using=DENSE_VECTOR, limit=candidates_per),
            Prefetch(query=to_sparse_vector(query), using=SPARSE_VECTOR, limit=candidates_per),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=_ticker_filter(ticker),
        limit=top_k,
        with_payload=True,
    )
    return [_hit_to_dict(hit) for hit in result.points]


if __name__ == "__main__":
    edgar = EdgarClient()
    qdrant = get_qdrant()
    ensure_collection(qdrant)
    result = ingest_filing(edgar, qdrant, "AAPL", form_type="8-K")
    print("ingested:", result)
    for query in ("financial results and revenue", "material agreement"):
        print(f"\n--- query: {query!r} ---")
        dense = retrieve(qdrant, query, ticker="AAPL", top_k=3)
        hybrid = hybrid_retrieve(qdrant, query, ticker="AAPL", top_k=3)
        print("[dense]")
        for item in dense:
            print(f"  {item['score']:.4f} [{item['doc_type']}]:{item['chunk_index']} :: {item['text'][:110]}")
        print("[hybrid RRF]")
        for item in hybrid:
            print(f"  {item['score']:.4f} [{item['doc_type']}]:{item['chunk_index']} :: {item['text'][:110]}")
    edgar.close()
