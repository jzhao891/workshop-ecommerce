"""THE ONLY FILE YOU EDIT.

Implement build_query(). You return a request. You never execute it, you never
see the results. The harness executes it and scores what comes back.

    build_query(query_text: str, seed_asin: str | None) -> models.QueryRequest

Rules (the harness enforces the first one, the other two are on your honour):
  1. ONE request per search. One models.QueryRequest. Prefetch as much as you
     like inside it -- that is still one request.
  2. No client-side reranking. Do not reorder, filter or trim results after the
     fact. You do not have the results anyway.
  3. No network calls at query time. No embedding API, no product lookup, no
     scraping. models.Document is not a network call on your side -- it is a
     field in the request that Qdrant resolves.

seed_asin is an ASIN (e.g. "B07XYZ1234"), NOT a point id. Point ids are
UUIDv5 over the ASIN. Two ways to use one, both fine:

    from common import point_id_for_asin
    pid = point_id_for_asin(seed_asin)      # re-derive it, no lookup
    models.RecommendInput(positive=[pid])

    # or filter on the indexed `asin` payload field
    models.FieldCondition(key="asin", match=models.MatchValue(value=seed_asin))

Run `python harness.py` to score yourself. Read docs/query-api.md for what is
available, and AGENTS.md if you brought a coding agent.
"""
from qdrant_client import models

from common import DENSE_MODEL

TOP_K = 10  # the harness scores precision@10; returning fewer can only hurt


def build_query(query_text: str, seed_asin: str | None = None) -> models.QueryRequest:
    """Naive dense-only baseline. It runs. It scores badly. That is the point.

    What it ignores, all of which is on the table:
      * `sparse` (BM25) -- so "Kestrel TR-450" is a coin flip
      * seed_asin entirely
      * every hard constraint in the query text ("under $80", "in stock")
      * SearchParams, oversampling, rescore
      * FormulaQuery, decay, Recommend, the `image` vector

    Replace the body. Keep the signature.
    """
    return models.QueryRequest(
        query=models.Document(text=query_text, model=DENSE_MODEL),
        using="dense",
        limit=TOP_K,
        with_payload=True,
    )
