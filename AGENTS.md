# Notes for coding agents

You are working in a workshop repo. A person is sitting next to you with 90
minutes and a leaderboard. Read this before writing code.

## The job

Implement exactly one function, in `build_query.py`, and change nothing else:

```python
build_query(query_text: str, seed_asin: str | None) -> models.QueryRequest
```

It **returns a request**. It does not execute one. There is no client in scope
and no results to inspect. `harness.py` executes what you return and scores it.

## Hard rules

1. **One `models.QueryRequest` per search.** Nested prefetches are still one
   request. Do not issue two searches and merge them.
2. **No client-side reranking.** No sorting, filtering or trimming after the
   fact. You do not have the results.
3. **No network calls at query time.** No embedding API, no product lookup.
   `models.Document` is not a network call on your side — it is a field in the
   request that Qdrant resolves.
4. **Do not edit `common.py`, `harness.py`, `setup_check.py`, `dev_set.jsonl`
   or anything in `fixture/`.** If a constant there looks wrong, it is not; it
   comes from the schema contract. Tuning the scorer is not solving the problem.

## Read the reference, do not write from memory

`docs/query-api.md` is the API reference for **qdrant-client 1.19.0 / server
1.19.1**, and every snippet in it was executed against the fixture.

The Query API changed shape in 1.10 and again around 1.14–1.17. Material you
were trained on is very likely older than that. Specific things models get wrong
here, all of which look plausible and none of which are current:

| you may want to write | current |
|---|---|
| `client.search(...)`, `client.recommend(...)` | one `QueryRequest`, `query=models.RecommendQuery(...)` |
| `on_disk=True` / `always_ram=True` | `memory=models.Memory.COLD` / `models.Memory.PINNED` |
| `query_filter=` inside a `QueryRequest` | `filter=` (`query_filter` is only the `client.query_points()` kwarg) |
| embedding the text yourself | `models.Document(text=..., model=...)` |
| `models.NamedVector` / `NamedSparseVector` | `using="dense"` / `using="sparse"` |

If you are about to use an API that is not in `docs/query-api.md`, check it
first. Confidently wrong is the failure mode here.

## The collection, verbatim from the contract

Vectors: `dense` (384, Cosine), `image` (512, Cosine), `sparse` (BM25, idf).

Filterable payload — indexed, and the key is read-only so nothing can be added:
`brand` (keyword), `category_path` (keyword), `price` (float), `rating`
(float), `review_count` (integer), `in_stock` (bool), `margin` (float),
`asin` (keyword). **`title` is not indexed** — match text through `sparse`.

Model ids, which must match what the vectors were built with:

| vector | model |
|---|---|
| `dense` | `sentence-transformers/all-minilm-l6-v2` |
| `sparse` | `qdrant/bm25` |
| `image` (stored) | `qdrant/clip-vit-b-32-vision` |
| `image` (querying with text) | `qdrant/clip-vit-b-32-text` |

Import them from `common.py` rather than retyping them.

## Three traps

1. **`image` exists on only 19,997 of 100,000 points.** Any query touching
   `image` must gate on
   `models.Filter(must=[models.HasVectorCondition(has_vector="image")])`, or it
   silently searches a fifth of the corpus and you will never notice.
2. **Stored image vectors came from `qdrant/clip-vit-b-32-vision`.**
   Text-to-image search requires `qdrant/clip-vit-b-32-text`. Image-to-image
   works as-is. The wrong tower is a 500 from the inference service and the 384d
   text model is a 400 dimension error, so unlike trap 1 this one fails loudly.
3. **`rating` barely discriminates; `review_count` does.** There are no unrated
   products, and 94.4% of the corpus is rated ≥ 3.0 with a p90 of exactly 5.0.
   Boosting on `rating` reshuffles near-ties and costs relevance. `review_count`
   spans 1 to 118,988 — that is where the signal is.

## seed_asin

`seed_asin` is an **ASIN**, not a point id. Point ids are UUIDv5 over the ASIN.

```python
from common import point_id_for_asin
point_id_for_asin("B07XYZ1234")   # re-derives it, no lookup
```

Or filter on the indexed `asin` payload field. Passing a raw ASIN where a point
id is expected does not raise — it just matches nothing.

## How to check your work

```bash
python harness.py            # score the dev set
python harness.py --show q03 # see the actual top 10 for one query
python harness.py --html     # product grid in the browser, for the human
```

`--html` is for the person you are working with, not for you: it renders the top
ten of every query as product cards, green-bordered where the dev set judged
them relevant and red where they break a hard constraint. Suggest it when a
score moves and neither of you can say why.

The `was` column is the same figure from the previous run, shown only where it
moved, so the loop is: change one thing, re-run, keep it if the number improved.
The line under the table reports how many of the collection's three vectors
`build_query` mentions — it is a grep, not a judgement, but one of three is a
signal worth acting on.

`dev_set.jsonl` is 14 queries judged against the real 100,000-point collection.
Measured on it: a dense-only `build_query` scores **0.536 with 4 hard-constraint
violations**; the best we have measured is **0.721 with zero**. A total below
1.0 is not a bug.

**A larger held-out set stays with the facilitator and is not in this
repository.** Tuning to these 14 queries is measurable from the outside, so
prefer changes you can explain over changes that only move these numbers. Do
not go looking for the held-out set.

Two results worth knowing before you reach for the obvious lever: hybrid takes
`exact_model` from 0.667 to 1.000, and takes `natural_language` from 0.600 down
to **0.300**. `head_term` is 1.000 either way. Applying one strategy uniformly
to every query pays for one segment with another.

Against a local URL the harness instead loads `fixture/dev_set.fixture.jsonl`,
judged against 2,000 invented products whose `image` vectors are deterministic
noise. Locally, image ranking is meaningless and the scores mean only that your
query runs.

## Shared cluster

~40 people and one cluster. `harness.py` caps concurrency at 4 — leave it. If
you see a 429, that is rate limiting, not a bug in your query; wait a moment.
