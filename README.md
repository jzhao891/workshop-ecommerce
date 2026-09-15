# E-commerce search workshop

You have one function to write and 90 minutes. Everything else in this repo
exists to score it.

The corpus is 100,000 Amazon clothing/shoes/jewelry products in Qdrant, with a
dense vector, a BM25 sparse vector, and a CLIP image vector on a fifth of them.
Your job is to turn a shopper's query into the best single Qdrant request you
can.

No vector search experience assumed. If you brought a coding agent, point it at
`AGENTS.md` first.

---

## Before the workshop

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # paste in the URL and read-only key you were emailed
python setup_check.py
```

`setup_check.py` prints one green line when you are ready. If it fails it names
the check and the fix — that is the only file you need to read to get unstuck.
It also confirms your key is read-only (without writing anything) and tells you
how many points carry an `image` vector, which is trap 1 in numbers.
**Do this before the session, not during it.**

## The contract

You implement exactly one function, in `build_query.py`:

```python
def build_query(query_text: str, seed_asin: str | None) -> models.QueryRequest:
    ...
```

You **return a request**. You never execute it and you never see results. The
harness executes it and scores what comes back. `build_query.py` ships with a
naive dense-only version that runs and scores badly; replace the body, keep the
signature.

### Rules

1. **One request per search.** One `models.QueryRequest`. Prefetch as much as
   you like inside it — that is still one request.
2. **No client-side reranking.** No reordering or filtering after the fact.
3. **No network calls at query time.** `models.Document` is not a network call
   on your side; it is a field in the request that Qdrant resolves.

Edit `build_query.py`. Leave `common.py`, `harness.py`, `setup_check.py`,
`dev_set.jsonl` and `fixture/` alone.

## ASINs are not point IDs

`seed_asin` is an ASIN like `B07XYZ1234`. Point IDs are **UUIDv5 over the
ASIN**, namespace `6f1d6b1e-0b3f-5c2a-9b77-4a1c7e9d2f01`. Two ways to use one:

```python
from common import point_id_for_asin
pid = point_id_for_asin(seed_asin)        # re-derive it, no lookup needed
models.RecommendInput(positive=[pid])
```

```python
# or filter on the indexed `asin` payload field
models.FieldCondition(key="asin", match=models.MatchValue(value=seed_asin))
```

Passing a raw ASIN where a point ID is expected does not raise an error. It just
quietly matches nothing.

## What you have to work with

The collection — names, dimensions and indexes are fixed, and your key is
read-only so you cannot add to them:

| vector | dim | distance | built from |
|---|---|---|---|
| `dense` | 384 | Cosine | `sentence-transformers/all-minilm-l6-v2` over `"{title}. {category_path}"` |
| `image` | 512 | Cosine | `qdrant/clip-vit-b-32-vision`, **present on only 19,997 of 100,000 points** |
| `sparse` | — | — | `qdrant/bm25`, `modifier=idf` |

Filterable payload: `brand` (keyword) · `category_path` (keyword) · `price`
(float) · `rating` (float) · `review_count` (integer) · `in_stock` (bool) ·
`margin` (float) · `asin` (keyword). `title` is **not** indexed — match text
through `sparse`.

### Capability menu

Roughly in order of how much they usually buy you. Full reference with runnable
examples in [`docs/query-api.md`](docs/query-api.md).

| lever | what it does |
|---|---|
| **Hybrid** — `prefetch` + `FusionQuery(RRF)` | dense and BM25 in one request. The biggest single win. `DBSF` if you want score-based fusion instead of rank-based. |
| **Filters** — `Filter(must/should/must_not)`, `Range` | hard constraints. Put them on the prefetches *and* the outer request. |
| **`SearchParams`** — `QuantizationSearchParams(oversampling, rescore)` | vectors are int8-quantized; oversample and rescore against the originals to win back precision. |
| **`FormulaQuery`** | re-score a prefetched list using payload — margin, reviews, stock. |
| **Decay** — `lin` / `exp` / `gauss` | soft preference along a number (price near $60) instead of a hard cut. |
| **`Recommend`** | more-like-this from `seed_asin`. Strategies: `average_vector`, `best_score`, `sum_scores`. |
| **`image`** + `HasVectorCondition` | visual similarity, on the 20% that have it. |
| **Weighted RRF** — `RrfQuery(rrf=Rrf(weights=...))` | make one retriever count more than the other. |
| **MMR** — `NearestQuery(nearest=..., mmr=Mmr(diversity=...))` | trade a little relevance for a less repetitive top 10. The direct lever on brand diversity. |

### Three traps

1. **`image` exists on only 19,997 of 100,000 points.** Any query touching
   `image` must gate on
   `models.Filter(must=[models.HasVectorCondition(has_vector="image")])` or it
   silently searches a fifth of the corpus and you will never notice.
2. **Stored image vectors came from `qdrant/clip-vit-b-32-vision`.**
   Text-to-image search requires `qdrant/clip-vit-b-32-text`. Image-to-image
   works as-is. This one fails loudly — a 500 from inference, or a 400
   dimension error — unlike trap 1, which is silent.
3. **`rating` barely discriminates; `review_count` does.** There are no unrated
   products, and 94.4% of the corpus is rated ≥ 3.0 with a p90 of exactly 5.0.
   Boosting on `rating` reshuffles near-ties and costs relevance. `review_count`
   spans 1 to 118,988 — that is where the signal is.

## Scoring

The dev set is 14 queries judged against the real 100,000-point collection.
**A dense-only `build_query` scores 0.536 and takes 4 hard-constraint
violations. The best we have measured is 0.721, with zero violations.** That is
a straightforward hybrid, and it is not a ceiling anyone should treat as one —
`constrained` sits at 0.55 and `natural_language` at 0.30 in that run, so there
is real room above it. Do not read a total below 1.0 as a bug.

A larger held-out set stays with the facilitator. The 14 here teach you the
rubric; the held-out set is the check on whether a change helps in general or
only on the queries you can see. Tuning to these 14 is visible from the outside.

```bash
python harness.py              # run the dev set, print the scorecard
python harness.py --show q03   # dump the actual top 10 for one query
python harness.py --html       # the same run as a product grid in your browser
```

`--html` is worth your time. Product search is visual: a sandal returned for a
boot query is obvious in a picture and invisible in a list of titles. The page
writes to `.workshop/results.html` and opens itself, one row of ten cards per
query, with a **green** border on results the dev set judged relevant, **red**
on any result that breaks a hard constraint, and the reason underneath it. Re-run
the command and refresh.

Against the cluster those are the real product photos. Against the local fixture
the products are invented and their image URLs are synthetic, so the cards fall
back to coloured tiles and the page says so.

```
segment              P@10    was  viol  stock  brands  n
------------------------------------------------------------
head_term           0.300            0   0.55    0.63  2
exact_model         1.000  0.250     0   0.57    0.10  4
constrained         0.950  0.000     0   1.00    0.80  2
...
------------------------------------------------------------
OVERALL             0.693  0.314     0   0.64    0.56  14

build_query names 2 of the 3 vectors this collection carries (dense, sparse)
```

- **precision@10** — relevant results in the top 10, over `min(10, |relevant|)`.
  Judgments are **predicate-derived, not hand-labelled**: each query states a
  rule over payload, and every point satisfying it counts as relevant. The rule
  travels with the query in the `judgment` field, so you can audit it, and
  argue with it.
- **hard constraint violations** — if any of your top 10 breaks a stated
  constraint ("under $80", "in stock"), that query scores **zero**, however good
  the rest of it was.
- **p95 latency** — server-side, read from Qdrant's own timing, not wall clock.
- **in-stock rate** — share of the top 10 that is actually buyable.
- **brand diversity** — distinct brands in the top 10, **excluding `"Unknown"`**.
  `"Unknown"` is a placeholder for a missing store name, not a brand; counting
  it would pay you for returning junk.
- **`was`** — the same figure from your previous run, shown only where it moved.
  Change one thing, re-run, see which way it went. You do not have to keep notes.
  Approximate search returns a slightly different set each time, so a segment can
  drift a few hundredths between identical runs, and a rebuilt fixture shifts it
  further. Chase changes you can explain, not the last digit.

The line under the table counts how many of the collection's three vectors your
`build_query` even mentions. It says nothing about whether you used them well —
it is a grep, not a judgement — but one of three is worth noticing.

### The six segments

| segment | what it is |
|---|---|
| `head_term` | broad category queries — "running shoes", "backpack" |
| `exact_model` | brand + model number. Dense alone struggles; this is where hybrid shows up. |
| `constrained` | hard constraints in the text — "under $80", "in stock" |
| `natural_language` | descriptive intent with little keyword overlap |
| `similar_item` | driven by `seed_asin` — more like this one |
| `long_tail` | rare, specific phrasing with small relevant sets |

A good answer is not one trick, and the measurements say so plainly. Going from
dense-only to a dense+BM25 hybrid takes `exact_model` from 0.667 to 1.000 — and
takes `natural_language` from 0.600 **down** to 0.300, because a descriptive
query gives BM25 nothing to match. `head_term` sits at 1.000 either way; a broad
term over a 1,500-point category is free precision, here and in real life.

The thing that wins `exact_model` is not the thing that wins
`natural_language`. Anything you apply to every query uniformly will pay for one
segment with another.

## Working offline, or against a local Qdrant

The cluster is shared. If you would rather develop against your own copy, there
is a local fixture: a 2,000-point collection under podman that matches the
schema contract exactly.

```bash
bash fixture/up.sh            # start Qdrant 1.19.1 + seed it
# then set QDRANT_URL=http://localhost:6333 in .env
python harness.py
```

`build_query()` does not change between the two. The only difference is a client
flag: locally `cloud_inference=False` and FastEmbed embeds your `Document` on
your machine; against the cluster `cloud_inference=True` and Qdrant Cloud
Inference embeds it server-side. Same request either way.

**The fixture exists to exercise code paths, not to tune relevance.** 2,000
invented products, and its `image` vectors are deterministic noise rather than
real CLIP embeddings — image queries run, but their ranking means nothing
locally.

There are two dev sets and they are not interchangeable. `dev_set.jsonl` is the
real one, judged against the cluster. `fixture/dev_set.fixture.jsonl` is a
placeholder judged against the fixture, every row marked `"placeholder": true`.
The harness picks whichever matches the URL you point it at, because running the
real set against the fixture scores a flat zero and looks like your bug.

```bash
bash fixture/up.sh --recreate   # reseed from scratch
bash fixture/up.sh --down       # stop it (keeps the data)
python fixture/verify_api.py    # check every shape in docs/query-api.md still runs
```

## If something breaks

- **429 / "rate limited"** — ~40 people share one cluster and Cloud Inference
  throttles. Wait a moment and re-run. Not a bug in your query. The harness caps
  concurrency at 4 for this reason; leave it there.
- **Zero results from an image query** — you probably did not gate on
  `HasVectorCondition`, or used the vision model id to embed text. See trap 1
  and 2.
- **`build_query returned X, expected models.QueryRequest`** — you returned a
  response, a list, or `None`.
- **Anything else** — `python setup_check.py` first.

## Files

| file | |
|---|---|
| `build_query.py` | **the only file you edit** |
| `README.md` / `AGENTS.md` | this, and the same for coding agents |
| `docs/query-api.md` | Query API reference, every snippet executed before shipping |
| `harness.py` | runs the dev set and scores you |
| `setup_check.py` | pre-work gate |
| `common.py` | collection name, model ids, `point_id_for_asin` |
| `dev_set.jsonl` | placeholder queries + judgments |
| `fixture/` | local podman Qdrant, seeder, API check |
