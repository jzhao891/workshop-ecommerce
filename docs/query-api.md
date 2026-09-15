# Qdrant Query API — what you can put in a `QueryRequest`

Reference for `build_query()`. Checked against **qdrant-client 1.19.0 / server
1.19.1** — the versions this workshop pins. Every snippet here was executed
against the fixture before this file shipped.

If your agent writes something that is not in here, be suspicious. The Query API
changed shape in 1.10 and again around 1.14–1.17, and models trained on older
material confidently emit `search()`, `recommend()`, `on_disk=True` and
`always_ram=True`. Those are the old spellings.

---

## The one thing you return

```python
models.QueryRequest(
    prefetch=...,      # optional: one or more sub-queries, can nest
    query=...,         # the query itself, or the fusion/formula over prefetches
    using=...,         # which vector: "dense" | "image" | "sparse"
    filter=...,        # models.Filter
    params=...,        # models.SearchParams
    limit=10,
    with_payload=True,
)
```

Note the field is **`filter`**, not `query_filter`. `query_filter` is the
argument name on `client.query_points()`; inside a `QueryRequest` (and inside a
`Prefetch`) it is `filter`. This one bites almost everybody.

`offset` only applies to the main query, so any prefetch must have a `limit` of
at least the main `limit + offset`.

## The collection

| vector | kind | dim | distance | notes |
|---|---|---|---|---|
| `dense` | dense | 384 | Cosine | `sentence-transformers/all-minilm-l6-v2` over `"{title}. {category_path}"` |
| `image` | dense | 512 | Cosine | `qdrant/clip-vit-b-32-vision`; **only on 19,997 of 100,000 points** |
| `sparse` | sparse | — | — | `qdrant/bm25`, `modifier=idf` |

Indexed payload fields — anything not in this list cannot be filtered, and you
hold a read-only key so you cannot add an index:

`brand` (keyword) · `category_path` (keyword) · `price` (float) ·
`rating` (float) · `review_count` (integer) · `in_stock` (bool) ·
`margin` (float) · `asin` (keyword)

`title` is deliberately **not** indexed. Match text through `sparse`, not a filter.

## `models.Document` — text in, vector out

```python
models.Document(text=query_text, model="sentence-transformers/all-minilm-l6-v2")
```

You never embed anything yourself. `Document` is a field in the request; Qdrant
resolves it. Use the same model id the vector was built with or the scores are
meaningless.

Against the cloud cluster (`cloud_inference=True`) the `Document` is passed
through and Qdrant Cloud Inference embeds it server-side. Against the local
fixture (`cloud_inference=False`) the client embeds it with FastEmbed before
sending. **Your `build_query()` is identical either way** — that is the whole
reason the fixture is useful.

(`qdrant/bm25` is a special case: it is passed through to the server in *both*
modes, because Qdrant 1.19 computes BM25 in the server core rather than through
an embedding model. Nothing you write changes.)

## Hybrid: prefetch + fusion

Two retrievers, one request. This is the single biggest win available to you.

```python
models.QueryRequest(
    prefetch=[
        models.Prefetch(
            query=models.Document(text=q, model=DENSE_MODEL),
            using="dense", limit=60),
        models.Prefetch(
            query=models.Document(text=q, model=SPARSE_MODEL),
            using="sparse", limit=60),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
    with_payload=True,
)
```

`models.Fusion.RRF` — reciprocal rank fusion, `score = Σ 1/(k + rank)`. Rank
based, so it does not care that cosine similarity and BM25 scores are on
different scales. This is the safe default.

`models.Fusion.DBSF` — distribution-based score fusion. Normalises each
retriever's scores using mean and standard deviation (3-sigma endpoints) and
then combines. Score based, so it can express "this one is *much* better", and
is correspondingly more sensitive to a retriever with a weird score
distribution.

Weighted RRF, if you want one retriever to count for more:

```python
models.RrfQuery(rrf=models.Rrf(k=2, weights=[3.0, 1.0]))   # weights match prefetch order
```

Prefetches nest. A common shape is "fuse dense+sparse, then re-score the fused
list with a formula" — that is a prefetch containing prefetches.

## Filters

```python
models.Filter(
    must=[                      # AND
        models.FieldCondition(key="in_stock", match=models.MatchValue(value=True)),
        models.FieldCondition(key="price", range=models.Range(gte=20.0, lte=80.0)),
    ],
    should=[                    # OR — raises score, does not restrict
        models.FieldCondition(key="brand", match=models.MatchValue(value="Kestrel")),
    ],
    must_not=[                  # NOT
        models.FieldCondition(key="brand", match=models.MatchValue(value="Unknown")),
    ],
)
```

- `models.Range(gt=, gte=, lt=, lte=)` for `price`, `rating`, `review_count`, `margin`.
- `models.MatchValue(value=...)` for one exact keyword, integer or bool.
  **It rejects floats.** `MatchValue(value=0.0)` raises a pydantic
  `ValidationError` before the request is ever sent, so for an exact match on
  `price`, `rating` or `margin` use `models.Range(gte=x, lte=x)`.
- `models.MatchAny(any=[...])` for "brand in this list", `models.MatchExcept(**{"except": [...]})` for "not in this list".
- `models.HasVectorCondition(has_vector="image")` — point has that named vector.

A filter on a `Prefetch` restricts that retriever. A filter on the
`QueryRequest` restricts the final result. When a constraint is genuinely hard
("in stock", "under $80"), put it on both — a prefetch that returns 60 candidates
of which 55 get filtered out afterwards has wasted 55 slots.

## `FormulaQuery` — re-score with payload

Rescores an existing candidate list. It must sit over a prefetch; `$score` is
the incoming score.

```python
models.QueryRequest(
    prefetch=models.Prefetch(
        query=models.Document(text=q, model=DENSE_MODEL),
        using="dense", limit=100),
    query=models.FormulaQuery(
        formula=models.SumExpression(sum=[
            "$score",
            models.MultExpression(mult=[0.2, "margin"]),
        ]),
        defaults={"margin": 0.0},   # what to use when the field is missing
    ),
    limit=10,
)
```

Available: `SumExpression`, `MultExpression`, `DivExpression`, `NegExpression`,
`AbsExpression`, `SqrtExpression`, `PowExpression`, `ExpExpression`,
`Log10Expression`, `LnExpression`, plus a bare field name as a string, a bare
number, and a `models.Filter` (which evaluates to 1.0 when the point matches and
0.0 when it does not — that is how you express a conditional boost).

Set `defaults` for every field you reference. A missing field with no default
makes the whole expression fail for that point.

### Decay functions

`LinDecayExpression`, `ExpDecayExpression`, `GaussDecayExpression`, each taking
`models.DecayParamsExpression(x=, target=, scale=, midpoint=)`. Score is 1.0
when `x == target` and falls to `midpoint` (default 0.5) at `scale` away.

```python
# prefer products priced near $60, without excluding anything
models.GaussDecayExpression(
    gauss_decay=models.DecayParamsExpression(x="price", target=60.0, scale=40.0))
```

Decay expresses a *preference*. A `Filter` in `must` expresses a *requirement*.
Using decay for a hard constraint is how you end up with a $300 boot in the
results for "under $80".

## `Recommend` — more like this

```python
models.QueryRequest(
    query=models.RecommendQuery(recommend=models.RecommendInput(
        positive=[point_id, ...],       # point ids, raw vectors, or Documents
        negative=[...],
        strategy=models.RecommendStrategy.BEST_SCORE,
    )),
    using="dense", limit=10,
)
```

Strategies: `AVERAGE_VECTOR` (mean of positives, then one search — cheap, blurs
several distinct positives into their average), `BEST_SCORE` (best score against
any positive — keeps distinct positives distinct), `SUM_SCORES`.

`seed_asin` is an **ASIN, not a point id**. Point ids are UUIDv5 over the ASIN:
`common.point_id_for_asin(asin)` re-derives one with no lookup. You can also
filter on the indexed `asin` field instead.

## `MMR` — diversity in the result list

Maximal Marginal Relevance re-ranks a candidate pool to trade a little relevance
for less redundancy. It rides on `NearestQuery`, not `FusionQuery`:

```python
models.QueryRequest(
    query=models.NearestQuery(
        nearest=models.Document(text=q, model=DENSE_MODEL),
        mmr=models.Mmr(diversity=0.5, candidates_limit=100)),
    using="dense", limit=10, with_payload=True)
```

`diversity` runs 0.0 (pure relevance, same as no MMR) to 1.0 (push hard for
dissimilar results). `candidates_limit` is the pool it re-ranks — too small and
there is nothing to diversify.

Measured on the fixture, `"running shoes"` at `limit=5`: plain nearest returns 3
distinct brands, `diversity=0.5` returns 5. The harness scores brand diversity,
so this is the direct lever for it — but note it diversifies on *vector*
similarity, not on the `brand` field, so it is a proxy and not a guarantee.

It composes with prefetch in both directions: as the main query over a hybrid
prefetch, or inside a prefetch with fusion on top. What it cannot do is *be* the
fusion step — `FusionQuery` and `NearestQuery` are alternatives in the `query`
slot.

## `SearchParams` and the quantization levers

```python
models.SearchParams(
    hnsw_ef=128,        # HNSW candidate list; higher = more accurate, slower
    exact=False,        # True = brute force, ignores HNSW. Ground truth, slow.
    indexed_only=False,
    quantization=models.QuantizationSearchParams(
        ignore=False,
        rescore=True,       # re-rank the shortlist with the original vectors
        oversampling=4.0,   # fetch 4x `limit` from quantized, then rescore
    ),
)
```

Both `dense` and `image` are stored int8-quantized (quantile 0.99), quantized
copies pinned in RAM, originals cold on disk. Searching hits the quantized
copies, which is fast and slightly lossy. `oversampling=4.0` with `rescore=True`
pulls a 4× shortlist from the quantized vectors and re-ranks it using the exact
ones — usually a real precision gain for a small latency cost.

`params` goes on the `QueryRequest` *and* on each `Prefetch`; a prefetch does
not inherit it. In a hybrid query the levers belong on the dense prefetch, which
is where the vector search actually happens.

---

## Traps

These three have each cost somebody an afternoon.

**1. `image` exists on only 19,997 of 100,000 points.** Any query touching
`image` must gate on

```python
models.Filter(must=[models.HasVectorCondition(has_vector="image")])
```

or it silently searches a fifth of the corpus and you will never notice. There
is no error, no warning, and the results look plausible.

**2. Stored image vectors came from `qdrant/clip-vit-b-32-vision`.**
Text-to-image search requires **`qdrant/clip-vit-b-32-text`** — the other tower
of the same CLIP model. Image-to-image works as-is.

Unlike trap 1, this one fails *loudly*, which is the good news. Measured against
the cluster:

| what you send to `using="image"` | what comes back |
|---|---|
| `Document(text=..., model="qdrant/clip-vit-b-32-text")` | works |
| `Document(text=..., model="qdrant/clip-vit-b-32-vision")` | `500` from the inference service — the vision tower cannot embed text |
| `Document(text=..., model="sentence-transformers/all-minilm-l6-v2")` | `400 Wrong input: Vector dimension error: expected dim: 512, got 384` |

**3. `rating` barely discriminates; `review_count` does.** There are no unrated
products — the minimum rating in the corpus is 1.0 — and the scale is squashed
into its top end: **94.4% of points are rated ≥ 3.0, 73.8% ≥ 4.0, and the 90th
percentile is a flat 5.0**. A `FormulaQuery` that boosts on `rating` is mostly
reshuffling near-ties, and it pays for that with relevance. `review_count` is
the field with actual spread — p10 = 1, p50 = 14, p90 = 254, max = 118,988 — so
if you want a popularity prior, use `log10(1 + review_count)`, not stars.

Price has the same shape in reverse: p50 is $41.97 but the maximum is $41,999.99,
so a raw `price` term in a formula is dominated by a handful of outliers. Put it
through a decay function, or a log, or leave it in a filter.

## Rules

- **One request per search.** One `QueryRequest`. Nested prefetches are still
  one request.
- **No client-side reranking.** No reordering, filtering or trimming after the
  fact. You never see the results, so there is nothing to rerank anyway.
- **No network calls at query time.** No embedding API, no product lookup.
  `models.Document` is not a network call on your side — it is a field in the
  request that Qdrant resolves.

## Fixture vs cluster

The local podman fixture holds ~2,000 invented products and exists to make your
code *run*. Its `image` vectors are deterministic noise, not CLIP embeddings, so
image queries execute but their ranking means nothing. Relevance is judged on
the 100,000-point cluster.
