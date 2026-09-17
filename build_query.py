"""THE ONLY FILE YOU EDIT.

Hybrid dense+sparse with constraints as hard filters, plus query-shape-aware
fusion weighting.

Strategy is switchable for A/B measurement via QSHAPE_STRATEGY:
    uniform    equal-weight RRF over both retrievers (no shape analysis)
    branch     prose queries go dense-only; everything else hybrid
    weighted   always hybrid, RRF weights vary continuously with query shape
"""
import os
import re

from qdrant_client import models

from common import DENSE_MODEL, SPARSE_MODEL, point_id_for_asin

TOP_K = 10        # the harness scores precision@10; returning fewer can only hurt
PREFETCH_K = int(os.environ.get("QSHAPE_PREFETCH_K", "60"))

# dense and image are stored int8-quantized (quantile 0.99) with the quantized
# copies pinned in RAM and the originals cold on disk, so an ordinary search is
# fast and slightly lossy. oversampling pulls a wider shortlist from the
# quantized vectors and rescore re-ranks it with the exact ones -- accuracy
# recovery, not a relevance bet, so it costs no segment anything.
HNSW_EF = int(os.environ.get("QSHAPE_HNSW_EF", "128"))
OVERSAMPLING = float(os.environ.get("QSHAPE_OVERSAMPLING", "4.0"))

DENSE_PARAMS = models.SearchParams(
    hnsw_ef=HNSW_EF,
    quantization=models.QuantizationSearchParams(
        rescore=True, oversampling=OVERSAMPLING),
) if OVERSAMPLING > 0 else None

STRATEGY = os.environ.get("QSHAPE_STRATEGY", "combo")

# k for weighted RRF. Large on purpose: 1/(k+rank) flattens as k grows, so the
# weight multiplier -- not the rank position -- decides which retriever wins.
# At the server default (k=2) a 3:1 weighting still lets the losing retriever's
# top hit land around rank 7, i.e. inside the scored top 10.
RRF_K = int(os.environ.get("QSHAPE_RRF_K", "20"))

PROSE_THRESHOLD = 0.15

STRIP_CONSTRAINT_TEXT = os.environ.get("QSHAPE_STRIP", "1") == "1"

# How to handle a described (prose) query: "dense" searches the sentence as
# written, "dense_stripped" strips grammatical scaffolding first,
# "hybrid_stripped" also brings BM25 back over the stripped text.
NL_MODE = os.environ.get("QSHAPE_NL_MODE", "dense")
NL_SPARSE_W = float(os.environ.get("QSHAPE_NL_SPARSE_W", "0.5"))

# How to handle a seed_asin: "off" ignores it, "only" uses the seed vector
# alone, "hybrid" fuses it with a BM25 match on the accompanying text.
SEED_MODE = os.environ.get("QSHAPE_SEED_MODE", "only")
SEED_WEIGHTS = [float(os.environ.get("QSHAPE_SEED_W", "2.0")), 1.0]

# First-person need language. Deliberately NOT a general stopword list: words
# like "under", "in", "only" express constraints, which arrive structured in the
# `constraints` argument anyway, and counting them drags constrained queries up
# into the prose range. These words describe a shopper describing themselves,
# and they never appear in a product title -- which is exactly why BM25 has
# nothing to match when they show up.
_NEED_WORDS = frozenset("""
i me my mine we our us you your that which who whom
can could would will shall should might may wo
without after before when while during something anything
""".split())

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'-]*")

# Grammatical scaffolding to drop from a prose query, so what is left reads
# more like a product title. Strictly closed-class words (determiners,
# pronouns, auxiliaries, prepositions, copulas) plus a few generic light verbs
# -- no product vocabulary, so this carries no knowledge of the 14 visible
# questions and cannot quietly encode their answers.
_SCAFFOLD = frozenset("""
a an the this that these those there here
i me my mine we us our you your it its
am is are was were be been being do does did done
can could will would shall should may might must
have has had get gets got getting
of in on at to for from with without by as into onto
and or but not no nor so if when while during after before
something anything thing things some any all every each
want wants need needs looking look find
""".split())


def _strip_scaffold(query_text: str) -> str:
    """Reduce a described need to its content words.

    "a watch I can wear swimming" -> "watch wear swimming". Product titles are
    noun phrases; prose queries are sentences. Dropping the sentence machinery
    moves the query into the same register as the thing it has to match.

    Guarded: if nothing survives, keep the original.
    """
    kept = [t for t in _TOKEN.findall(query_text.lower()) if t not in _SCAFFOLD]
    return " ".join(kept) if kept else query_text

# Constraint language, stripped only when `constraints` already carries the
# same limit structurally. "baseball caps under $20, in stock only" retrieves
# on "baseball caps": the rest is redundant with the filter, and actively
# harmful to BM25, which happily matches "stock" and "only" against titles.
_CONSTRAINT_PHRASE = re.compile(
    r"""\b(?:
          (?:for\s+)?(?:under|over|below|above|less\s+than|more\s+than|
             cheaper\s+than|at\s+least|at\s+most|up\s+to)\s*\$?\d[\d,.]*
        | \$\d[\d,.]*(?:\s*(?:or\s+)?(?:less|under|below|and\s+under))?
        | in\s+stock(?:\s+only)?
        | out\s+of\s+stock
        | (?:rated\s+)?(?:above|over|at\s+least)?\s*[\d.]+\s*(?:\+|stars?)
        )\b""",
    re.IGNORECASE | re.VERBOSE)


def _strip_constraints(query_text: str, constraints: dict | None) -> str:
    """Drop constraint language that `constraints` already states structurally.

    Guarded: if stripping would leave nothing, keep the original text. Better a
    noisy query than an empty one.
    """
    if not constraints or not STRIP_CONSTRAINT_TEXT:
        return query_text
    cleaned = _CONSTRAINT_PHRASE.sub(" ", query_text)
    cleaned = re.sub(r"\s*,\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;")
    return cleaned if cleaned else query_text


def _shape(query_text: str) -> tuple[float, bool]:
    """(prose_ratio, has_model_code) for one query.

    prose_ratio    fraction of tokens that are first-person need language.
                   ~0.33-0.36 on described queries, 0.0 on keyword queries.
    has_model_code a token mixing letters and digits ("W51012Q4", "TR-450").
                   This is where BM25 beats dense outright -- embeddings are
                   poor at model codes -- so it argues for MORE sparse, not less.
    """
    tokens = _TOKEN.findall(query_text.lower())
    if not tokens:
        return 0.0, False
    prose = sum(1 for t in tokens if t in _NEED_WORDS) / len(tokens)
    has_code = any(any(c.isalpha() for c in t) and any(c.isdigit() for c in t)
                   for t in tokens)
    return prose, has_code


def _constraint_filter(constraints: dict | None) -> models.Filter | None:
    """The five constraint keys -> a hard `must` filter.

    Every one is a hard limit: break it and the question scores zero, so they
    belong in `must`, never in a decay or a score boost.

    MatchValue rejects floats, so price and rating go through Range even though
    in_stock and brand can match directly.
    """
    if not constraints:
        return None

    must: list[models.Condition] = []

    min_price, max_price = constraints.get("min_price"), constraints.get("max_price")
    if min_price is not None or max_price is not None:
        # one Range carries both ends; either may be None, meaning unbounded
        must.append(models.FieldCondition(
            key="price", range=models.Range(gte=min_price, lte=max_price)))

    min_rating = constraints.get("min_rating")
    if min_rating is not None:
        must.append(models.FieldCondition(
            key="rating", range=models.Range(gte=min_rating)))

    in_stock = constraints.get("in_stock")
    if in_stock is not None:
        must.append(models.FieldCondition(
            key="in_stock", match=models.MatchValue(value=bool(in_stock))))

    brand = constraints.get("brand")
    if brand is not None:
        must.append(models.FieldCondition(
            key="brand", match=models.MatchValue(value=brand)))

    return models.Filter(must=must) if must else None


def _prefetch(query_text: str, flt: models.Filter | None) -> list[models.Prefetch]:
    """Dense and sparse candidate lists, in that order.

    The filter goes on each prefetch as well as the outer request: a prefetch
    that returns 60 candidates of which 55 are filtered out afterwards has
    wasted 55 slots.
    """
    return [
        models.Prefetch(
            query=models.Document(text=query_text, model=DENSE_MODEL),
            using="dense", limit=PREFETCH_K, filter=flt, params=DENSE_PARAMS),
        # sparse is not quantized and has no HNSW graph, so the params above
        # are meaningless here; a prefetch does not inherit them anyway
        models.Prefetch(
            query=models.Document(text=query_text, model=SPARSE_MODEL),
            using="sparse", limit=PREFETCH_K, filter=flt),
    ]


def _weights(prose: float, has_code: bool) -> list[float]:
    """RRF weights [dense, sparse] from query shape.

    Continuous rather than a branch, so a misread query degrades gradually
    instead of flipping to a completely different query plan.
    """
    if has_code:
        return [1.0, 2.0]          # model codes: lean on BM25
    # fades sparse out as the query gets more descriptive; 1.0 at prose=0,
    # ~0.3 by the time a query looks like "shoes I can stand in all day"
    return [1.0, max(0.0, 1.0 - 2.0 * prose)]


def _seed_request(query_text: str,
                  seed_asin: str,
                  flt: models.Filter | None) -> models.QueryRequest:
    """'More like this one', anchored on the seed product's own vector.

    seed_asin is an ASIN, not a point id. point_id_for_asin re-derives the
    UUIDv5 with no lookup, so this stays one request and zero network calls.
    Getting it backwards fails two ways: an ASIN in a point-id slot is a loud
    400, a point id in the `asin` field is a silent zero hits.

    BEST_SCORE rather than AVERAGE_VECTOR -- with a single positive they agree,
    but BEST_SCORE keeps distinct positives distinct if one is ever added.
    RecommendQuery already excludes the seed from its own results, so the seed
    does not come back and waste a slot.
    """
    recommend = models.RecommendQuery(recommend=models.RecommendInput(
        positive=[point_id_for_asin(seed_asin)],
        strategy=models.RecommendStrategy.BEST_SCORE,
    ))

    # No text to fuse with, so the seed vector is all there is.
    if SEED_MODE == "only" or not query_text.strip():
        return models.QueryRequest(
            query=recommend, using="dense", filter=flt, params=DENSE_PARAMS,
            limit=TOP_K, with_payload=True)

    # Fuse "near the seed in vector space" with "shares words with the seed
    # title". The seed vector leads; the text is corroboration.
    return models.QueryRequest(
        prefetch=[
            models.Prefetch(query=recommend, using="dense",
                            limit=PREFETCH_K, filter=flt),
            models.Prefetch(
                query=models.Document(text=query_text, model=SPARSE_MODEL),
                using="sparse", limit=PREFETCH_K, filter=flt),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=RRF_K, weights=SEED_WEIGHTS)),
        filter=flt,
        limit=TOP_K,
        with_payload=True,
    )


def build_query(query_text: str,
                seed_asin: str | None = None,
                constraints: dict | None = None) -> models.QueryRequest:
    """Build one request."""
    flt = _constraint_filter(constraints)
    query_text = _strip_constraints(query_text, constraints)
    prose, has_code = _shape(query_text)

    if seed_asin and SEED_MODE != "off":
        return _seed_request(query_text, seed_asin, flt)

    if STRATEGY in ("branch", "combo") and prose >= PROSE_THRESHOLD and not has_code:
        # Prose query. BM25 has nothing to match in the raw sentence, so the
        # default is to drop it entirely rather than downweight it -- weighting
        # alone cannot fully silence a retriever.
        nl_text = query_text if NL_MODE == "dense" else _strip_scaffold(query_text)

        if NL_MODE == "hybrid_stripped":
            # Once the sentence machinery is gone, the content words are title
            # register again, so BM25 may have something real to match after all.
            return models.QueryRequest(
                prefetch=_prefetch(nl_text, flt),
                query=models.RrfQuery(rrf=models.Rrf(
                    k=RRF_K, weights=[1.0, NL_SPARSE_W])),
                filter=flt,
                limit=TOP_K,
                with_payload=True,
            )

        return models.QueryRequest(
            query=models.NearestQuery(
                nearest=models.Document(text=nl_text, model=DENSE_MODEL)),
            using="dense",
            filter=flt,
            params=DENSE_PARAMS,
            limit=TOP_K,
            with_payload=True,
        )

    if STRATEGY in ("weighted", "combo"):
        query = models.RrfQuery(rrf=models.Rrf(
            k=RRF_K, weights=_weights(prose, has_code)))
    else:
        query = models.FusionQuery(fusion=models.Fusion.RRF)

    return models.QueryRequest(
        prefetch=_prefetch(query_text, flt),
        query=query,
        filter=flt,  # note: `filter` here, not `query_filter`
        limit=TOP_K,
        with_payload=True,
    )
