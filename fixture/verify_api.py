"""Execute every API shape documented in docs/query-api.md against the fixture.

    python fixture/verify_api.py

Not part of the exercise. This exists so docs/query-api.md cannot quietly rot:
if a client or server upgrade changes one of these shapes, this fails instead of
a participant discovering it mid-workshop.
""" 
import sys
from qdrant_client import QdrantClient, models
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import COLLECTION, DENSE_MODEL, SPARSE_MODEL, IMAGE_DIM, point_id_for_asin

c = QdrantClient(url="http://localhost:6333", cloud_inference=False, timeout=60)
Q = "waterproof hiking boots"
pid = c.scroll(COLLECTION, limit=1)[0][0].id
img_pid = c.scroll(COLLECTION, limit=1,
    scroll_filter=models.Filter(must=[models.HasVectorCondition(has_vector="image")]))[0][0].id

def run(name, req):
    try:
        n = len(c.query_batch_points(COLLECTION, requests=[req])[0].points)
        print(f"  OK   {name:<44} {n} hits")
    except Exception as e:
        print(f"  FAIL {name:<44} {type(e).__name__}: {str(e).splitlines()[0][:110]}")
        return False
    return True

D = lambda: models.Document(text=Q, model=DENSE_MODEL)
S = lambda: models.Document(text=Q, model=SPARSE_MODEL)
hybrid_prefetch = lambda: [
    models.Prefetch(query=D(), using="dense", limit=60),
    models.Prefetch(query=S(), using="sparse", limit=60)]

ok = True
print("fusion:")
ok &= run("FusionQuery RRF", models.QueryRequest(
    prefetch=hybrid_prefetch(), query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10, with_payload=True))
ok &= run("FusionQuery DBSF", models.QueryRequest(
    prefetch=hybrid_prefetch(), query=models.FusionQuery(fusion=models.Fusion.DBSF),
    limit=10, with_payload=True))
ok &= run("RrfQuery weighted", models.QueryRequest(
    prefetch=hybrid_prefetch(),
    query=models.RrfQuery(rrf=models.Rrf(k=2, weights=[3.0, 1.0])),
    limit=10, with_payload=True))

print("filters:")
ok &= run("must/should/must_not + Range", models.QueryRequest(
    query=D(), using="dense", limit=10, with_payload=True,
    filter=models.Filter(
        must=[models.FieldCondition(key="in_stock", match=models.MatchValue(value=True)),
              models.FieldCondition(key="price", range=models.Range(gte=20.0, lte=80.0))],
        should=[models.FieldCondition(key="brand", match=models.MatchValue(value="Kestrel"))],
        must_not=[models.FieldCondition(key="brand", match=models.MatchValue(value="Unknown"))])))
ok &= run("MatchAny", models.QueryRequest(query=D(), using="dense", limit=10,
    filter=models.Filter(must=[models.FieldCondition(key="brand",
        match=models.MatchAny(any=["Kestrel", "Torrent"]))])))
ok &= run("MatchExcept", models.QueryRequest(query=D(), using="dense", limit=10,
    filter=models.Filter(must=[models.FieldCondition(key="brand",
        match=models.MatchExcept(**{"except": ["Unknown"]}))])))
ok &= run("HasVectorCondition gate on image", models.QueryRequest(
    query=[0.0] * IMAGE_DIM, using="image", limit=10,
    filter=models.Filter(must=[models.HasVectorCondition(has_vector="image")])))

print("formula + decay:")
ok &= run("FormulaQuery sum/mult + defaults", models.QueryRequest(
    prefetch=models.Prefetch(query=D(), using="dense", limit=100),
    query=models.FormulaQuery(
        formula=models.SumExpression(sum=["$score", models.MultExpression(mult=[0.2, "margin"])]),
        defaults={"margin": 0.0}),
    limit=10, with_payload=True))
ok &= run("Filter-as-boost inside formula", models.QueryRequest(
    prefetch=models.Prefetch(query=D(), using="dense", limit=100),
    query=models.FormulaQuery(formula=models.SumExpression(sum=["$score",
        models.MultExpression(mult=[0.3, models.Filter(must=[models.FieldCondition(
            key="in_stock", match=models.MatchValue(value=True))])])])),
    limit=10, with_payload=True))
for nm, expr in [("gauss", models.GaussDecayExpression(gauss_decay=models.DecayParamsExpression(
                    x="price", target=60.0, scale=40.0))),
                 ("exp", models.ExpDecayExpression(exp_decay=models.DecayParamsExpression(
                    x="price", target=60.0, scale=40.0))),
                 ("lin", models.LinDecayExpression(lin_decay=models.DecayParamsExpression(
                    x="price", target=60.0, scale=40.0, midpoint=0.5)))]:
    ok &= run(f"{nm}_decay on price", models.QueryRequest(
        prefetch=models.Prefetch(query=D(), using="dense", limit=100),
        query=models.FormulaQuery(formula=models.SumExpression(sum=["$score", expr]),
                                  defaults={"price": 0.0}),
        limit=10, with_payload=True))
ok &= run("rating>0 gated boost (trap 3)", models.QueryRequest(
    prefetch=models.Prefetch(query=D(), using="dense", limit=100),
    query=models.FormulaQuery(formula=models.SumExpression(sum=["$score",
        models.MultExpression(mult=[0.1, "rating",
            models.Filter(must=[models.FieldCondition(key="rating",
                range=models.Range(gt=0.0))])])]), defaults={"rating": 0.0}),
    limit=10, with_payload=True))

print("recommend:")
for strat in (models.RecommendStrategy.AVERAGE_VECTOR, models.RecommendStrategy.BEST_SCORE,
              models.RecommendStrategy.SUM_SCORES):
    ok &= run(f"Recommend {strat.value}", models.QueryRequest(
        query=models.RecommendQuery(recommend=models.RecommendInput(
            positive=[pid], strategy=strat)), using="dense", limit=10, with_payload=True))
ok &= run("Recommend on image + gate", models.QueryRequest(
    query=models.RecommendQuery(recommend=models.RecommendInput(positive=[img_pid])),
    using="image", limit=10,
    filter=models.Filter(must=[models.HasVectorCondition(has_vector="image")])))

print("search params:")
ok &= run("SearchParams + quantization", models.QueryRequest(
    query=D(), using="dense", limit=10, with_payload=True,
    params=models.SearchParams(hnsw_ef=128, exact=False, indexed_only=False,
        quantization=models.QuantizationSearchParams(
            ignore=False, rescore=True, oversampling=4.0))))
ok &= run("params on a Prefetch", models.QueryRequest(
    prefetch=[models.Prefetch(query=D(), using="dense", limit=60,
                params=models.SearchParams(quantization=models.QuantizationSearchParams(
                    rescore=True, oversampling=4.0))),
              models.Prefetch(query=S(), using="sparse", limit=60)],
    query=models.FusionQuery(fusion=models.Fusion.RRF), limit=10, with_payload=True))
ok &= run("exact=True brute force", models.QueryRequest(
    query=D(), using="dense", limit=10, params=models.SearchParams(exact=True)))
ok &= run("nested prefetch (fuse then rescore)", models.QueryRequest(
    prefetch=models.Prefetch(prefetch=hybrid_prefetch(),
        query=models.FusionQuery(fusion=models.Fusion.RRF), limit=100),
    query=models.FormulaQuery(formula=models.SumExpression(sum=["$score",
        models.MultExpression(mult=[0.1, "margin"])]), defaults={"margin": 0.0}),
    limit=10, with_payload=True))

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
