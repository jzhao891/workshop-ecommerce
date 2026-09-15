#!/usr/bin/env python3
"""Build the local podman fixture: a ~2,000 point `workshop_products` collection
that matches schema_contract.md exactly, plus a placeholder dev set.

WHAT THIS FIXTURE IS FOR
------------------------
Exercising code paths, not tuning relevance. The products are invented, the
image vectors are noise, and 2,000 points is a toy. A build_query() that scores
well here is not thereby a good query -- it is a query that *runs*. Relevance is
judged on the real 100,000-point cluster with the real dev set.

What it does reproduce faithfully, because these are what break code:
  * every vector name, dimension, distance and quantization setting
  * memory=cold on originals, memory=pinned on the quantized copies
  * modifier=idf on `sparse`, which BM25 needs
  * all eight payload indexes
  * `image` present on only ~20% of points   -> HasVectorCondition matters
  * brand == "Unknown" on ~8%                -> brand diversity gets gamed
  * rating squashed into its top end, as in the real corpus (no unrated points)
  * exact brand + model-number titles        -> hybrid visibly beats dense

indexing_threshold is set to 1,000 so HNSW actually builds. At the 20,000
default a 2,000-point collection stays a flat scan forever, and then
oversampling and rescore do nothing at all -- the levers would be inert and the
exercise would quietly teach the wrong lesson.

Usage:
    python fixture/seed_fixture.py               # create + load + write dev set
    python fixture/seed_fixture.py --recreate    # drop first
"""
import argparse
import hashlib
import json
import math
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient, models

from common import (COLLECTION, DENSE_DIM, DENSE_MODEL, IMAGE_DIM, SPARSE_MODEL,
                    point_id_for_asin)

SEED = 20260914          # same seed as the real corpus manifest
N_PRODUCTS = 2_000
IMAGE_FRACTION = 0.20
# applied to non-family products; families are real branded lines and never
# "Unknown", so this is tuned to land the whole corpus at ~8%.
UNKNOWN_BRAND_FRACTION = 0.094
# The live corpus has NO unrated products: minimum rating 1.0, 94.4% >= 3.0,
# p90 exactly 5.0. Mirror that squashed shape rather than inventing 0.0 ratings.
RATING_MEAN, RATING_SD = 4.2, 0.75
INDEXING_THRESHOLD = 1_000

# ---------------------------------------------------------------- corpus shape

ROOT = "Clothing, Shoes & Jewelry"

# category_path -> (product nouns, attribute words that may appear in titles)
CATEGORIES = {
    f"{ROOT} > Men > Shoes > Hiking Boots":
        (["Hiking Boot", "Trail Boot", "Backpacking Boot"],
         ["Waterproof", "Insulated", "Gore-Tex", "Lightweight", "Steel Toe"]),
    f"{ROOT} > Men > Shoes > Running Shoes":
        (["Running Shoe", "Trail Runner", "Road Running Shoe"],
         ["Cushioned", "Breathable", "Lightweight", "Stability", "Carbon Plate"]),
    f"{ROOT} > Women > Shoes > Running Shoes":
        (["Running Shoe", "Trail Runner", "Road Running Shoe"],
         ["Cushioned", "Breathable", "Lightweight", "Stability", "Wide Fit"]),
    f"{ROOT} > Women > Shoes > Sandals":
        (["Sandal", "Slide Sandal", "Strappy Sandal"],
         ["Leather", "Arch Support", "Waterproof", "Adjustable"]),
    f"{ROOT} > Men > Clothing > Jackets":
        (["Rain Jacket", "Insulated Jacket", "Softshell Jacket"],
         ["Waterproof", "Insulated", "Packable", "Windproof", "Merino Wool"]),
    f"{ROOT} > Women > Clothing > Dresses":
        (["Wrap Dress", "Midi Dress", "Shirt Dress"],
         ["Linen", "Cotton", "Machine Washable", "Pockets"]),
    f"{ROOT} > Men > Accessories > Watches":
        (["Dive Watch", "Field Watch", "Chronograph Watch"],
         ["Sapphire Crystal", "Automatic", "Titanium", "Solar", "200m"]),
    f"{ROOT} > Women > Jewelry > Necklaces":
        (["Pendant Necklace", "Chain Necklace", "Locket Necklace"],
         ["Sterling Silver", "14k Gold", "Hypoallergenic", "Adjustable"]),
    f"{ROOT} > Unisex > Accessories > Backpacks":
        (["Backpack", "Daypack", "Travel Backpack"],
         ["Waterproof", "Laptop Sleeve", "Carry-On", "35L", "Recycled"]),
    f"{ROOT} > Men > Clothing > Shirts":
        (["Flannel Shirt", "Oxford Shirt", "Base Layer Shirt"],
         ["Merino Wool", "Organic Cotton", "Wrinkle Free", "Long Sleeve"]),
}

BRANDS = ["Kestrel", "Northvale", "Ridgeline", "Halcyon", "Ironwood", "Vesper",
          "Cobalt Creek", "Marlowe", "Sable & Finch", "Torrent", "Alpenglow",
          "Bramble", "Quarry", "Lumen", "Foxglove", "Stonecrop", "Meridian",
          "Tanager", "Wildfell", "Corvus", "Juniper Lane", "Aldwin", "Pellucid",
          "Ваlemont"]
BRANDS[-1] = "Valemont"

COLORS = ["Black", "Charcoal", "Navy", "Olive", "Rust", "Sand", "Burgundy",
          "Forest Green", "Slate", "Cream"]

# Model-code shapes. These are the tokens BM25 nails and dense embeddings blur.
CODE_SHAPES = ["{a}{a}-{n3}", "{a}{n4}", "{a}{a}{n3}-{a}", "{n3}-{n2}"]

# Model-code FAMILIES are the reason hybrid visibly beats dense.
#
# A family is one brand's product line: same brand, same category, same words,
# 48 SKUs that differ only in a long numeric model number. To a 384d dense
# embedding of "{title}. {category_path}" those 48 titles are nearly one point.
#
# The code shape matters and was measured, not guessed. Asking minilm for
# "{brand} {code}" and looking for that exact SKU inside its own 36-member
# family, the target lands in the top 10:
#     RM-703      (letters + dash + 3 digits)   69%   <- too easy, dense wins
#     1570-891    (digits + dash + digits)      50%
#     DW5610E1V   (letters + digits + letters)  47%
#     10457821    (long numeric)                42%   <- what we use
# BM25 puts the exact code first every time. At 48 siblings the gap is wide
# enough to see in one run instead of being a coin flip.
N_FAMILIES = 6
FAMILY_SIZE = 48


def make_code(rng: random.Random) -> str:
    shape = rng.choice(CODE_SHAPES)
    out = shape
    while "{a}" in out:
        out = out.replace("{a}", rng.choice("ABCDEFGHJKLMNPRSTVWXZ"), 1)
    out = out.replace("{n4}", str(rng.randint(1000, 9999)))
    out = out.replace("{n3}", str(rng.randint(100, 999)))
    out = out.replace("{n2}", str(rng.randint(10, 99)))
    return out


def make_asin(rng: random.Random) -> str:
    return "B0" + "".join(rng.choice("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
                          for _ in range(8))


def synth(asin: str, price: float, review_count: int, rating: float,
          seed: int = SEED) -> tuple[bool, float]:
    """in_stock + margin, verbatim from schema_contract.md.

    blake2b(asin, person=seed) means these are order-independent: reseeding in a
    different order produces byte-identical payloads.
    """
    h = hashlib.blake2b(asin.encode(), digest_size=8,
                        person=seed.to_bytes(8, "little", signed=False)).digest()
    rng = random.Random(int.from_bytes(h, "little"))
    a = (0.55 + 0.75 * math.log10(1 + review_count)
         - 1.35 * math.log10(max(price, 1.0) / 30.0)
         + 0.30 * (rating - 4.0))
    in_stock = rng.random() < 1.0 / (1.0 + math.exp(-a))
    mu = min(0.72, max(0.04, 0.22 + 0.11 * math.log10(max(price, 1.0) / 20.0)
                       - 0.02 * math.log10(1 + review_count)
                       + 0.03 * (rating - 4.2)))
    return in_stock, round(rng.betavariate(mu * 25.0, (1 - mu) * 25.0), 4)


def fake_image_vector(asin: str) -> list[float]:
    """Deterministic unit vector, 512d.

    NOT a real CLIP embedding -- there are no real images here. Image *queries*
    run and return points; image *relevance* is meaningless in the fixture.
    On the cluster these are qdrant/clip-vit-b-32-vision embeddings of the real
    product photo.
    """
    rng = random.Random("img:" + asin)
    v = [rng.gauss(0, 1) for _ in range(IMAGE_DIM)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def build_families(rng: random.Random, used_codes: set[str]) -> list[dict]:
    """Near-identical SKU families: dense cannot separate them, BM25 can."""
    cats = list(CATEGORIES)
    rows = []
    for f in range(N_FAMILIES):
        cat = cats[f % len(cats)]
        nouns, attrs = CATEGORIES[cat]
        brand = BRANDS[f * 3 % len(BRANDS)]
        noun, attr = rng.choice(nouns), rng.choice(attrs)
        base = rng.randint(10_000_000, 99_000_000)
        for k in range(FAMILY_SIZE):
            code = str(base + k)
            if code in used_codes:
                continue
            used_codes.add(code)
            asin = make_asin(rng)
            # Title differs from its siblings by the code and nothing else.
            title = f"{brand} {attr} {noun} Model {code}"
            price = round(math.exp(rng.uniform(math.log(20), math.log(300))), 2)
            review_count = int(rng.paretovariate(1.2)) - 1
            rating = round(min(5.0, max(1.0, rng.gauss(RATING_MEAN, RATING_SD))), 2)
            in_stock, margin = synth(asin, price, review_count, rating)
            rows.append({
                "asin": asin, "title": title, "brand": brand, "category_path": cat,
                "price": price, "rating": rating, "review_count": review_count,
                "in_stock": in_stock, "margin": margin,
                "image_url": f"https://m.media-amazon.com/images/I/{asin}.jpg",
                "_code": code, "_attr": attr, "_noun": noun, "_family": f,
                "_has_image": rng.random() < IMAGE_FRACTION,
            })
    return rows


def build_corpus() -> list[dict]:
    rng = random.Random(SEED)
    cats = list(CATEGORIES)
    used_codes: set[str] = set()
    rows = build_families(rng, used_codes)
    for i in range(N_PRODUCTS - len(rows)):
        cat = cats[i % len(cats)]                  # even spread across categories
        nouns, attrs = CATEGORIES[cat]
        noun = rng.choice(nouns)
        attr = rng.choice(attrs)
        color = rng.choice(COLORS)
        asin = make_asin(rng)

        unknown = rng.random() < UNKNOWN_BRAND_FRACTION
        brand = "Unknown" if unknown else rng.choice(BRANDS)

        # ~30% of *branded* products carry a real model code in the title.
        # These are the exact-match targets: "Kestrel TR-450" is trivial for
        # BM25 and mush for a 384d dense embedding of the whole title.
        code = None
        if not unknown and rng.random() < 0.30:
            while (code := make_code(rng)) in used_codes:
                pass
            used_codes.add(code)
            title = f"{brand} {code} {attr} {noun} - {color}"
        elif unknown:
            title = f"{attr} {noun} for Men and Women - {color}"
        else:
            title = f"{brand} {attr} {noun} - {color}"

        price = round(math.exp(rng.uniform(math.log(6), math.log(420))), 2)
        review_count = int(rng.paretovariate(1.2)) - 1
        rating = round(min(5.0, max(1.0, rng.gauss(RATING_MEAN, RATING_SD))), 2)
        in_stock, margin = synth(asin, price, review_count, rating)

        rows.append({
            "asin": asin, "title": title, "brand": brand, "category_path": cat,
            "price": price, "rating": rating, "review_count": review_count,
            "in_stock": in_stock, "margin": margin,
            # synthetic placeholder; nothing fetches it
            "image_url": f"https://m.media-amazon.com/images/I/{asin}.jpg",
            "_code": code, "_attr": attr, "_noun": noun, "_family": None,
            "_has_image": rng.random() < IMAGE_FRACTION,
        })
    rng.shuffle(rows)
    return rows


# ------------------------------------------------------------------ collection

INDEXES = [("brand", models.PayloadSchemaType.KEYWORD),
           ("category_path", models.PayloadSchemaType.KEYWORD),
           ("price", models.PayloadSchemaType.FLOAT),
           ("rating", models.PayloadSchemaType.FLOAT),
           ("review_count", models.PayloadSchemaType.INTEGER),
           ("in_stock", models.PayloadSchemaType.BOOL),
           ("margin", models.PayloadSchemaType.FLOAT),
           ("asin", models.PayloadSchemaType.KEYWORD)]


def dense_params(size: int) -> models.VectorParams:
    """Originals cold on disk, int8 quantized copies pinned in RAM.

    `memory=cold` / `memory=pinned` is the CURRENT spelling. The old
    on_disk=True / always_ram=True pair still parses on some paths but is not
    what this contract specifies -- if your agent wrote those, it wrote them
    from memory.
    """
    return models.VectorParams(
        size=size,
        distance=models.Distance.COSINE,
        memory=models.Memory.COLD,
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                memory=models.Memory.PINNED)))


def create(client: QdrantClient, recreate: bool) -> None:
    if client.collection_exists(COLLECTION):
        if not recreate:
            print(f"  {COLLECTION} already exists -- pass --recreate to drop it")
            return
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config={"dense": dense_params(DENSE_DIM),
                        "image": dense_params(IMAGE_DIM)},
        sparse_vectors_config={
            # idf is not optional: without it the server does not apply inverse
            # document frequency and "bm25" is just term frequency.
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=INDEXING_THRESHOLD))
    for field, schema in INDEXES:
        client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)
    print(f"  created {COLLECTION}: dense(384) + image(512) + sparse(idf), "
          f"{len(INDEXES)} payload indexes, indexing_threshold={INDEXING_THRESHOLD}")


def load(client: QdrantClient, rows: list[dict], batch_size: int = 128) -> None:
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        points = []
        for r in chunk:
            # Document(...) here is embedded locally by FastEmbed for `dense`
            # and computed by the Qdrant core for `sparse` -- same call either way.
            vec = {
                "dense": models.Document(
                    text=f'{r["title"]}. {r["category_path"]}', model=DENSE_MODEL),
                "sparse": models.Document(
                    text=f'{r["title"]} {r["brand"]} {r["category_path"]}',
                    model=SPARSE_MODEL),
            }
            if r["_has_image"]:
                vec["image"] = fake_image_vector(r["asin"])
            points.append(models.PointStruct(
                id=point_id_for_asin(r["asin"]),
                vector=vec,
                payload={k: r[k] for k in
                         ("title", "brand", "category_path", "price", "rating",
                          "review_count", "in_stock", "margin", "image_url", "asin")}))
        client.upsert(COLLECTION, points=points, wait=True)
        print(f"\r  upserted {min(start + batch_size, len(rows)):,}/{len(rows):,}",
              end="", flush=True)
    print()


# -------------------------------------------------------------- placeholder dev set

def build_dev_set(rows: list[dict]) -> list[dict]:
    """Twelve queries, two per segment, judged against THIS fixture.

    Derived from the generated corpus rather than typed by hand so the
    judgments cannot drift from the data. Every row is marked placeholder.
    The real dev set replaces this file and nothing else changes.
    """
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category_path"], []).append(r)

    def asins(rs, cap=60):
        return [r["asin"] for r in rs[:cap]]

    def in_cat(*needles):
        return [r for r in rows if any(n in r["category_path"] for n in needles)]

    def titled(rs, word):
        return [r for r in rs if word.lower() in r["title"].lower()]

    # Pick exact-model targets from the middle of two different families: the
    # hard case, where 35 near-identical siblings crowd the dense ranking.
    fams: dict[int, list[dict]] = {}
    for r in rows:
        if r["_family"] is not None:
            fams.setdefault(r["_family"], []).append(r)
    picks = []
    for n, f in enumerate(sorted(fams)[:4]):
        members = sorted(fams[f], key=lambda r: r["_code"])
        picks.append(members[(FAMILY_SIZE * (n + 1)) // 5])
    m1, m2, m3, m4 = picks

    run = in_cat("Running Shoes")
    packs = in_cat("Backpacks")
    boots = in_cat("Hiking Boots")
    necks = in_cat("Necklaces")
    jackets = in_cat("Jackets")
    shirts = in_cat("Shirts")

    # similar_item seeds: a boot, and a watch
    seed_boot = boots[3]
    watches = in_cat("Watches")
    seed_watch = watches[2]

    cheap_instock_boots = [r for r in titled(boots, "Waterproof")
                           if r["price"] <= 80 and r["in_stock"]]
    cheap_instock_necks = [r for r in necks if r["price"] <= 50 and r["in_stock"]]

    q = [
        dict(id="q01", segment="head_term", query="running shoes",
             seed_asin=None, constraints={}, relevant_asins=asins(run),
             note="broad category head term"),
        dict(id="q02", segment="head_term", query="backpack",
             seed_asin=None, constraints={}, relevant_asins=asins(packs),
             note="broad category head term"),

        dict(id="q03", segment="exact_model",
             query=f'{m1["brand"]} {m1["_code"]}', seed_asin=None, constraints={},
             relevant_asins=[m1["asin"]],
             note="brand + model number; 35 near-identical siblings, "
                  "dense alone rarely lands the right one in the top 10"),
        dict(id="q04", segment="exact_model",
             query=f'{m2["brand"]} {m2["_code"]}', seed_asin=None, constraints={},
             relevant_asins=[m2["asin"]], note="brand + model number"),
        dict(id="q13", segment="exact_model",
             query=f'{m3["_code"]}', seed_asin=None, constraints={},
             relevant_asins=[m3["asin"]],
             note="bare model number, no brand; sparse/BM25 territory"),
        dict(id="q14", segment="exact_model",
             query=f'{m4["brand"]} model {m4["_code"]}', seed_asin=None, constraints={},
             relevant_asins=[m4["asin"]], note="brand + model number"),

        dict(id="q05", segment="constrained",
             query="waterproof hiking boots under $80 that are in stock",
             seed_asin=None, constraints={"max_price": 80.0, "in_stock": True},
             relevant_asins=asins(cheap_instock_boots),
             note="hard constraints: any violation in top 10 scores the query zero"),
        dict(id="q06", segment="constrained",
             query="necklace under 50 dollars, in stock only",
             seed_asin=None, constraints={"max_price": 50.0, "in_stock": True},
             relevant_asins=asins(cheap_instock_necks),
             note="hard constraints"),

        dict(id="q07", segment="natural_language",
             query="something warm and dry for hiking in the rain",
             seed_asin=None, constraints={},
             relevant_asins=asins(titled(jackets, "Waterproof")
                                  + titled(jackets, "Insulated")
                                  + titled(boots, "Waterproof")),
             note="descriptive intent, no keyword overlap with titles"),
        dict(id="q08", segment="natural_language",
             query="comfortable shoes I can stand in all day",
             seed_asin=None, constraints={},
             relevant_asins=asins(titled(run, "Cushioned") + titled(run, "Stability")),
             note="descriptive intent; dense should beat sparse here"),

        dict(id="q09", segment="similar_item",
             query=seed_boot["title"], seed_asin=seed_boot["asin"], constraints={},
             relevant_asins=asins([r for r in boots if r["asin"] != seed_boot["asin"]]),
             note="seed_asin is an ASIN, not a point id -- derive the UUID or filter"),
        dict(id="q10", segment="similar_item",
             query=seed_watch["title"], seed_asin=seed_watch["asin"], constraints={},
             relevant_asins=asins([r for r in watches if r["asin"] != seed_watch["asin"]]),
             note="seed_asin is an ASIN, not a point id"),

        dict(id="q11", segment="long_tail",
             query="merino wool base layer",
             seed_asin=None, constraints={},
             relevant_asins=asins(titled(shirts, "Merino") + titled(jackets, "Merino")),
             note="rare phrasing, small relevant set"),
        dict(id="q12", segment="long_tail",
             query="steel toe waterproof work boot",
             seed_asin=None, constraints={},
             relevant_asins=asins(titled(boots, "Steel Toe")),
             note="rare phrasing, small relevant set"),
    ]
    for row in q:
        row["placeholder"] = True
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()

    # Local fixture: cloud_inference=False, so FastEmbed embeds `dense` here.
    client = QdrantClient(url=args.url, cloud_inference=False, timeout=120)

    print(f"seeding fixture at {args.url}")
    create(client, args.recreate)

    rows = build_corpus()
    n_img = sum(r["_has_image"] for r in rows)
    n_unk = sum(r["brand"] == "Unknown" for r in rows)
    n_hi = sum(r["rating"] >= 3.0 for r in rows)
    n_coded = sum(bool(r["_code"]) for r in rows)
    n_fam = sum(r["_family"] is not None for r in rows)
    print(f"  generated {len(rows):,} products: {n_img:,} with image vectors "
          f"({100*n_img/len(rows):.1f}%), {n_unk:,} brand=Unknown "
          f"({100*n_unk/len(rows):.1f}%), {100*n_hi/len(rows):.1f}% rated >=3.0, "
          f"{n_coded:,} with model codes "
          f"({n_fam:,} of them in {N_FAMILIES} near-identical SKU families)")

    load(client, rows)

    dev = build_dev_set(rows)
    # NOT dev_set.jsonl: that is the real, cluster-judged set. Seeding the
    # fixture must never clobber it.
    out = Path(__file__).resolve().parent / "dev_set.fixture.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in dev))
    print(f"  wrote fixture/{out.name}: {len(dev)} PLACEHOLDER queries "
          f"(the harness picks this up automatically on a local URL)")

    info = client.get_collection(COLLECTION)
    print(f"  collection reports {info.points_count:,} points, "
          f"status={info.status}, indexed={info.indexed_vectors_count:,}")


if __name__ == "__main__":
    main()
