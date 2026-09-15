#!/usr/bin/env python3
"""Local scorer. Runs the dev set through YOUR build_query() and reports.

    python harness.py                 # whatever QDRANT_URL points at
    python harness.py --show q03      # dump the top 10 for one query
    python harness.py --html          # same run, as a product grid in your browser

You do not need to read this file to do the exercise, but nothing here is
hidden from you either.

Reported per segment and overall:
  precision@10          hits / min(10, |relevant|); a query that violates a
                        hard constraint scores 0 regardless of its hits
  violations            queries with >=1 top-10 result breaking a hard constraint
  p95 latency           server-side, taken from Qdrant's own `time` field
  in-stock rate         share of top-10 results with in_stock == true
  brand diversity       distinct brands in top 10, EXCLUDING "Unknown", over the
                        number of non-Unknown results. "Unknown" is a placeholder
                        for a missing store name, not a brand -- counting it
                        would pay you for returning junk.

The `was` column is the same figure from your previous run, so you can change
one thing, re-run, and see which way it moved.
"""
import argparse
import contextlib
import hashlib
import html
import json
import statistics
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qdrant_client.http import models as http_models

import build_query as build_query_module
from build_query import build_query
from common import COLLECTION, SEGMENTS, is_local, make_client

# Previous run's scores, so the table can answer "did the change I just made
# help" without you keeping notes. Kept per dev set: the fixture set and the
# real set are scored against different corpora, and comparing one against the
# other produces a delta that means nothing.
STATE_DIR = Path(__file__).resolve().parent / ".workshop"

# One shared cluster, ~40 people in the room. Four in flight from each laptop is
# already 160 concurrent requests against it; fanning out wider does not make
# your harness finish sooner, it makes everyone's queries slower and trips Cloud
# Inference rate limits for the whole room. Leave this at 4.
MAX_CONCURRENCY = 4

TOP_K = 10

# The client's local FastEmbed path keeps mutable per-client batch state, so two
# threads calling it at once raise "dictionary changed size during iteration".
# Only the embedding step needs serialising -- the HTTP query itself stays
# concurrent, and in cloud mode this lock is never taken at all.
_EMBED_LOCK = threading.Lock()
_NULL_LOCK = contextlib.nullcontext()


def _raw_capable(client) -> bool:
    """Can we reach the batch endpoint that reports server-side time?"""
    return all(hasattr(client, a) for a in
               ("_inference_inspector", "_embed_models", "cloud_inference", "http"))


def execute(client, req, raw_ok: bool):
    """Run one request. Returns (points, server_ms).

    We call the raw batch endpoint rather than client.query_batch_points because
    the typed wrapper returns only `.result` and drops Qdrant's `time` field,
    and server-side latency is one of the numbers we report.

    ponytail: reaches two underscore-prefixed attrs to reuse the client's own
    Document-embedding step; if a client upgrade moves them we fall back to the
    public call and wall-clock timing (flagged in the output).
    """
    req = req.model_copy(update={"with_payload": True})  # we need payload to score
    if raw_ok:
        reqs = [req]
        # Local: FastEmbed resolves `dense` Documents here. Cloud: they pass
        # through untouched for Cloud Inference. Identical request either way.
        if not client.cloud_inference and client._inference_inspector.inspect(reqs):
            with _EMBED_LOCK:
                reqs = list(client._embed_models(reqs, is_query=True))
        resp = client.http.search_api.query_batch_points(
            collection_name=COLLECTION,
            query_request_batch=http_models.QueryRequestBatch(searches=reqs))
        return resp.result[0].points, (resp.time or 0.0) * 1000.0
    # Fallback: the public call embeds Documents internally, so locally the
    # whole call has to be serialised, not just the embedding step.
    lock = _EMBED_LOCK if not getattr(client, "cloud_inference", True) else _NULL_LOCK
    with lock:
        t0 = time.perf_counter()
        pts = client.query_batch_points(COLLECTION, requests=[req])[0].points
        return pts, (time.perf_counter() - t0) * 1000.0


def violates(payload: dict, constraints: dict) -> str | None:
    """Return the first hard-constraint violation, or None."""
    if not payload:
        return "no payload returned"
    price, stock = payload.get("price"), payload.get("in_stock")
    if "max_price" in constraints and (price is None or price > constraints["max_price"]):
        return f"price {price} > {constraints['max_price']}"
    if "min_price" in constraints and (price is None or price < constraints["min_price"]):
        return f"price {price} < {constraints['min_price']}"
    if "in_stock" in constraints and stock is not constraints["in_stock"]:
        return f"in_stock {stock} != {constraints['in_stock']}"
    if "brand" in constraints and payload.get("brand") != constraints["brand"]:
        return f"brand {payload.get('brand')!r} != {constraints['brand']!r}"
    if "min_rating" in constraints:
        r = payload.get("rating")
        if r is None or r < constraints["min_rating"]:
            return f"rating {r} < {constraints['min_rating']}"
    return None


def score_one(case: dict, points) -> dict:
    top = points[:TOP_K]
    payloads = [p.payload or {} for p in top]
    got = [pl.get("asin") for pl in payloads]
    relevant = set(case["relevant_asins"])

    violation = None
    for pl in payloads:
        if (v := violates(pl, case.get("constraints") or {})):
            violation = f'{pl.get("asin")}: {v}'
            break

    hits = sum(1 for a in got if a in relevant)
    denom = min(TOP_K, len(relevant)) or 1
    precision = 0.0 if violation else hits / denom

    non_unknown = [pl.get("brand") for pl in payloads if pl.get("brand") != "Unknown"]
    diversity = len(set(non_unknown)) / len(non_unknown) if non_unknown else 0.0
    in_stock = (sum(1 for pl in payloads if pl.get("in_stock")) / len(top)) if top else 0.0

    return dict(id=case["id"], segment=case["segment"], precision=precision,
                hits=hits, denom=denom, violation=violation, returned=len(top),
                in_stock=in_stock, diversity=diversity, top=payloads)


def run_case(client, case, raw_ok):
    try:
        req = build_query(case["query"], case.get("seed_asin"))
    except Exception as e:
        return dict(id=case["id"], segment=case["segment"], error=f"build_query raised: {e}")

    if not isinstance(req, http_models.QueryRequest):
        return dict(id=case["id"], segment=case["segment"],
                    error=f"build_query returned {type(req).__name__}, "
                          f"expected models.QueryRequest")
    if (req.limit or 10) < TOP_K:
        print(f"  note: {case['id']} asked for limit={req.limit}; "
              f"precision@{TOP_K} cannot reach 1.0 with fewer than {TOP_K} results")

    try:
        points, server_ms = execute(client, req, raw_ok)
    except Exception as e:
        code = getattr(e, "status_code", None)
        if code == 429 or "429" in str(e) or "Too Many Requests" in str(e):
            return dict(id=case["id"], segment=case["segment"],
                        error="rate limited, wait a moment and run it again")
        return dict(id=case["id"], segment=case["segment"],
                    error=f"{type(e).__name__}: {str(e).splitlines()[0][:160]}")

    out = score_one(case, points)
    out["server_ms"] = server_ms
    return out


def remember(current: dict, state: Path) -> dict:
    """Return the previous run's figures, then record this run's.

    A truncated or missing file costs a comparison; it must never cost you the
    score itself, so every failure here is swallowed.
    """
    try:
        previous = json.loads(state.read_text())
    except Exception:
        previous = {}
    # A run where everything failed (cluster down, bad key) has nothing to say.
    # Recording it would throw away the baseline you actually want to compare
    # against, so leave the previous run in place.
    if current:
        try:
            state.parent.mkdir(exist_ok=True)
            state.write_text(json.dumps(current))
        except Exception:
            pass
    return previous


def vectors_mentioned(client) -> str | None:
    """How many of the collection's vectors your build_query even mentions.

    Execution, never correctness -- it greps your source for quoted vector
    names. A collection carrying three representations and a build_query that
    names one is not necessarily wrong, but it is worth knowing.
    """
    try:
        params = client.get_collection(COLLECTION).config.params
        present = set(params.vectors or {}) | set(params.sparse_vectors or {})
        src = Path(build_query_module.__file__).read_text()
        used = {v for v in present if f'"{v}"' in src or f"'{v}'" in src}
    except Exception:
        return None
    if not present:
        return None
    names = ", ".join(sorted(used)) if used else "none by name"
    return (f"build_query names {len(used)} of the {len(present)} vectors this "
            f"collection carries ({names})")


def report(results: list[dict], placeholder: bool, raw_ok: bool, client=None,
           state: Path | None = None) -> None:
    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]

    current = {seg: round(statistics.fmean(r["precision"] for r in rs), 3)
               for seg in SEGMENTS
               if (rs := [r for r in ok if r["segment"] == seg])}
    if ok:
        current["OVERALL"] = round(statistics.fmean(r["precision"] for r in ok), 3)
    previous = remember(current, state) if state else {}

    def was(key: str) -> str:
        """Blank when there is nothing to compare, or nothing moved."""
        before = previous.get(key)
        if before is None or abs(before - current.get(key, 0.0)) < 5e-4:
            return ""
        return f"{before:.3f}"

    print(f"\n{'segment':<18} {'P@10':>6} {'was':>6} {'viol':>5} {'stock':>6} "
          f"{'brands':>7}  n")
    print("-" * 60)
    for seg in SEGMENTS:
        rs = [r for r in ok if r["segment"] == seg]
        if not rs:
            print(f"{seg:<18} {'-':>6} {'':>6} {'-':>5} {'-':>6} {'-':>7}  0")
            continue
        print(f"{seg:<18} "
              f"{statistics.fmean(r['precision'] for r in rs):>6.3f} "
              f"{was(seg):>6} "
              f"{sum(1 for r in rs if r['violation']):>5} "
              f"{statistics.fmean(r['in_stock'] for r in rs):>6.2f} "
              f"{statistics.fmean(r['diversity'] for r in rs):>7.2f}  {len(rs)}")
    print("-" * 60)
    if ok:
        lat = sorted(r["server_ms"] for r in ok)
        p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
        print(f"{'OVERALL':<18} "
              f"{statistics.fmean(r['precision'] for r in ok):>6.3f} "
              f"{was('OVERALL'):>6} "
              f"{sum(1 for r in ok if r['violation']):>5} "
              f"{statistics.fmean(r['in_stock'] for r in ok):>6.2f} "
              f"{statistics.fmean(r['diversity'] for r in ok):>7.2f}  {len(ok)}")
        label = "server-side" if raw_ok else "wall-clock (server time unavailable)"
        print(f"\np95 latency ({label}): {p95:.1f} ms   "
              f"median: {statistics.median(lat):.1f} ms")

    viol = [r for r in ok if r["violation"]]
    if viol:
        print(f"\nhard constraint violations ({len(viol)} queries scored zero):")
        for r in viol:
            print(f"  {r['id']}  {r['violation']}")

    if bad:
        print(f"\nfailed ({len(bad)}):")
        for r in bad:
            print(f"  {r['id']}  {r['error']}")

    if client is not None and (nudge := vectors_mentioned(client)):
        print(f"\n{nudge}")

    if placeholder:
        print(f"\n*** dev_set.jsonl is the PLACEHOLDER set: {len(results)} queries whose")
        print("*** judgments were written against the 2,000-point local fixture. The")
        print("*** numbers above say your query runs and is roughly sane. They are not")
        print("*** relevance. The real dev set drops into this same file.")


# --------------------------------------------------------------- html report

# Product search is visual. A wrong result is obvious in a picture and invisible
# in a title, which is the whole reason this exists. It is a file, not a server:
# you already re-run this command after every edit, so regenerating a page and
# hitting refresh is the same loop for none of the code.
CSS = """
:root { color-scheme: light dark; }
body { margin: 0; padding: 24px; font: 14px/1.5 system-ui, -apple-system, sans-serif;
       background: #fafaf9; color: #1c1917; }
@media (prefers-color-scheme: dark) { body { background: #1c1917; color: #e7e5e4; } }
h1 { font-size: 18px; margin: 0 0 4px; }
.sub { opacity: .65; margin-bottom: 20px; font-size: 13px; }
.q { margin: 0 0 26px; }
.qhead { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap;
         border-top: 1px solid rgba(128,128,128,.3); padding-top: 10px; }
.qhead b { font-size: 15px; }
.seg { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
       opacity: .6; }
.p10 { margin-left: auto; font-variant-numeric: tabular-nums; font-size: 13px; }
.row { display: flex; gap: 10px; overflow-x: auto; padding: 12px 2px 4px; }
.card { flex: 0 0 132px; border: 2px solid transparent; border-radius: 8px;
        overflow: hidden; background: rgba(128,128,128,.10); }
.card.hit { border-color: #16a34a; }
.card.bad { border-color: #dc2626; }
.thumb { height: 132px; position: relative; overflow: hidden; }
/* The image sits ON TOP of the fallback tile. If it 404s it hides itself and
   the tile shows through; height is fixed rather than 100% so a tall portrait
   photo cannot stretch the card. */
.thumb img { position: relative; z-index: 1; display: block; width: 100%;
             height: 132px; object-fit: contain; background: #fff; }
.cap { padding: 7px 8px 9px; font-size: 11.5px; }
.cap .t { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
          overflow: hidden; min-height: 2.6em; }
.meta { opacity: .7; margin-top: 4px; font-variant-numeric: tabular-nums; }
.unknown { font-style: italic; opacity: .55; }
.oos { color: #dc2626; }
.why { color: #dc2626; font-size: 11px; margin-top: 4px; }
.legend { font-size: 12px; opacity: .7; margin-bottom: 18px; }
.note { font-size: 12px; opacity: .8; background: rgba(234,179,8,.15);
        padding: 10px 12px; border-radius: 6px; margin-bottom: 18px; }
"""


def _tile(asin: str) -> str:
    """Deterministic colour per product, so a missing image is still a distinct
    card rather than a broken-image icon."""
    h = int(hashlib.blake2b(asin.encode(), digest_size=2).hexdigest(), 16) % 360
    return f"hsl({h} 45% 80%)"


def _card(pl: dict, relevant: set, constraints: dict) -> str:
    e = html.escape
    asin = pl.get("asin") or "?"
    title = pl.get("title") or "(no title)"
    why = violates(pl, constraints)
    cls = "card bad" if why else ("card hit" if asin in relevant else "card")
    brand = pl.get("brand") or "?"
    brand_html = (f'<span class="unknown">{e(brand)}</span>'
                  if brand == "Unknown" else e(brand))
    rating = pl.get("rating")
    rating_html = "unrated" if rating in (0.0, None) else f"{rating}\u2605"
    price = pl.get("price")
    price_html = f"${price:,.2f}" if isinstance(price, (int, float)) else "?"
    stock = "" if pl.get("in_stock") else ' <span class="oos">out of stock</span>'
    img = pl.get("image_url") or ""
    # The <img> sits on the coloured tile. If it 404s, or a proxy blocks the
    # host, or this is the fixture where the URLs are synthetic, it hides itself
    # and the tile shows through with the title still readable.
    img_html = (f'<img src="{e(img)}" alt="{e(title)}" loading="lazy" '
                f'onerror="this.style.display=\'none\'">') if img else ""
    # No text on the tile itself: the caption below already carries the title,
    # and low-contrast text over a pastel block helped nobody.
    return (f'<figure class="{cls}">'
            f'<div class="thumb" style="background:{_tile(asin)}">'
            f'{img_html}</div>'
            f'<figcaption class="cap"><div class="t">{e(title)}</div>'
            f'<div class="meta">{price_html} \u00b7 {rating_html} \u00b7 '
            f'{brand_html}{stock}</div>'
            + (f'<div class="why">{e(why)}</div>' if why else "")
            + '</figcaption></figure>')


def render_html(results: list[dict], cases: list[dict], url: str, out: Path) -> Path:
    e = html.escape
    by_id = {c["id"]: c for c in cases}
    ok = [r for r in results if "error" not in r]
    overall = statistics.fmean(r["precision"] for r in ok) if ok else 0.0

    blocks = []
    for r in results:
        case = by_id.get(r["id"], {})
        head = (f'<div class="qhead"><span class="seg">{e(r["segment"])}</span>'
                f'<b>{e(case.get("query", ""))}</b>')
        if "error" in r:
            blocks.append(f'<section class="q">{head}</span></div>'
                          f'<div class="why">{e(r["error"])}</div></section>')
            continue
        head += (f'<span class="p10">P@10 {r["precision"]:.3f}'
                 f'{"  &middot; constraint violated" if r["violation"] else ""}</span></div>')
        relevant = set(case.get("relevant_asins", []))
        cards = "".join(_card(pl, relevant, case.get("constraints") or {})
                        for pl in r["top"])
        blocks.append(f'<section class="q">{head}<div class="row">{cards}</div></section>')

    placeholder = any(c.get("placeholder") for c in cases)
    note = ('<div class="note">Placeholder dev set against the local fixture. The '
            'products are invented and their image URLs are synthetic, so the '
            'cards fall back to coloured tiles. Against the cluster these are '
            'real product photos.</div>') if placeholder and is_local(url) else ""

    out.parent.mkdir(exist_ok=True)
    out.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>harness results</title><style>{CSS}</style></head><body>"
        f"<h1>{len(ok)} queries &middot; overall P@10 {overall:.3f}</h1>"
        f"<div class='sub'>{e(url)}</div>{note}"
        "<div class='legend'>Green border: judged relevant. Red border: breaks a "
        "hard constraint, which scores the whole query zero. Grey: returned but "
        "not judged relevant.</div>"
        + "".join(blocks) + "</body></html>", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-set", help="default: dev_set.jsonl against the cluster, "
                                      "fixture/dev_set.fixture.jsonl against a local URL")
    ap.add_argument("--show", help="dump the top 10 for one query id and exit")
    ap.add_argument("--html", action="store_true",
                    help="also write a product grid to .workshop/results.html and open it")
    args = ap.parse_args()

    client, url = make_client()

    # The two sets are not interchangeable: the real one is judged against the
    # 100,000-point cluster and its ASINs do not exist in the fixture, so
    # running it locally would score a flat zero and look like your bug.
    here = Path(__file__).resolve().parent
    if args.dev_set:
        path = Path(args.dev_set)
    else:
        path = (here / "fixture" / "dev_set.fixture.jsonl" if is_local(url)
                else here / "dev_set.jsonl")
    if not path.exists():
        sys.exit(f"no {path}." + ("\n  seed the fixture first: bash fixture/up.sh"
                                  if is_local(url) else ""))
    cases = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    placeholder = any(c.get("placeholder") for c in cases)
    raw_ok = _raw_capable(client)
    mode = "local fixture (cloud_inference=False)" if is_local(url) \
        else "cloud cluster (cloud_inference=True)"
    print(f"{url}  --  {mode}\n{len(cases)} queries from {path.name}, "
          f"concurrency {MAX_CONCURRENCY}")

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        results = list(pool.map(lambda c: run_case(client, c, raw_ok), cases))

    if args.show:
        r = next((x for x in results if x["id"] == args.show), None)
        if r is None:
            sys.exit(f"no query {args.show!r} in {path}")
        if "error" in r:
            sys.exit(f"{args.show}: {r['error']}")
        case = next(c for c in cases if c["id"] == args.show)
        print(f"\n{args.show}  [{r['segment']}]  {case['query']!r}")
        print(f"P@10={r['precision']:.3f}  hits={r['hits']}/{r['denom']}  "
              f"violation={r['violation']}")
        for i, pl in enumerate(r["top"], 1):
            mark = "*" if pl.get("asin") in set(case["relevant_asins"]) else " "
            print(f" {mark}{i:>2}. {pl.get('title','?')[:60]:<60} "
                  f"${pl.get('price',0):>8.2f} stock={str(pl.get('in_stock'))[:1]} "
                  f"rating={pl.get('rating')}")
        return

    report(results, placeholder, raw_ok, client,
           STATE_DIR / f"last_run.{path.stem}.json")

    if args.html:
        page = render_html(results, cases, url,
                           Path(__file__).resolve().parent / ".workshop" / "results.html")
        print(f"\nwrote {page}")
        try:
            webbrowser.open(page.as_uri())
        except Exception:
            pass  # headless or no browser: the path above is enough


if __name__ == "__main__":
    main()
