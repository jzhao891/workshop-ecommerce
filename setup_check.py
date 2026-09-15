#!/usr/bin/env python3
"""Pre-work gate. Run this BEFORE the workshop.

    python setup_check.py

One green line means you are done. Anything else names the check that failed
and what to do about it. You do not need to read any other file to fix it.
"""
import sys
import uuid
from pathlib import Path

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def fail(check: str, problem: str, fix: str) -> None:
    print(f"{RED}FAILED: {check}{RESET}")
    print(f"  problem: {problem}")
    print(f"  fix:     {fix}")
    sys.exit(1)


def main() -> None:
    # 1. Python version -----------------------------------------------------
    if sys.version_info < (3, 10):
        fail("python version",
             f"this is Python {sys.version_info.major}.{sys.version_info.minor}, "
             f"the workshop needs 3.10 or newer",
             "install Python 3.10+ and re-create the venv:\n"
             "           python3.12 -m venv .venv && source .venv/bin/activate\n"
             "           pip install -r requirements.txt")

    # 2. Dependencies -------------------------------------------------------
    try:
        import qdrant_client  # noqa: F401
        from qdrant_client import QdrantClient, models
    except ImportError as e:
        fail("dependencies", f"cannot import qdrant_client ({e})",
             "pip install -r requirements.txt")

    import importlib.metadata as md
    version = md.version("qdrant-client")
    if version != "1.19.0":
        fail("client version",
             f"qdrant-client is {version}, the workshop pins 1.19.0 against "
             f"server 1.19.1",
             "pip install -r requirements.txt  (or: pip install qdrant-client==1.19.0)")

    # 3. Credentials --------------------------------------------------------
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import COLLECTION, CLOUD_POINTS, DENSE_MODEL, SPARSE_MODEL, is_local, make_client

    import os
    from common import load_env
    load_env()
    url = os.environ.get("QDRANT_URL")
    if not url:
        fail("credentials", "QDRANT_URL is not set",
             "copy .env.example to .env and paste in the URL and key you were "
             "emailed:\n           cp .env.example .env")
    local = is_local(url)
    if not local and not os.environ.get("QDRANT_API_KEY"):
        fail("credentials", "QDRANT_API_KEY is not set (needed for the cloud cluster)",
             "put your read-only key in .env as QDRANT_API_KEY=...")

    client, url = make_client()

    # 4. Cluster reachable --------------------------------------------------
    try:
        client.get_collections()
    except Exception as e:
        fail("cluster reachable", f"could not reach {url} ({type(e).__name__}: "
             f"{str(e).splitlines()[0][:120]})",
             "check QDRANT_URL in .env for typos, and that you are not behind a "
             "proxy that blocks outbound 6333/443.\n"
             "           If you are on the local fixture, start it: bash fixture/up.sh")

    # 5. The key is read-only -----------------------------------------------
    # Participants are meant to hold a read-only key. Handing out the wrong one
    # is a facilitator mistake that nobody notices until something is deleted.
    #
    # This probes scope WITHOUT writing anything: it attempts an index creation
    # on a collection name that cannot exist. A read-only key is refused (403)
    # before Qdrant ever looks for the collection; a write-capable key gets as
    # far as "no such collection" (404). Either way nothing is created and
    # nothing on the shared cluster is touched.
    if not local and os.environ.get("QDRANT_API_KEY"):
        probe = f"setup-check-probe-{uuid.uuid4()}"
        try:
            client.create_payload_index(collection_name=probe, field_name="probe",
                                        field_schema=models.PayloadSchemaType.KEYWORD)
            writable = True          # it got through -- definitely not read-only
        except Exception as e:
            status = getattr(e, "status_code", None)
            writable = status == 404
            # Any other outcome (403, a network error, an unfamiliar status) is
            # not evidence of a write-capable key, and this check must never be
            # the thing that blocks somebody from starting.
        if writable:
            fail("key scope",
                 "this API key can write to the cluster; the workshop key is "
                 "read-only",
                 "you were given the wrong key. Ask the workshop host for the "
                 "read-only one and replace QDRANT_API_KEY in .env.\n"
                 "           Do not use the write key -- the cluster is shared.")

    # 6. Collection exists --------------------------------------------------
    try:
        exists = client.collection_exists(COLLECTION)
    except Exception as e:
        # Anything that is not a clean yes/no -- a proxy returning HTML, a
        # gateway rewriting the body -- lands here. A raw traceback at this
        # point tells an attendee nothing they can act on.
        fail("collection exists",
             f"{url} did not answer with something this client understands "
             f"({type(e).__name__}: {str(e).splitlines()[0][:120]})",
             "usually a corporate proxy or VPN rewriting the response. Try "
             "another network, or check QDRANT_URL points at Qdrant itself and "
             "not at a gateway in front of it.")
    if not exists:
        fail("collection exists", f"{url} has no collection named {COLLECTION!r}",
             "your key may point at the wrong cluster -- re-check QDRANT_URL.\n"
             "           On the local fixture: python fixture/seed_fixture.py")

    # 7. Point count --------------------------------------------------------
    try:
        count = client.count(COLLECTION, exact=True).count
    except Exception as e:
        fail("point count",
             f"could not count points in {COLLECTION} ({type(e).__name__}: "
             f"{str(e).splitlines()[0][:120]})",
             "if this says forbidden, your key is not scoped to this "
             "collection -- tell the workshop host.")
    if local:
        if count == 0:
            fail("point count", f"the local fixture collection is empty",
                 "python fixture/seed_fixture.py --recreate")
    elif count != CLOUD_POINTS:
        fail("point count",
             f"{COLLECTION} holds {count:,} points, expected {CLOUD_POINTS:,}",
             "the cluster may still be loading -- wait a minute and re-run. If it "
             "persists, tell the workshop host; do not try to fix it yourself, "
             "your key is read-only.")

    # 8. Cloud Inference actually works under this key -----------------------
    # This is the check that matters. A read-only key that can read the
    # collection can still be unable to run inference, and you would not find
    # out until the room is live.
    try:
        got = client.query_points(
            COLLECTION, using="dense",
            query=models.Document(text="waterproof hiking boots", model=DENSE_MODEL),
            limit=3).points
    except Exception as e:
        code = getattr(e, "status_code", None)
        if code == 429 or "429" in str(e):
            fail("inference query", "the cluster rate limited this request",
                 "wait a moment and run setup_check.py again -- this is not a "
                 "configuration problem.")
        fail("inference query",
             f"a Document-based dense query failed ({type(e).__name__}: "
             f"{str(e).splitlines()[0][:140]})",
             "if this says inference is not configured, your key cannot use Cloud "
             "Inference -- tell the workshop host.\n"
             "           On the local fixture this needs FastEmbed: pip install fastembed")
    if not got:
        fail("inference query",
             "the dense query ran but returned zero results",
             "that should be impossible on a loaded collection -- tell the host.")

    # 9. Sparse/BM25 round trip ---------------------------------------------
    # Stored sparse vectors and the query-side BM25 must tokenise the same way.
    # If they ever diverge this returns nothing, and it is much better to learn
    # that here than mid-exercise.
    try:
        sparse_hits = client.query_points(
            COLLECTION, using="sparse",
            query=models.Document(text="waterproof hiking boots", model=SPARSE_MODEL),
            limit=3).points
    except Exception as e:
        fail("sparse query",
             f"a BM25 query failed ({type(e).__name__}: {str(e).splitlines()[0][:140]})",
             "tell the workshop host -- this is a server-side configuration issue.")
    if not sparse_hits:
        fail("sparse query",
             "the BM25 query ran but returned zero results",
             "tell the workshop host: the stored sparse vectors and query-side "
             "BM25 disagree. Do not work around it.")

    # How many points actually carry an `image` vector. Printed because trap 1
    # is much harder to ignore as a number you saw during setup than as a
    # sentence in a README: an ungated image query searches only these.
    try:
        with_image = client.count(COLLECTION, exact=True, count_filter=models.Filter(
            must=[models.HasVectorCondition(has_vector="image")])).count
        image_note = f", image vectors on {with_image:,} of them"
    except Exception:
        image_note = ""

    where = "local fixture" if local else "cloud cluster"
    print(f"{GREEN}All checks passed{RESET} -- {where} at {url}, "
          f"{COLLECTION} with {count:,} points{image_note}, dense + sparse "
          f"queries both returning results. You are ready.")


if __name__ == "__main__":
    main()
