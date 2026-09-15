"""Shared constants and client construction.

Every name, dimension and model id in this file comes from schema_contract.md.
If you find yourself editing a constant here to make something work, the bug is
somewhere else.
"""
import os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient

COLLECTION = "workshop_products"

# Model ids. Use these exact strings -- query vectors must come from the same
# model as the stored vectors or the scores are meaningless.
DENSE_MODEL = "sentence-transformers/all-minilm-l6-v2"  # 384d -> vector "dense"
SPARSE_MODEL = "qdrant/bm25"                            # -> vector "sparse" (modifier=idf)
IMAGE_MODEL = "qdrant/clip-vit-b-32-vision"             # 512d -> stored "image" vectors
IMAGE_TEXT_MODEL = "qdrant/clip-vit-b-32-text"          # 512d -> query "image" WITH TEXT

DENSE_DIM, IMAGE_DIM = 384, 512

# Point ids are UUIDv5 over the ASIN. seed_asin is an ASIN, not a point id.
ASIN_NAMESPACE = uuid.UUID("6f1d6b1e-0b3f-5c2a-9b77-4a1c7e9d2f01")

CLOUD_POINTS = 100_000  # what the real cluster holds; the local fixture holds ~2,000

# The six evaluation segments, in report order.
SEGMENTS = ("head_term", "exact_model", "constrained",
            "natural_language", "similar_item", "long_tail")


def point_id_for_asin(asin: str) -> str:
    """ASIN -> point id. Re-derivable, no lookup needed."""
    return str(uuid.uuid5(ASIN_NAMESPACE, asin))


def load_env(path: str = ".env") -> None:
    """Minimal .env reader. Handles `KEY=value` and `export KEY=value`.

    Deliberately not python-dotenv: one less thing to install on a locked-down
    laptop. Existing environment variables always win.
    """
    f = Path(path)
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


def is_local(url: str) -> bool:
    return "localhost" in url or "127.0.0.1" in url


def make_client() -> tuple[QdrantClient, str]:
    """Build the client for whichever Qdrant QDRANT_URL points at.

    The only difference between the local podman fixture and the cloud cluster
    is the cloud_inference flag:

      cloud_inference=False -> models.Document(model="sentence-transformers/...")
                               is embedded on this machine by FastEmbed.
      cloud_inference=True  -> the Document is passed through untouched and
                               Qdrant Cloud Inference embeds it server-side.

    Either way build_query() is identical. That is the whole point.

    (models.Document(model="qdrant/bm25") is passed through in BOTH modes --
    Qdrant 1.19 computes BM25 in the server core. See docs/query-api.md.)
    """
    load_env()
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    local = is_local(url)
    client = QdrantClient(url=url, api_key=api_key, cloud_inference=not local, timeout=60)
    return client, url
