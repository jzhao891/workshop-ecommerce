#!/usr/bin/env python3
"""Sealed workshop credentials.

The cluster URL and the read-only API key ship in this repo as `.env.enc`,
encrypted under a password the facilitator reads out when the session starts.
Nobody has to be emailed a key, and the key is not sitting in the repository in
plain text if it ever goes public.

Participants do not run this file. `python setup_check.py` calls it, asks for
the password once, and writes `.env`.

Facilitator, to produce `.env.enc` from a filled-in `.env`:

    python credentials.py --seal

Encryption is AES-256-GCM under a scrypt-derived key, and the format matches
the one used by the legal-retrieval lab so the two workshops behave the same.
Encrypt and decrypt live in this one file on purpose: split across two, they
drift, and a bundle that will not open on the day is unrecoverable.

This stops a key being scraped out of a public repo. It is not a secret-manager
and the password is shared with a room of forty people -- the real protection
is that the key is read-only, scoped to one collection, and expires.
"""
import argparse
import base64
import getpass
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "qdrant-workshop-ecommerce-credentials-v1"
AAD = FORMAT.encode()
SALT_BYTES, NONCE_BYTES, KEY_BYTES = 16, 12, 32
SCRYPT = dict(n=2**15, r=8, p=1)
REQUIRED = ("QDRANT_URL", "QDRANT_API_KEY")

ROOT = Path(__file__).resolve().parent
ENC, ENV = ROOT / ".env.enc", ROOT / ".env"


def _b64(b):  return base64.urlsafe_b64encode(b).decode()
def _unb64(s): return base64.urlsafe_b64decode(s.encode())


def _key(password: str, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    if not password:
        raise ValueError("password must not be empty")
    return Scrypt(salt=salt, length=KEY_BYTES, **SCRYPT).derive(password.encode())


def encrypt(plaintext: bytes, password: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, nonce = os.urandom(SALT_BYTES), os.urandom(NONCE_BYTES)
    ct = AESGCM(_key(password, salt)).encrypt(nonce, plaintext, AAD)
    return json.dumps({"format": FORMAT, "kdf": "scrypt-n32768-r8-p1",
                       "cipher": "aes-256-gcm", "salt": _b64(salt),
                       "nonce": _b64(nonce), "ciphertext": _b64(ct)},
                      indent=2, sort_keys=True).encode() + b"\n"


def decrypt(bundle: bytes, password: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    d = json.loads(bundle)
    if (d.get("format") != FORMAT or d.get("kdf") != "scrypt-n32768-r8-p1"
            or d.get("cipher") != "aes-256-gcm"):
        raise ValueError("unsupported credential bundle")
    salt, nonce, ct = _unb64(d["salt"]), _unb64(d["nonce"]), _unb64(d["ciphertext"])
    if len(salt) != SALT_BYTES or len(nonce) != NONCE_BYTES:
        raise ValueError("invalid credential bundle")
    return AESGCM(_key(password, salt)).decrypt(nonce, ct, AAD)


def key_expiry(api_key: str):
    """Qdrant cloud keys are JWTs. Return the expiry, or None if unreadable.

    Worth knowing before a room of forty people discovers it together.
    """
    try:
        payload = api_key.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload))["exp"]
        return datetime.fromtimestamp(exp, timezone.utc)
    except Exception:
        return None


def _parse_env(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip("'\"")
    return out


def _write_env(text: str) -> None:
    """Write .env with owner-only permissions, atomically."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=ROOT,
                                         prefix=".env.", delete=False) as f:
            f.write(text)
            tmp = f.name
        os.chmod(tmp, 0o600)
        os.replace(tmp, ENV)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def ensure_env() -> str | None:
    """Make sure .env exists. Returns a note to print, or None.

    Never overwrites an existing .env -- someone who has already set their own
    credentials keeps them.
    """
    if ENV.exists() or not ENC.exists():
        return None
    # flush: getpass writes straight to the terminal, so unflushed prints can
    # land after the prompt and read as if the password box came first.
    print("This repo ships its credentials sealed. The workshop password was",
          flush=True)
    print("given out at the start of the session; nothing echoes while you type.",
          flush=True)
    for attempt in (3, 2, 1):
        try:
            plaintext = decrypt(ENC.read_bytes(), getpass.getpass("Workshop password: "))
        except KeyboardInterrupt:
            sys.exit("\ncancelled")
        except Exception:
            if attempt > 1:
                print(f"  that password did not open the bundle, {attempt - 1} left")
                continue
            sys.exit("  that password did not open the bundle. Ask the facilitator "
                     "to read it out again -- it is not recoverable from this repo.")
        text = plaintext.decode()
        values = _parse_env(text)
        missing = [k for k in REQUIRED if not values.get(k)]
        if missing:
            sys.exit(f"the bundle opened but is missing {', '.join(missing)}. "
                     f"Tell the facilitator; this is not something you can fix.")
        _write_env(text)
        return "Credentials unlocked into .env, which is gitignored."


def _seal() -> None:
    if not ENV.exists():
        sys.exit(f"no {ENV.name} to seal. Write one with QDRANT_URL and "
                 f"QDRANT_API_KEY first.")
    text = ENV.read_text()
    values = _parse_env(text)
    missing = [k for k in REQUIRED if not values.get(k)]
    if missing:
        sys.exit(f".env is missing {', '.join(missing)}")

    exp = key_expiry(values["QDRANT_API_KEY"])
    if exp:
        days = (exp - datetime.now(timezone.utc)).days
        print(f"key expires {exp:%Y-%m-%d %H:%M UTC} ({days} days from now)")
        if days < 0:
            sys.exit("that key has already expired. Issue a new one before sealing.")
        if days < 14:
            print("  WARNING: that is inside two weeks. If the workshop is after "
                  "that date,\n  every participant fails at the same moment with a "
                  "key that looks correct.")
            if input("  seal it anyway? [y/N] ").strip().lower() != "y":
                sys.exit("not sealed")
    else:
        print("could not read an expiry from that key (not a JWT?) -- check it by hand")

    pw = getpass.getpass("Workshop password: ")
    if pw != getpass.getpass("Confirm: "):
        sys.exit("passwords did not match")
    if len(pw) < 6:
        sys.exit("pick something longer; a room has to type it correctly once")
    ENC.write_bytes(encrypt(text.encode(), pw))
    print(f"wrote {ENC.name} ({ENC.stat().st_size} bytes). Commit it. "
          f"Do NOT commit .env.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seal", action="store_true",
                    help="facilitator: encrypt .env into .env.enc")
    args = ap.parse_args()
    if args.seal:
        _seal()
    else:
        print(ensure_env() or "Nothing to do: .env already exists, or there is "
                              "no .env.enc to open.")
