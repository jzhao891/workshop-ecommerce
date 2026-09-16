#!/usr/bin/env python3
"""Check your build_query.py, then hand it in.

    python submit.py

It runs the same structural checks the judge runs, copies your file to the
clipboard, and opens the submission form. Paste, add your name, submit.

The checks are the point. A submission that does not import, or returns the
wrong type, or only handles the constraint keys that happen to appear in the
questions you can see, scores zero for every question. Much better to find that
out here than after the deadline.
"""
import importlib.util
import platform
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

from qdrant_client.http import models as http_models

FORM = "https://forms.gle/9daJ1FQWvCo8pBn38"
HERE = Path(__file__).resolve().parent
TARGET = HERE / "build_query.py"

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

# Deliberately includes constraint keys the 14 visible questions never use. If
# your filter builder only handles max_price and in_stock, this is where you
# find out -- not from a zero on a question you never saw.
CASES = [
    ("waterproof hiking boots", None, None),
    ("baseball caps under $20, in stock only", None,
     {"max_price": 20.0, "in_stock": True}),
    ("Casio MTP1370D-1A2 Men's Black Dial Metal Watch", "B00RM95F16", None),
    ("running shoes", None, {"brand": "Nike", "min_rating": 4.0, "min_price": 25.0}),
]


def fail(problem: str, fix: str) -> None:
    print(f"{RED}Not submitted.{RESET}\n  problem: {problem}\n  fix:     {fix}")
    sys.exit(1)


def load():
    if not TARGET.exists():
        fail(f"no {TARGET.name} in {HERE}", "run this from the workshop repository")
    spec = importlib.util.spec_from_file_location("submission", TARGET)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        fail(f"{TARGET.name} does not import: {type(e).__name__}: {e}",
             "fix the error above, then run this again")
    fn = getattr(module, "build_query", None)
    if not callable(fn):
        fail("no build_query() function found",
             "keep the function named exactly build_query")
    return fn


def check(fn) -> None:
    """Build every case with the network switched off."""
    real_socket, real_connect = socket.socket, socket.create_connection

    def blocked(*_a, **_k):
        raise RuntimeError("network call")

    for query, seed, constraints in CASES:
        socket.socket, socket.create_connection = blocked, blocked
        try:
            req = fn(query, seed, constraints)
        except TypeError as e:
            if "positional argument" in str(e):
                fail("build_query takes the wrong number of arguments",
                     "the signature is build_query(query_text, seed_asin, constraints)")
            fail(f"build_query raised TypeError on {query!r}: {e}",
                 "fix the error above, then run this again")
        except RuntimeError as e:
            if "network call" in str(e):
                fail("build_query tried to open a network connection",
                     "rule 3: no network calls at query time. models.Document is a "
                     "field in the request, not a call you make.")
            fail(f"build_query raised on {query!r}: {e}", "fix the error above")
        except Exception as e:
            fail(f"build_query raised on {query!r}: {type(e).__name__}: {e}",
                 "fix the error above, then run this again")
        finally:
            socket.socket, socket.create_connection = real_socket, real_connect

        if not isinstance(req, http_models.QueryRequest):
            fail(f"build_query returned {type(req).__name__} for {query!r}, "
                 f"expected models.QueryRequest",
                 "return the request object itself, not results and not a list")
        if (req.limit or 10) < 10:
            print(f"{YELLOW}note{RESET}: limit={req.limit} on {query!r} — "
                  f"precision@10 cannot reach 1.0 with fewer than 10 results")

    # The fourth case carries brand, min_rating and min_price. A filter builder
    # that ignores them is legal, and quietly loses points on questions the
    # visible 14 never show you, so say so rather than failing.
    last = fn(*CASES[3])
    keys = str(last.filter) if last.filter else ""
    missed = [k for k in ("brand", "rating", "price") if k not in keys]
    if missed:
        print(f"{YELLOW}note{RESET}: with constraints "
              f"{{brand, min_rating, min_price}} your filter does not mention "
              f"{', '.join(missed)}. That is allowed, but the hidden questions "
              f"can use any of the five keys.")


def to_clipboard(text: str) -> bool:
    cmds = {"Darwin": ["pbcopy"], "Windows": ["clip"]}
    for cmd in [cmds.get(platform.system())] if platform.system() in cmds else \
               [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-ib"]]:
        try:
            subprocess.run(cmd, input=text.encode(), check=True)
            return True
        except Exception:
            continue
    return False


def main() -> None:
    fn = load()
    check(fn)
    print(f"{GREEN}build_query.py looks good{RESET} — imports, correct signature, "
          f"returns a QueryRequest on all {len(CASES)} shapes, no network calls.")

    source = TARGET.read_text()
    if to_clipboard(source):
        print("Copied to your clipboard.")
    else:
        print(f"Could not reach the clipboard. Copy the contents of {TARGET} by hand.")

    print(f"\nOpening the form: {FORM}")
    print("  Paste into 'Your build_query.py code', add your name, submit.")
    print("  You can submit as many times as you like — the last one counts.")
    try:
        webbrowser.open(FORM)
    except Exception:
        pass


if __name__ == "__main__":
    main()
