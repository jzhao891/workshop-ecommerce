# E-commerce search workshop

You write one function. You have 90 minutes. Everything else in this repo exists
to score that function.

There are 100,000 real Amazon products — clothes, shoes, jewellery — already
loaded into a Qdrant database. Your job is to take what a shopper typed and turn
it into the best possible database query.

You don't need to have used vector search before. Nothing here assumes it.

If you brought a coding agent, point it at `AGENTS.md` first. If you didn't, or
your laptop blocks them, everything here works fine by hand.

---

## Before the workshop

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python setup_check.py
```

There's no API key to copy and paste. The database address and a read-only key
are already in this repo, encrypted. `setup_check.py` asks once for the workshop
password — the facilitator says it out loud at the start — and writes the
credentials to a `.env` file for you. Nothing shows on screen while you type the
password, and `.env` never gets committed.

Doing this early and don't have the password yet? Run the three lines anyway.
The check will tell you what it's waiting for.

When everything is fine you get one green line. When it isn't, you get the name
of the check that failed and what to do about it. That's the only file you need
to read to get unstuck. **Please run it before the session, not during it** —
forty people debugging their setup at once is how an hour disappears.

## What you actually write

One function, in `build_query.py`:

```python
def build_query(query_text: str, seed_asin: str | None) -> models.QueryRequest:
    ...
```

You **build a query and hand it back**. You don't run it, and you never see the
results. A test harness runs it for you and scores what comes back.

The file already contains a working version. It's deliberately basic — it works,
and it scores badly. Replace the inside of the function and keep the name and
arguments the same.

### Three rules

1. **One query per search.** You return a single request object. You can nest as
   much as you like inside it — that still counts as one.
2. **No sorting or filtering the results afterwards.** You never get the results,
   so there's nothing to sort anyway.
3. **No calling other services while building the query.** No embedding APIs, no
   lookups. (Asking Qdrant to turn your text into a vector doesn't count — that's
   part of the request, not a separate call you make.)

Edit `build_query.py`. Leave everything else alone.

## The catalogue

Every product is stored three different ways, and you choose which one to search.

| name | what it is |
|---|---|
| `dense` | The meaning of the title, as 384 numbers. Good at "sneakers" matching "trainers". Bad at exact codes. |
| `sparse` | Classic keyword matching (BM25). Nails exact words and model numbers. Useless when the shopper's words don't appear in the product title. |
| `image` | The product photo, as 512 numbers. Only 19,997 of the 100,000 products have one. |

Each product also carries fields you can filter on: `brand`, `category_path`,
`price`, `rating`, `review_count`, `in_stock`, `margin` and `asin`.

Note that the product **title is not filterable**. If you want to match words in
a title, that's what `sparse` is for.

You have a read-only key, so you can't add new fields or indexes — what's listed
above is what exists.

## Product IDs are not ASINs

Some questions hand you a `seed_asin` — an Amazon product code like
`B07XYZ1234`, meaning "find me more things like this one".

Qdrant doesn't store products under their ASIN. It uses an ID derived from it.
There's a helper that does the conversion with no database lookup:

```python
from common import point_id_for_asin
pid = point_id_for_asin(seed_asin)
```

Or you can skip IDs entirely and filter on the `asin` field instead:

```python
models.FieldCondition(key="asin", match=models.MatchValue(value=seed_asin))
```

Mixing the two up fails in two different ways, and only one is obvious. Using an
ASIN where an ID belongs gives you a **400 error** straight away. Using an ID
where the `asin` field is expected gives you **zero results and no error**, which
looks exactly like a query that just didn't find anything.

## What you can use

Roughly most useful first. Working examples for all of these are in
[`docs/query-api.md`](docs/query-api.md) — worth reading, because a few of them
will happily return a confident-looking list of wrong answers.

| tool | what it does |
|---|---|
| **Hybrid search** | Search `dense` and `sparse` at the same time and merge the results. Usually the single biggest improvement available. |
| **Filters** | Hard limits: under $50, in stock, this brand only. |
| **Search settings** | The vectors are compressed for speed. You can ask for a wider first pass and a more accurate re-check. |
| **Score formulas** | Adjust ranking using product fields — margin, review counts, stock. |
| **Decay functions** | "Prefer around $60" rather than "nothing over $60". |
| **Recommend** | More-like-this, starting from a product rather than text. |
| **Image search** | Visual similarity. Read the table above before using it. |
| **Weighted merging** | Make keyword matching count for more than meaning, or the reverse. |
| **MMR** | Stop the top ten being ten near-identical products. |

## Scoring

```bash
python harness.py              # score yourself
python harness.py --show dev03 # see the actual top 10 for one question
python harness.py --html       # see the results as product photos in your browser
```

You get a table like this — this is what the starter code scores, so it's your
floor:

```
segment              P@10    was  viol  stock  brands  n
------------------------------------------------------------
head_term           1.000            0   0.80    1.00  1
exact_model         0.667            0   0.50    0.17  3
constrained         0.000            4   0.68    0.72  4
natural_language    0.600            0   0.87    0.90  3
similar_item        0.900            0   0.90    0.10  1
long_tail           0.900            0   0.70    0.80  2
------------------------------------------------------------
OVERALL             0.536            4   0.71    0.63  14

build_query names 1 of the 3 vectors this collection carries (dense)
```

**The starter scores 0.536 and breaks four rules about price and stock. The best
we've measured is 0.721 with none broken.** A score below 1.0 isn't a bug —
nobody has reached it, and two whole categories of question still have obvious
room in them.

### Reading the columns

- **P@10** — out of your top ten results, how many were right. There's an answer
  key: 14 questions, each with a list of correct products.
- **viol** — questions where you broke a stated limit. If the shopper said "under
  $50" and one result costs $60, that whole question scores **zero**, no matter
  how good the other nine were. This is the harshest rule here and it's
  deliberate.
- **stock** — how much of your top ten is actually buyable.
- **brands** — how many different brands are in your top ten, ignoring the
  placeholder brand `"Unknown"`. A page of ten unbranded items shouldn't count as
  variety.
- **was** — the same number from your last run, shown only when it changed. So
  the loop is: change one thing, run it again, see which way it moved. No need to
  keep notes. Small wobbles between identical runs are normal — the search is
  approximate — so chase changes you can explain, not the last decimal place.

The line underneath counts how many of the three vectors your code even mentions.
It's a crude check, not a judgement, but one out of three is worth noticing.

`--html` is worth your time. Product search is visual. A sandal returned for a
boot query is obvious in a photo and invisible in a list of titles. It writes a
page and opens it: one row of product cards per question, **green** where the
answer key agrees with you, **red** where you broke a price or stock limit, with
the reason underneath.

### Where the answer key comes from

Each question has a **rule** rather than a human verdict. "Baseball caps" counts
anything filed under baseball caps; "under $20, in stock" adds those two
conditions. The rule is written next to every question in `dev_set.jsonl`, so you
can read it and disagree with it.

It's blunt. A genuinely good result that falls outside the rule is marked wrong.
That's the honest cost of scoring 100,000 products without a human reading them.

**There's a second, larger set of questions you don't get**, kept by the
facilitator. The 14 here teach you what's being measured; the hidden set checks
whether your changes actually help in general, or only on the questions you could
see. Tuning to these 14 specifically is visible from the outside.

### The six kinds of question

| kind | what it is |
|---|---|
| `head_term` | Broad category searches — "baseball caps". |
| `exact_model` | A brand and a model number — "CARTIER W51012Q4". |
| `constrained` | Limits in the sentence — "under $40, in stock only". |
| `natural_language` | Described, not named — "a watch I can wear swimming". |
| `similar_item` | "More like this one", starting from a product. |
| `long_tail` | Rare, specific phrasing — "merino wool base layer". |

**No single trick wins all six**, and the numbers say so bluntly. Turning on
hybrid search takes `exact_model` from 0.667 up to 1.000 — and takes
`natural_language` from 0.600 **down** to 0.300, because when a shopper describes
something in their own words, keyword matching has nothing to grab onto and drags
in junk. Meanwhile `head_term` sits at 1.000 whatever you do.

Whatever you apply to every question uniformly, you'll pay for one kind of
question with another.

## Running against a local copy instead

The cluster is shared by everyone in the room. If you'd rather work against your
own copy, there's a small local one — 2,000 made-up products in the same shape —
that runs in a container:

```bash
bash fixture/up.sh            # start it and fill it
# then put QDRANT_URL=http://localhost:6333 in your .env
python harness.py
```

Your `build_query` doesn't change between the two. The only difference is one
setting: locally your text gets turned into vectors on your laptop, and against
the cluster Qdrant does it. Same request either way.

**The local copy is for checking that your code runs, not whether it's any
good.** The products are invented and the images are random noise, so image
searches will execute and mean nothing.

There are two answer keys and they don't mix. `dev_set.jsonl` is the real one.
`fixture/dev_set.fixture.jsonl` is a stand-in for the local copy. The harness
picks whichever matches the database you're pointed at — using the real answer
key against the made-up products would score a flat zero and look like your bug.

```bash
bash fixture/up.sh --recreate   # wipe and refill
bash fixture/up.sh --down       # stop it, keep the data
python fixture/verify_api.py    # check the examples in docs/query-api.md still run
```

## When something breaks

- **"rate limited" or a 429** — forty people, one database. Wait a few seconds and
  run it again. It isn't your query. The harness deliberately only makes four
  requests at a time; please leave that alone.
- **An image search returns nothing, or nonsense** — `image` doesn't behave like
  the other two. Re-read its row in the table above, check which model built it,
  and check how many products actually have one.
- **"build_query returned X, expected models.QueryRequest"** — you returned
  results, a list, or nothing, instead of a query.
- **Anything else** — run `python setup_check.py` first.

## What's in here

| file | |
|---|---|
| `build_query.py` | **the only file you edit** |
| `README.md` / `AGENTS.md` | this, and the same thing written for coding agents |
| `docs/query-api.md` | what you can put in a query, with working examples |
| `harness.py` | runs the questions and scores you |
| `setup_check.py` | the pre-workshop check |
| `common.py` | database name, model names, the ASIN-to-ID helper |
| `dev_set.jsonl` | the 14 questions and their answer keys |
| `credentials.py` | unlocks the encrypted key; you don't run this directly |
| `fixture/` | the local 2,000-product copy |
