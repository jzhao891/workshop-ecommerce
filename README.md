# E-Commerce Search Workshop

You write one function. You have 90 minutes. Everything else in this repository
exists to score that function.

A database holds 100,000 real Amazon products: clothes, shoes, and jewelry. Your
job is to take what a shopper typed and turn it into the best possible search
query.

You don't need any experience with vector search. This guide explains each idea
as it comes up.

If you brought a coding agent, point it at `AGENTS.md` first. If you didn't, or
your laptop blocks them, everything here works fine by hand.

---

## Before the Workshop

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python setup_check.py
```

There's no API key to copy and paste. The database address and a read-only key
already sit in this repository, encrypted. The setup check asks once for the
workshop password, which the facilitator reads out at the start, and then writes
the credentials into a `.env` file for you. Nothing appears on screen while you
type the password, and `.env` is never committed.

Running this early, before you have the password? Run the three lines anyway.
The check tells you what it's waiting for.

When everything works, you get one green line. When something's wrong, you get
the name of the check that failed and the fix for it. That's the only file you
need to read to get unstuck.

**Please run this before the session, not during it.** Forty people debugging
their setup at the same time is how an hour disappears.

## What You Write

One function, in `build_query.py`:

```python
def build_query(query_text: str, seed_asin: str | None) -> models.QueryRequest:
    ...
```

You **build a query and hand it back**. You don't run it, and you never see the
results. A scoring program runs it for you and grades what comes back.

The file already contains a working version. It's deliberately basic: it runs,
and it scores poorly. Replace the body of the function, and keep the name and
arguments the same.

### Three Rules

1. **One query per search.** You return a single request. You can nest as much as
   you like inside it, and that still counts as one.
2. **Don't sort or filter the results afterward.** You never receive the results,
   so there's nothing to sort.
3. **Don't call other services while building the query.** No embedding services,
   no lookups. Asking the database to turn your text into numbers doesn't count,
   because that happens inside the request rather than as a separate call.

Edit `build_query.py`, and leave every other file alone.

## How Products Are Stored

A **vector** is a list of numbers that represents something. Similar things get
similar numbers, so the database can find related products by comparing lists of
numbers instead of matching words.

Every product is stored three different ways, and you choose which one to search.

| Name | What It Is |
|---|---|
| `dense` | The meaning of the product title, stored as 384 numbers. Finds "sneakers" when a shopper types "trainers." Struggles with exact model codes. |
| `sparse` | Traditional keyword matching, using a ranking method called BM25. Matches exact words and model numbers reliably. Finds nothing when the shopper's words don't appear in the title. |
| `image` | The product photo, stored as 512 numbers. Only 19,997 of the 100,000 products have one. |

Every product also carries fields you can filter on: `brand`, `category_path`,
`price`, `rating`, `review_count`, `in_stock`, `margin`, and `asin`.

The product **title isn't filterable**. To match words in a title, search the
`sparse` vector instead.

Your key is read-only, so you can't add fields or indexes. What's listed here is
what exists.

## Product IDs and ASINs

Some questions hand you a `seed_asin`. An ASIN, or Amazon Standard
Identification Number, is a product code such as `B07XYZ1234`. A question with
one is asking you to find products similar to that one.

The database doesn't store products under their ASIN. It uses an internal ID
derived from it. A helper converts between them without touching the database:

```python
from common import point_id_for_asin
pid = point_id_for_asin(seed_asin)
```

You can also skip IDs entirely and filter on the `asin` field:

```python
models.FieldCondition(key="asin", match=models.MatchValue(value=seed_asin))
```

Mixing the two up fails in two different ways, and only one is obvious:

- An ASIN where an internal ID belongs returns a **400 error** right away.
- An internal ID where the `asin` field belongs returns **zero results and no
  error**, which looks exactly like a search that found nothing.

## What You Can Use

These are roughly ordered by how much they tend to help. Working examples for all
of them live in [the Query API reference](docs/query-api.md), which is worth
reading, because several will cheerfully return a confident list of wrong
answers.

| Tool | What It Does |
|---|---|
| **Hybrid Search** | Searches `dense` and `sparse` at the same time and merges the results. Usually the single biggest improvement available. |
| **Filters** | Hard limits, such as under $50, in stock, or one specific brand. |
| **Search Settings** | The vectors are compressed for speed. You can ask for a wider first pass and a more accurate second look. |
| **Score Formulas** | Adjust the ranking using product fields such as margin, review count, or stock. |
| **Decay Functions** | Express "prefer around $60" rather than "nothing over $60." |
| **Recommend** | Find more products like a given one, starting from a product instead of text. |
| **Image Search** | Visual similarity. Read the storage table before using it. |
| **Weighted Merging** | Make keyword matching count for more than meaning, or the reverse. |
| **Maximal Marginal Relevance** | Stops the top 10 from being 10 nearly identical products. |

## Scoring

```bash
python harness.py               # score yourself
python harness.py --show dev03  # see the top 10 for one question
python harness.py --html        # see the results as product photos in a browser
```

You get a table like this one. These are the scores of the starter code, so
they're your floor:

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

**The starter code scores 0.536 and breaks four limits on price and stock. The
best score we've measured is 0.721, with none broken.** A total under 1.0 isn't a
bug. Nobody has reached 1.0, and two kinds of question still have clear room in
them.

### Reading the Scorecard

- **P@10**, short for precision at 10, counts how many of your top 10 results
  were correct. There's an answer key: 14 questions, each with a list of correct
  products.
- **viol** counts questions where you broke a stated limit. If the shopper asked
  for "under $50" and one result costs $60, that entire question scores **zero**,
  however good the other nine results were. This is the harshest rule here, and
  it's deliberate.
- **stock** shows how much of your top 10 a shopper could buy today.
- **brands** counts how many different brands appear in your top 10, ignoring the
  placeholder brand `"Unknown"`. Ten unbranded items shouldn't count as variety.
- **was** repeats the same number from your previous run, and appears only when
  the number changed. The loop is: change one thing, run it again, and see which
  way it moved. You don't need to keep notes. Small wobbles between identical
  runs are normal, because the search is approximate, so follow changes you can
  explain rather than the last decimal place.

The line after the table counts how many of the three vectors your code mentions.
It's a rough check rather than a judgment, but one out of three is worth
noticing.

The `--html` view is worth your time. Product search is visual. A sandal returned
for a boot query is obvious in a photo and invisible in a list of titles. The
command writes a page and opens it, showing one row of product cards per
question. Correct results have a **green** border, results that break a price or
stock limit have a **red** border, and each one states the reason in text as
well, so the page reads correctly without color.

### Where the Answer Key Comes From

Each question carries a **rule** rather than a human verdict. The question
"baseball caps" counts anything filed under baseball caps. Adding "under $20, in
stock" counts those two conditions as well. Every rule is written next to its
question in `dev_set.jsonl`, so you can read it and disagree with it.

The rules are blunt. A genuinely good result that falls outside a rule is marked
wrong. That's the honest cost of scoring 100,000 products without a person
reading them.

**A second, larger set of questions stays with the facilitator.** The 14 here
teach you what's being measured. The hidden set checks whether your changes help
in general, or only on the questions you could see. Tuning to these 14 questions
specifically is visible from the outside.

### The Six Kinds of Question

| Kind | What It Is |
|---|---|
| `head_term` | A broad category search, such as "baseball caps." |
| `exact_model` | A brand and a model number, such as "CARTIER W51012Q4." |
| `constrained` | Limits stated in the sentence, such as "under $40, in stock only." |
| `natural_language` | Described rather than named, such as "a watch I can wear swimming." |
| `similar_item` | "More like this one," starting from a product. |
| `long_tail` | Rare, specific wording, such as "merino wool base layer." |

**No single approach wins all six**, and the measurements say so plainly. Turning
on hybrid search lifts `exact_model` from 0.667 to 1.000. The same change drops
`natural_language` from 0.600 to **0.300**, because when a shopper describes
something in their own words, keyword matching has nothing to match and pulls in
irrelevant products. Meanwhile `head_term` stays at 1.000 whatever you do.

Anything you apply to every question uniformly will pay for one kind of question
with another.

## Running Against a Local Copy

Everyone in the room shares one database. If you'd rather work against your own
copy, a smaller one runs in a container on your laptop, holding 2,000 invented
products in the same shape:

```bash
bash fixture/up.sh            # start it and fill it
# then set QDRANT_URL=http://localhost:6333 in your .env file
python harness.py
```

Your `build_query` function doesn't change between the two. The only difference
is one setting: on your laptop, your text becomes numbers locally, and against
the shared database, the server does that work. The request is the same either
way.

**The local copy checks that your code runs, not whether it's any good.** The
products are invented and the images are random noise, so image searches will
run and mean nothing.

There are two answer keys, and they don't mix. `dev_set.jsonl` is the real one.
`fixture/dev_set.fixture.jsonl` is a stand-in for the local copy. The scoring
program picks whichever matches the database you point it at, because using the
real answer key against invented products scores a flat zero and looks like a bug
in your code.

```bash
bash fixture/up.sh --recreate   # empty it and refill it
bash fixture/up.sh --down       # stop it and keep the data
python fixture/verify_api.py    # check the reference examples still run
```

## When Something Breaks

- **A "rate limited" message, or error 429.** Forty people share one database.
  Wait a few seconds and run it again. This isn't a problem with your query. The
  scoring program deliberately sends only four requests at a time, so please
  leave that setting alone.
- **An image search returns nothing, or nonsense.** The `image` vector doesn't
  behave like the other two. Reread its row in the storage table, check which
  model built it, and check how many products have one.
- **"build_query returned X, expected models.QueryRequest".** You returned
  results, a list, or nothing, instead of a query.
- **Anything else.** Run `python setup_check.py` first.

## What's in This Repository

| File | Purpose |
|---|---|
| `build_query.py` | **The only file you edit.** |
| `README.md` and `AGENTS.md` | This guide, and the same guide written for coding agents. |
| `docs/query-api.md` | What you can put in a query, with working examples. |
| `harness.py` | Runs the questions and scores you. |
| `setup_check.py` | The check to run before the workshop. |
| `common.py` | Database name, model names, and the ASIN-to-ID helper. |
| `dev_set.jsonl` | The 14 questions and their answer keys. |
| `credentials.py` | Unlocks the encrypted key. You don't run this directly. |
| `fixture/` | The local copy of 2,000 products. |
