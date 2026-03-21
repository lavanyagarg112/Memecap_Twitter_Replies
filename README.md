# Meme Reply Selection Pipeline

## Quick Start

To view the pre-generated candidate meme pairings:

```
pip install flask
python view_candidates_v2.py
```
Open `http://localhost:5001`

## Full Pipeline Setup

```
pip install requests numpy sentence-transformers flask python-dotenv
```

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_key_here
```

## 1. Generate Synthetic Tweets

**Direct style** (tweet closely matches meme context):
```
python create_train_context_3.py
```
Output: `train_memecap_tweets.jsonl`

**Indirect style** (tweet is topically different, shares emotional energy):
```
python create_train_context_4.py
```
Output: `train_memecap_tweets_indirect.jsonl`

Both scripts generate 1 tweet per meme from the MemeCap trainval split (1500 memes).

## 2. Clean Tweet Files

Removes errors and keeps both files balanced (if a meme errors in either file, it's removed from both):
```
python clean_tweets.py
```
Output: `train_memecap_tweets_clean.jsonl`, `train_memecap_tweets_indirect_clean.jsonl`

## 3. Flag Offensive Memes

Interactive review — scans memes for offensive words, you decide which to exclude:
```
python flag_memes.py
```
Output: `flagged_memes.json`

You can quit with `q` and resume later — progress is saved.

## 4. Select Candidate Memes

For each tweet, selects 9 distractor memes (4 semantic + 5 random) plus the original:
```
python select_candidates_v2.py
```
Reads `flagged_memes.json` to exclude confirmed-offensive memes.
Output: `annotation_tasks_clean.json`

## 5. View Candidates

Browse tweet-meme pairings locally. Filter by direct/indirect style:
```
python view_candidates_v2.py
```
Open `http://localhost:5001`

## 6. Annotation App

Collect human judgments on whether memes are funny replies to tweets.

Create `annotation_app/.env`:
```
ADMIN_PASSWORD=yourpassword
```

Run:
```
cd annotation_app
python app.py
```
Open `http://localhost:5000`

- Annotators enter their name and start rating
- Each meme-tweet pair needs 5 annotations
- Annotators can stop anytime
- Admin dashboard at `/admin` (password required)
- Export annotations as CSV from dashboard
