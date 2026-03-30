"""
Visualise tweet + original meme + top-10 candidate memes from the
non-annotation pipeline output.

Run:
    python view_rankings.py                          # default file
    python view_rankings.py --file other_output.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from flask import Flask, redirect, render_template_string, request

app = Flask(__name__)

RANKINGS = []  # loaded at startup


def load_rankings(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


INDEX_HTML = """
<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>Meme Rankings Viewer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; background: #f5f5f5; padding: 1rem; }
  h1 { margin-bottom: 0.3rem; }
  .meta { color: #666; margin-bottom: 1rem; font-size: 0.9rem; }
  .controls { display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; align-items: center; }
  .search {
    padding: 0.5rem 1rem; width: 100%; max-width: 400px;
    border: 2px solid #ddd; border-radius: 8px; font-size: 1rem;
  }
  .random-btn {
    padding: 0.4rem 1rem; border: 2px solid #2ecc71; border-radius: 6px;
    background: #2ecc71; color: #fff; cursor: pointer; font-size: 0.85rem;
    font-weight: 600; margin-left: auto; text-decoration: none;
  }
  .random-btn:hover { background: #27ae60; }
  .tweet-list { max-width: 900px; }
  .tweet-card {
    background: #fff; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); cursor: pointer; text-decoration: none;
    color: inherit; display: block; border-left: 4px solid #1da1f2;
  }
  .tweet-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
  .tweet-header { display: flex; justify-content: space-between; align-items: center; }
  .tweet-id { font-size: 0.75rem; color: #999; }
  .score-tag {
    font-size: 0.7rem; font-weight: 700; padding: 0.1rem 0.4rem;
    border-radius: 3px; background: #1da1f2; color: #fff;
  }
  .tweet-text { margin-top: 0.3rem; }
</style>
</head><body>
<h1>Meme Rankings Viewer</h1>
<p class="meta">{{ total }} tweets &middot; top-{{ top_k }} candidates each</p>
<div class="controls">
  <input class="search" type="text" placeholder="Search tweets..." oninput="filter()">
  <a class="random-btn" href="/random">Random</a>
</div>
<div class="tweet-list" id="list">
{% for r in rankings %}
<a class="tweet-card" href="/tweet/{{ loop.index0 }}" data-text="{{ r.post_text | lower }}">
  <div class="tweet-header">
    <span class="tweet-id">{{ r.tweet_id }}</span>
    <span class="score-tag">top: {{ "%.3f"|format(r.top_memes[0].similarity_score) if r.top_memes else "?" }}</span>
  </div>
  <div class="tweet-text">{{ r.post_text[:200] }}{% if r.post_text|length > 200 %}...{% endif %}</div>
</a>
{% endfor %}
</div>
<script>
function filter() {
  const q = document.querySelector('.search').value.toLowerCase();
  document.querySelectorAll('.tweet-card').forEach(c => {
    c.style.display = (!q || c.dataset.text.includes(q)) ? '' : 'none';
  });
}
</script>
</body></html>
"""

TWEET_HTML = """
<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>{{ rec.tweet_id }} - Rankings</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; background: #f5f5f5; padding: 1rem; max-width: 1100px; }
  a { color: #1a1a2e; }
  .nav { margin-bottom: 1rem; font-size: 0.9rem; display: flex; gap: 1rem; }
  .tweet-box {
    background: #fff; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1rem;
    border: 2px solid #1da1f2;
  }
  .tweet-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #1da1f2; }
  .tweet-text { font-size: 1.05rem; margin-top: 0.3rem; }
  .tweet-desc {
    font-size: 0.85rem; color: #666; margin-top: 0.5rem; padding-top: 0.5rem;
    border-top: 1px solid #eee; font-style: italic;
  }
  .tweet-desc strong { color: #333; font-style: normal; }

  .original-box {
    background: #fff; border-radius: 10px; padding: 1rem; margin-bottom: 1.5rem;
    border: 3px solid #2ecc71; text-align: center;
  }
  .original-box h3 { color: #2ecc71; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 0.5rem; }
  .original-box img {
    max-width: 100%; max-height: 350px; border-radius: 6px; object-fit: contain;
  }
  .original-box img.broken { min-height: 80px; background: #f0f0f0; border: 2px dashed #ccc; }

  h2 { margin-bottom: 0.75rem; font-size: 1.1rem; }
  .candidates { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
  .cand {
    background: #fff; border-radius: 10px; padding: 0.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center; position: relative;
  }
  .cand img {
    max-width: 100%; max-height: 250px; border-radius: 6px; object-fit: contain;
  }
  .cand img.broken { min-height: 80px; background: #f0f0f0; border: 2px dashed #ccc; }
  .cand-title { font-size: 0.85rem; color: #555; margin-top: 0.4rem; font-style: italic; }
  .cand-id { font-size: 0.7rem; color: #999; }
  .rank-badge {
    position: absolute; top: 0.5rem; left: 0.5rem; background: #1a1a2e; color: #fff;
    width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 0.8rem; font-weight: 700;
  }
  .score {
    display: inline-block; margin-top: 0.3rem; padding: 0.15rem 0.5rem; border-radius: 4px;
    font-size: 0.75rem; font-weight: 700; color: #fff;
  }
  .score.high { background: #2ecc71; }
  .score.mid  { background: #f39c12; }
  .score.low  { background: #e74c3c; }
  .captions { font-size: 0.75rem; color: #888; margin-top: 0.3rem; text-align: left; }
  .captions span { display: block; margin-top: 0.15rem; }

  .nav-arrows { display: flex; gap: 1rem; margin-top: 1.5rem; }
  .nav-arrows a {
    padding: 0.5rem 1rem; background: #1a1a2e; color: #fff;
    border-radius: 6px; text-decoration: none;
  }
  .nav-arrows a.disabled { background: #ccc; pointer-events: none; }
  .nav-arrows a.random { background: #2ecc71; }
</style>
</head><body>
<div class="nav">
  <a href="/">&larr; Back to list</a>
  <a href="/random">Random</a>
</div>

<div class="tweet-box">
  <span class="tweet-label">Tweet ({{ rec.tweet_id }})</span>
  <p class="tweet-text">{{ rec.post_text }}</p>
  {% if rec.tweet_description %}
  <div class="tweet-desc"><strong>VLM description:</strong> {{ rec.tweet_description }}</div>
  {% endif %}
</div>

{% if rec.img_link %}
<div class="original-box">
  <h3>Original Meme (from tweet)</h3>
  <img src="{{ rec.img_link }}" alt="Original meme"
       onerror="this.onerror=null; this.src=''; this.alt='Image unavailable'; this.classList.add('broken');">
</div>
{% endif %}

<h2>Top {{ rec.top_memes | length }} Candidate Memes (MemeCap)</h2>
<div class="candidates">
{% for m in rec.top_memes %}
<div class="cand">
  <div class="rank-badge">{{ m.rank }}</div>
  <img src="{{ m.image_url }}" alt="{{ m.title }}"
       onerror="this.onerror=null; this.src=''; this.alt='Image failed to load'; this.classList.add('broken');">
  <div class="cand-title">{{ m.title }}</div>
  <div class="cand-id">{{ m.memecap_post_id }}</div>
  {% set s = m.similarity_score %}
  <span class="score {{ 'high' if s >= 0.5 else ('mid' if s >= 0.3 else 'low') }}">{{ "%.3f"|format(s) }}</span>
  {% if m.meme_captions %}
  <div class="captions">
    {% for cap in m.meme_captions[:2] %}
    <span>{{ cap }}</span>
    {% endfor %}
  </div>
  {% endif %}
</div>
{% endfor %}
</div>

<div class="nav-arrows">
  <a href="/tweet/{{ idx - 1 }}" {% if idx == 0 %}class="disabled"{% endif %}>&larr; Prev</a>
  <a href="/random" class="random">Random</a>
  <a href="/tweet/{{ idx + 1 }}" {% if idx >= total - 1 %}class="disabled"{% endif %}>Next &rarr;</a>
</div>
</body></html>
"""


@app.route("/")
def index():
    top_k = len(RANKINGS[0]["top_memes"]) if RANKINGS and RANKINGS[0].get("top_memes") else 0
    return render_template_string(INDEX_HTML, rankings=RANKINGS, total=len(RANKINGS), top_k=top_k)


@app.route("/tweet/<int:idx>")
def tweet(idx):
    if idx < 0 or idx >= len(RANKINGS):
        return "Not found", 404
    return render_template_string(TWEET_HTML, rec=RANKINGS[idx], idx=idx, total=len(RANKINGS))


@app.route("/random")
def random_tweet():
    if not RANKINGS:
        return "No data", 404
    return redirect(f"/tweet/{random.randrange(len(RANKINGS))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise meme ranking results.")
    parser.add_argument("--file", default="twitter_meme_rankings.jsonl",
                        help="Path to rankings JSONL")
    parser.add_argument("--port", type=int, default=5002,
                        help="Port (default 5002, avoids clash with annotation app)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: {path} not found. Run rank_similar_memes.py first.")
        exit(1)

    RANKINGS = load_rankings(str(path))
    print(f"Loaded {len(RANKINGS)} ranked tweets from {path}")
    app.run(debug=True, port=args.port)
