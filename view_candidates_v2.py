"""
View selected candidate memes per tweet.

Loads both tweet JSONL files separately to allow filtering
between direct and indirect tweet styles.

Run:
    python view_candidates.py
"""

import json
import random
from pathlib import Path
from flask import Flask, render_template_string, request

app = Flask(__name__)

TASKS_FILE = "annotation_tasks_clean.json"

# Map post_id -> tweet style by loading both source files
DIRECT_FILE = "train_memecap_tweets_clean.jsonl"
INDIRECT_FILE = "train_memecap_tweets_indirect_clean.jsonl"


def load_post_styles():
    """Build post_id+tweet_text -> style mapping from source files."""
    styles = {}
    for fname, style in [(DIRECT_FILE, "direct"), (INDIRECT_FILE, "indirect")]:
        path = Path(fname)
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                tweet = rec.get("tweet_that_meme_replies_to", "")
                if tweet:
                    styles[tweet] = style
    return styles


with open(TASKS_FILE, encoding="utf-8") as f:
    data = json.load(f)
META = data["metadata"]

# Tag each task with its style
tweet_styles = load_post_styles()
TASKS = []
for t in data["tasks"]:
    t["style"] = tweet_styles.get(t["tweet_text"], "unknown")
    TASKS.append(t)

direct_count = sum(1 for t in TASKS if t["style"] == "direct")
indirect_count = sum(1 for t in TASKS if t["style"] == "indirect")
print(f"Tagged {direct_count} direct, {indirect_count} indirect, {len(TASKS) - direct_count - indirect_count} unknown")

INDEX_HTML = """
<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>Candidate Viewer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; background: #f5f5f5; padding: 1rem; }
  h1 { margin-bottom: 0.5rem; }
  .meta { color: #666; margin-bottom: 1rem; font-size: 0.9rem; }
  .controls { display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; align-items: center; }
  .filter-btn {
    padding: 0.4rem 1rem; border: 2px solid #ddd; border-radius: 6px; background: #fff;
    cursor: pointer; font-size: 0.85rem; font-weight: 600;
  }
  .filter-btn:hover { border-color: #999; }
  .filter-btn.active { border-color: #1a1a2e; background: #1a1a2e; color: #fff; }
  .filter-btn.direct.active { border-color: #1da1f2; background: #1da1f2; color: #fff; }
  .filter-btn.indirect.active { border-color: #9b59b6; background: #9b59b6; color: #fff; }
  .random-btn {
    padding: 0.4rem 1rem; border: 2px solid #2ecc71; border-radius: 6px; background: #2ecc71;
    color: #fff; cursor: pointer; font-size: 0.85rem; font-weight: 600; margin-left: auto;
  }
  .random-btn:hover { background: #27ae60; }
  .tweet-list { max-width: 900px; }
  .tweet-card {
    background: #fff; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); cursor: pointer; text-decoration: none; color: inherit; display: block;
    border-left: 4px solid #ddd;
  }
  .tweet-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
  .tweet-card.direct { border-left-color: #1da1f2; }
  .tweet-card.indirect { border-left-color: #9b59b6; }
  .tweet-header { display: flex; justify-content: space-between; align-items: center; }
  .tweet-id { font-size: 0.75rem; color: #999; }
  .style-tag {
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase; padding: 0.1rem 0.4rem;
    border-radius: 3px; color: #fff;
  }
  .style-tag.direct { background: #1da1f2; }
  .style-tag.indirect { background: #9b59b6; }
  .tweet-text { margin-top: 0.3rem; }
  .search { padding: 0.5rem 1rem; width: 100%; max-width: 400px; border: 2px solid #ddd; border-radius: 8px; font-size: 1rem; }
  .count { font-size: 0.85rem; color: #888; margin-left: 0.5rem; }
</style>
</head><body>
<h1>Candidate Viewer</h1>
<p class="meta">{{ total }} tweets &middot; {{ meta.candidates_per_task }} candidates each &middot; {{ direct_count }} direct &middot; {{ indirect_count }} indirect</p>
<div class="controls">
  <button class="filter-btn active" onclick="setFilter('all')">All</button>
  <button class="filter-btn direct" onclick="setFilter('direct')">Direct</button>
  <button class="filter-btn indirect" onclick="setFilter('indirect')">Indirect</button>
  <input class="search" type="text" placeholder="Search tweets..." oninput="applyFilters()">
  <a class="random-btn" id="random-btn" href="/random">Random</a>
</div>
<div class="tweet-list" id="list">
{% for t in tasks %}
<a class="tweet-card {{ t.style }}" href="/task/{{ loop.index0 }}" data-text="{{ t.tweet_text | lower }}" data-style="{{ t.style }}">
  <div class="tweet-header">
    <span class="tweet-id">{{ t.task_id }}</span>
    <span class="style-tag {{ t.style }}">{{ t.style }}</span>
  </div>
  <div class="tweet-text">{{ t.tweet_text }}</div>
</a>
{% endfor %}
</div>
<script>
let currentFilter = 'all';
function setFilter(f) {
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`.filter-btn${f === 'all' ? ':first-child' : '.' + f}`).classList.add('active');
  // update random link
  document.getElementById('random-btn').href = f === 'all' ? '/random' : '/random?style=' + f;
  applyFilters();
}
function applyFilters() {
  const q = document.querySelector('.search').value.toLowerCase();
  document.querySelectorAll('.tweet-card').forEach(c => {
    const matchText = !q || c.dataset.text.includes(q);
    const matchStyle = currentFilter === 'all' || c.dataset.style === currentFilter;
    c.style.display = (matchText && matchStyle) ? '' : 'none';
  });
}
</script>
</body></html>
"""

TASK_HTML = """
<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>{{ task.task_id }} - Candidates</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; background: #f5f5f5; padding: 1rem; max-width: 1000px; }
  a { color: #1a1a2e; }
  .nav { margin-bottom: 1rem; font-size: 0.9rem; display: flex; gap: 1rem; }
  .tweet-box {
    background: #fff; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;
    border: 2px solid {% if task.style == 'indirect' %}#9b59b6{% else %}#1da1f2{% endif %};
  }
  .tweet-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
  .tweet-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: {% if task.style == 'indirect' %}#9b59b6{% else %}#1da1f2{% endif %}; }
  .style-tag {
    font-size: 0.65rem; font-weight: 700; text-transform: uppercase; padding: 0.15rem 0.5rem;
    border-radius: 3px; color: #fff;
    background: {% if task.style == 'indirect' %}#9b59b6{% else %}#1da1f2{% endif %};
  }
  .tweet-text { font-size: 1.05rem; }
  .candidates { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
  .cand {
    background: #fff; border-radius: 10px; padding: 0.75rem; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    text-align: center;
  }
  .cand.original { border: 3px solid #2ecc71; }
  .cand.semantic { border: 3px solid #f39c12; }
  .cand.random { border: 3px solid #bbb; }
  .cand img { max-width: 100%; max-height: 250px; border-radius: 6px; object-fit: contain; }
  .cand img.broken { min-height: 80px; background: #f0f0f0; border: 2px dashed #ccc; }
  .cand-title { font-size: 0.85rem; color: #555; margin-top: 0.4rem; font-style: italic; }
  .cand-id { font-size: 0.7rem; color: #999; }
  .badge {
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #fff; margin-top: 0.4rem;
  }
  .badge.original { background: #2ecc71; }
  .badge.semantic { background: #f39c12; }
  .badge.random { background: #bbb; }
  .nav-arrows { display: flex; gap: 1rem; margin-top: 1.5rem; }
  .nav-arrows a { padding: 0.5rem 1rem; background: #1a1a2e; color: #fff; border-radius: 6px; text-decoration: none; }
  .nav-arrows a.disabled { background: #ccc; pointer-events: none; }
  .nav-arrows a.random { background: #2ecc71; }
</style>
</head><body>
<div class="nav">
  <a href="/">&larr; Back to list</a>
  <a href="/random?style={{ task.style }}">Random {{ task.style }}</a>
</div>

<div class="tweet-box">
  <div class="tweet-header">
    <span class="tweet-label">Tweet ({{ task.task_id }})</span>
    <span class="style-tag">{{ task.style }}</span>
  </div>
  <p class="tweet-text">{{ task.tweet_text }}</p>
</div>

<div class="candidates">
{% for c in task.candidates %}
<div class="cand {{ c.selection_method }}">
  <img src="{{ c.image_url }}" alt="{{ c.title }}"
       onerror="this.onerror=null; this.src=''; this.alt='Image failed to load'; this.classList.add('broken');">
  <div class="cand-title">{{ c.title }}</div>
  <div class="cand-id">{{ c.meme_post_id }}</div>
  <div class="badge {{ c.selection_method }}">{{ c.selection_method }}</div>
</div>
{% endfor %}
</div>

<div class="nav-arrows">
  <a href="/task/{{ idx - 1 }}" {% if idx == 0 %}class="disabled"{% endif %}>&larr; Prev</a>
  <a href="/random?style={{ task.style }}" class="random">Random</a>
  <a href="/task/{{ idx + 1 }}" {% if idx >= total - 1 %}class="disabled"{% endif %}>Next &rarr;</a>
</div>
</body></html>
"""


@app.route("/")
def index():
    return render_template_string(
        INDEX_HTML, tasks=TASKS, meta=META, total=len(TASKS),
        direct_count=direct_count, indirect_count=indirect_count,
    )


@app.route("/task/<int:idx>")
def task(idx):
    if idx < 0 or idx >= len(TASKS):
        return "Not found", 404
    return render_template_string(TASK_HTML, task=TASKS[idx], idx=idx, total=len(TASKS))


@app.route("/random")
def random_task():
    style = request.args.get("style")
    if style and style in ("direct", "indirect"):
        filtered = [i for i, t in enumerate(TASKS) if t["style"] == style]
    else:
        filtered = list(range(len(TASKS)))
    if not filtered:
        return "No tasks found", 404
    from flask import redirect
    return redirect(f"/task/{random.choice(filtered)}")


if __name__ == "__main__":
    print(f"Loaded {len(TASKS)} tasks ({direct_count} direct, {indirect_count} indirect)")
    app.run(debug=True, port=5001)
