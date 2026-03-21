"""
Meme Annotation Web App

A simple Flask app for collecting human judgments on whether
a meme is a funny reply to a tweet.

Reads from annotation_tasks.json (output of select_candidates.py).

Install:
    pip install flask python-dotenv

Run:
    cd annotation_app
    python app.py

Or:
    FLASK_APP=annotation_app/app.py flask run
"""

import csv
import io
import json
import os
import random
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from flask import (
    Flask,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# =========================
# CONFIG
# =========================

TASKS_FILE = os.environ.get(
    "TASKS_FILE",
    str(Path(__file__).resolve().parent.parent / "annotation_tasks_clean.json"),
)
DB_PATH = os.environ.get(
    "DB_PATH",
    str(Path(__file__).resolve().parent / "annotations.db"),
)
ANNOTATIONS_REQUIRED = 5  # per item
MAX_PER_ANNOTATOR = int(os.environ.get("MAX_PER_ANNOTATOR", 100))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1000))  # items per batch
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")
if not ADMIN_PASSWORD:
    raise ValueError("Set ADMIN_PASSWORD in annotation_app/.env")
PORT = 5000


# =========================
# DATABASE
# =========================

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create tables and populate items from annotation_tasks.json."""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")

    db.executescript("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            candidate_index INTEGER NOT NULL,
            tweet_text TEXT NOT NULL,
            meme_post_id TEXT NOT NULL,
            image_url TEXT NOT NULL,
            meme_title TEXT NOT NULL,
            selection_method TEXT NOT NULL,
            annotation_count INTEGER DEFAULT 0,
            skip_count INTEGER DEFAULT 0,
            batch_num INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id INTEGER NOT NULL REFERENCES items(id),
            annotator TEXT NOT NULL,
            is_funny INTEGER NOT NULL,
            flag_inappropriate INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(item_id, annotator)
        );

        CREATE INDEX IF NOT EXISTS idx_annotations_item
            ON annotations(item_id);
        CREATE INDEX IF NOT EXISTS idx_annotations_annotator
            ON annotations(annotator);
        CREATE INDEX IF NOT EXISTS idx_items_count
            ON items(annotation_count);
    """)

    # Check if items already populated
    count = db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    if count > 0:
        print(f"Database already has {count} items. Skipping import.")
        db.close()
        return

    # Load tasks
    if not Path(TASKS_FILE).exists():
        print(f"Warning: {TASKS_FILE} not found. Run select_candidates.py first.")
        db.close()
        return

    with open(TASKS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    random.seed(42)
    random.shuffle(tasks)
    inserted = 0
    items_per_task = 10  # candidates per tweet
    tasks_per_batch = BATCH_SIZE // items_per_task
    for ti, task in enumerate(tasks):
        task_id = f"{task['task_id']}_{ti}"
        tweet_text = task["tweet_text"]
        batch_num = ti // tasks_per_batch + 1
        for ci, cand in enumerate(task["candidates"]):
            db.execute(
                """INSERT INTO items
                   (task_id, candidate_index, tweet_text, meme_post_id,
                    image_url, meme_title, selection_method, batch_num)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_id,
                    ci,
                    tweet_text,
                    cand["meme_post_id"],
                    cand["image_url"],
                    cand["title"],
                    cand["selection_method"],
                    batch_num,
                ),
            )
            inserted += 1

    db.commit()
    total_batches = (len(tasks) + tasks_per_batch - 1) // tasks_per_batch if tasks else 0
    print(f"Imported {inserted} items from {len(tasks)} tasks into {total_batches} batches.")
    db.close()


# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    if "annotator" in session:
        return redirect(url_for("annotate"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    name = request.form.get("annotator", "").strip()
    if not name:
        return redirect(url_for("index"))
    session["annotator"] = name
    return redirect(url_for("annotate"))


@app.route("/logout")
def logout():
    session.pop("annotator", None)
    return redirect(url_for("index"))


@app.route("/annotate")
def annotate():
    if "annotator" not in session:
        return redirect(url_for("index"))

    annotator = session["annotator"]
    db = get_db()
    skipped_later = session.get("skipped_later", [])

    # Count real annotations (not skips) for cap check
    done_count = db.execute(
        "SELECT COUNT(*) FROM annotations WHERE annotator = ? AND is_funny != -1",
        (annotator,),
    ).fetchone()[0]

    if done_count >= MAX_PER_ANNOTATOR:
        return render_template("done.html", annotator=annotator, done_count=done_count,
                               hit_cap=True, cap=MAX_PER_ANNOTATOR)

    # Find the current batch: the lowest batch_num that still has
    # items needing annotations.
    current_batch = db.execute(
        """SELECT MIN(i.batch_num) FROM items i
           WHERE i.annotation_count < ?
             AND NOT (i.annotation_count < 3 AND i.skip_count >= ?)""",
        (ANNOTATIONS_REQUIRED, ANNOTATIONS_REQUIRED),
    ).fetchone()[0]

    if current_batch is None:
        return render_template("done.html", annotator=annotator, done_count=done_count)

    # Get next item from the current batch
    query = """SELECT i.* FROM items i
           WHERE i.batch_num = ?
             AND i.annotation_count < ?
             AND NOT (i.annotation_count < 3 AND i.skip_count >= ?)
             AND i.id NOT IN (
                 SELECT item_id FROM annotations WHERE annotator = ?
             )"""
    params = [current_batch, ANNOTATIONS_REQUIRED, ANNOTATIONS_REQUIRED, annotator]

    if skipped_later:
        placeholders = ",".join("?" * len(skipped_later))
        query += f" AND i.id NOT IN ({placeholders})"
        params.extend(skipped_later)

    query += " ORDER BY i.annotation_count ASC, RANDOM() LIMIT 1"
    item = db.execute(query, params).fetchone()

    if item is None:
        return render_template("done.html", annotator=annotator, done_count=done_count)

    total_remaining = db.execute(
        """SELECT COUNT(*) FROM items
           WHERE annotation_count < ?
             AND NOT (annotation_count < 3 AND skip_count >= ?)
             AND id NOT IN (
                 SELECT item_id FROM annotations WHERE annotator = ?
             )""",
        (ANNOTATIONS_REQUIRED, ANNOTATIONS_REQUIRED, annotator),
    ).fetchone()[0]

    return render_template(
        "annotate.html",
        item=item,
        annotator=annotator,
        done_count=done_count,
        cap=MAX_PER_ANNOTATOR,
        remaining=total_remaining,
    )


@app.route("/skip_later", methods=["POST"])
def skip_later():
    """Skip this item for now — it stays available for later or other annotators."""
    if "annotator" not in session:
        return redirect(url_for("index"))
    item_id = request.form.get("item_id", type=int)
    if item_id is not None:
        skipped = session.get("skipped_later", [])
        if item_id not in skipped:
            skipped.append(item_id)
            session["skipped_later"] = skipped
    return redirect(url_for("annotate"))


@app.route("/submit", methods=["POST"])
def submit():
    if "annotator" not in session:
        return redirect(url_for("index"))

    annotator = session["annotator"]
    item_id = request.form.get("item_id", type=int)
    is_funny = request.form.get("is_funny", type=int)
    flag_inappropriate = 1 if request.form.get("flag_inappropriate") else 0

    if item_id is None or is_funny not in (0, 1, -1):
        return redirect(url_for("annotate"))

    db = get_db()

    # Check not already annotated by this user
    existing = db.execute(
        "SELECT id FROM annotations WHERE item_id = ? AND annotator = ?",
        (item_id, annotator),
    ).fetchone()

    if existing is None:
        db.execute(
            "INSERT INTO annotations (item_id, annotator, is_funny, flag_inappropriate) VALUES (?, ?, ?, ?)",
            (item_id, annotator, is_funny, flag_inappropriate),
        )
        # Real annotations count toward required total; skips tracked separately
        if is_funny != -1:
            db.execute(
                "UPDATE items SET annotation_count = annotation_count + 1 WHERE id = ?",
                (item_id,),
            )
        else:
            db.execute(
                "UPDATE items SET skip_count = skip_count + 1 WHERE id = ?",
                (item_id,),
            )
        db.commit()

    # Store last annotation for undo
    session["last_annotation_id"] = item_id

    return redirect(url_for("annotate"))


@app.route("/undo", methods=["POST"])
def undo():
    if "annotator" not in session:
        return redirect(url_for("index"))

    annotator = session["annotator"]
    item_id = session.pop("last_annotation_id", None)

    if item_id is None:
        return redirect(url_for("annotate"))

    db = get_db()

    # Find the annotation to check if it was a real vote or a skip
    ann = db.execute(
        "SELECT is_funny FROM annotations WHERE item_id = ? AND annotator = ?",
        (item_id, annotator),
    ).fetchone()

    if ann is not None:
        was_skip = ann["is_funny"] == -1
        db.execute(
            "DELETE FROM annotations WHERE item_id = ? AND annotator = ?",
            (item_id, annotator),
        )
        if was_skip:
            db.execute(
                "UPDATE items SET skip_count = MAX(skip_count - 1, 0) WHERE id = ?",
                (item_id,),
            )
        else:
            db.execute(
                "UPDATE items SET annotation_count = MAX(annotation_count - 1, 0) WHERE id = ?",
                (item_id,),
            )
        db.commit()

    return redirect(url_for("annotate"))


@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["is_admin"] = True
            next_page = request.args.get("next", "dashboard")
            return redirect(url_for(next_page))
        return render_template("admin_login.html", error="Wrong password")
    session.pop("is_admin", None)
    return render_template("admin_login.html")


@app.route("/dashboard")
def dashboard():
    if not session.pop("is_admin", None):
        return redirect(url_for("admin_login", next="dashboard"))
    db = get_db()

    total_items = db.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    total_annotations = db.execute(
        "SELECT COUNT(*) FROM annotations WHERE is_funny != -1"
    ).fetchone()[0]
    total_skips = db.execute(
        "SELECT COUNT(*) FROM annotations WHERE is_funny = -1"
    ).fetchone()[0]
    completed_items = db.execute(
        "SELECT COUNT(*) FROM items WHERE annotation_count >= ?",
        (ANNOTATIONS_REQUIRED,),
    ).fetchone()[0]
    broken_items = db.execute(
        "SELECT COUNT(*) FROM items WHERE annotation_count < 3 AND skip_count >= ?",
        (ANNOTATIONS_REQUIRED,),
    ).fetchone()[0]

    # Per-annotator stats
    annotators = db.execute(
        """SELECT annotator, COUNT(*) as count,
                  SUM(CASE WHEN is_funny = 1 THEN 1 ELSE 0 END) as yes_count,
                  SUM(CASE WHEN is_funny = -1 THEN 1 ELSE 0 END) as skip_count
           FROM annotations
           GROUP BY annotator
           ORDER BY count DESC""",
    ).fetchall()

    # Annotation distribution
    dist = db.execute(
        """SELECT annotation_count, COUNT(*) as num_items
           FROM items GROUP BY annotation_count
           ORDER BY annotation_count""",
    ).fetchall()

    # Batch progress
    batch_nums = db.execute(
        "SELECT DISTINCT batch_num FROM items ORDER BY batch_num"
    ).fetchall()
    total_batches = len(batch_nums)
    batches = []
    for row in batch_nums:
        bnum = row["batch_num"]
        batch_total = db.execute(
            "SELECT COUNT(*) FROM items WHERE batch_num = ?", (bnum,),
        ).fetchone()[0]
        batch_done = db.execute(
            "SELECT COUNT(*) FROM items WHERE batch_num = ? AND annotation_count >= ?",
            (bnum, ANNOTATIONS_REQUIRED),
        ).fetchone()[0]
        batch_broken = db.execute(
            "SELECT COUNT(*) FROM items WHERE batch_num = ? AND annotation_count < 3 AND skip_count >= ?",
            (bnum, ANNOTATIONS_REQUIRED),
        ).fetchone()[0]
        batch_in_progress = db.execute(
            "SELECT COUNT(*) FROM items WHERE batch_num = ? AND annotation_count > 0 AND annotation_count < ?",
            (bnum, ANNOTATIONS_REQUIRED),
        ).fetchone()[0]
        if batch_done + batch_broken >= batch_total:
            status = "complete"
        elif batch_done > 0 or batch_in_progress > 0:
            status = "in_progress"
        else:
            status = "pending"
        batches.append({
            "num": bnum,
            "total": batch_total,
            "done": batch_done,
            "broken": batch_broken,
            "status": status,
            "pct": round(batch_done / batch_total * 100, 1) if batch_total > 0 else 0,
        })

    # Current batch number
    current_batch = next((b["num"] for b in batches if b["status"] != "complete"), total_batches)

    return render_template(
        "dashboard.html",
        total_items=total_items,
        total_annotations=total_annotations,
        total_skips=total_skips,
        completed_items=completed_items,
        broken_items=broken_items,
        required=ANNOTATIONS_REQUIRED,
        max_per_annotator=MAX_PER_ANNOTATOR,
        batch_size=BATCH_SIZE,
        annotators=annotators,
        distribution=dist,
        batches=batches,
        current_batch=current_batch,
        total_batches=total_batches,
    )


@app.route("/export")
def export():
    """Export all annotations as CSV."""
    if not session.pop("is_admin", None):
        return redirect(url_for("admin_login", next="export"))
    db = get_db()
    # Build annotator -> anonymous ID mapping
    annotator_rows = db.execute(
        "SELECT DISTINCT annotator FROM annotations ORDER BY annotator"
    ).fetchall()
    anon_map = {r["annotator"]: f"annotator_{i+1}" for i, r in enumerate(annotator_rows)}

    rows = db.execute(
        """SELECT
               a.id,
               i.task_id,
               i.candidate_index,
               i.tweet_text,
               i.meme_post_id,
               i.image_url,
               i.meme_title,
               i.selection_method,
               a.annotator,
               a.is_funny,
               a.flag_inappropriate,
               a.created_at
           FROM annotations a
           JOIN items i ON a.item_id = i.id
           ORDER BY i.task_id, i.candidate_index, a.annotator""",
    ).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "annotation_id", "task_id", "candidate_index", "tweet_text",
        "meme_post_id", "image_url", "meme_title", "selection_method",
        "annotator_id", "is_good_reply", "flag_inappropriate", "created_at",
    ])
    for row in rows:
        r = list(row)
        r[8] = anon_map[r[8]]  # replace email with anonymous ID
        writer.writerow(r)

    from flask import Response
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=annotations.csv"},
    )


# =========================
# STARTUP
# =========================

with app.app_context():
    init_db()

if __name__ == "__main__":
    app.run(debug=True, port=PORT)
