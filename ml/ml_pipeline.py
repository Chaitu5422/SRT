# app/ml_pipeline.py
import os
from datetime import datetime, timedelta
from pydoc_data import topics
from typing import List, Dict, Any
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    from bertopic import BERTopic
    _HAS_BERTOPIC = True
except Exception:
    _HAS_BERTOPIC = False

try:
    import yake
    _HAS_YAKE = True
except Exception:
    _HAS_YAKE = False

from app.db import get_reflection_collection, get_mongo_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _stringify_keys(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert non-string dict keys to strings (Mongo requires string keys)."""
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        new_k = str(k)
        if isinstance(v, dict):
            out[new_k] = _stringify_keys(v)
        else:
            out[new_k] = v
    return out


# ✅ FIXED — Uses CreatedAt + Email fields from your DB
def fetch_reflections(start_date: datetime, end_date: datetime, email: str = None) -> List[Dict[str, Any]]:
    col = get_reflection_collection()

    # Correct filter for both daily/weekly/monthly reflections
    q = {
        "$and": [
            {
                "$or": [
                    {"CreatedAt": {"$gte": start_date, "$lt": end_date}},
                    {"SelectedDate": {"$gte": start_date, "$lt": end_date}}
                ]
            }
        ]
    }

    # Filter by email if provided
    if email:
        q["$and"].append({"Email": email})

    docs = list(col.find(q))
    out = []

    for d in docs:
        answers = []
        for k in ("Question1", "Question2", "Question3"):
            val = d.get("Answers", {}).get(k, {})
            if isinstance(val, dict):
                answers.extend(val.get("answers", []) or [])
            elif isinstance(val, str):
                answers.append(val)

        text = " ".join(answers).strip()

        out.append({
            "reflection_id": d.get("ReflectionID"),
            "email": d.get("Email"),
            "empID": d.get("empID"),
            "Name": d.get("Name"),
            "date": d.get("SelectedDate") or d.get("CreatedAt"),
            "text": text
        })

    return out


class MLPipeline:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        try:
            self.embed_model = SentenceTransformer(embedding_model_name)
        except Exception as e:
            logger.warning("SentenceTransformer load failed: %s", e)
            self.embed_model = None

        self.sentiment = SentimentIntensityAnalyzer()

    def _sentiment_scores(self, texts: List[str]) -> List[float]:
        scores = []
        for t in texts:
            v = self.sentiment.polarity_scores(t)
            scores.append(v["compound"])
        return scores

    def _extract_keywords_yake(self, texts: List[str], top_k=5):
        if not _HAS_YAKE:
            return None
        extractor = yake.KeywordExtractor(lan="en", n=1, top=top_k)
        combined = " ".join(texts)
        kws = extractor.extract_keywords(combined)
        return [k for k, _ in kws][:top_k]

    def _tfidf_nmf_topics(self, texts: List[str], n_topics=5):
        # Handle small dataset problems safely
        min_df_value = 1 if len(texts) < 3 else 3

        vectorizer = TfidfVectorizer(
            max_df=0.85,
            min_df=min_df_value,
            stop_words="english",
            ngram_range=(1, 2)
        )

        X = vectorizer.fit_transform(texts)

        n_samples, n_features = X.shape

        # If no features → return empty
        if n_samples == 0 or n_features == 0:
            return {}

        # SAFE n_components
        max_allowed = min(n_samples, n_features)

        if max_allowed <= 1:
            # Only 1 reflection or 1 feature → no topic modeling possible
            # return top terms instead
            terms = vectorizer.get_feature_names_out()
            return {0: terms[:5]}

        # Reduce topics to safe level
        n_topics = min(n_topics, max_allowed)

        # Build NMF safely
        nmf = NMF(
            n_components=n_topics,
            random_state=42,
            init="nndsvda",
            max_iter=400
        )

        W = nmf.fit_transform(X)
        H = nmf.components_

        terms = vectorizer.get_feature_names_out()
        topics = {}

        for idx, comp in enumerate(H):
            term_idx = comp.argsort()[::-1][:8]
            topics[int(idx)] = [terms[i] for i in term_idx]

        return topics

    def _bertopic_topics(self, texts: List[str], embeddings: np.ndarray = None):
        topic_model = BERTopic(verbose=False, calculate_probabilities=False)
        topics, probs = topic_model.fit_transform(texts, embeddings)
        info = topic_model.get_topic_info()
        topics_map = {}
        for tid in info.Topic.unique():
            if tid == -1:
                continue
            terms = [t for t, _ in topic_model.get_topic(tid)]
            topics_map[int(tid)] = terms
        return topics_map, topic_model, topics, probs

    def _cluster_texts(self, texts: List[str], embeddings: np.ndarray = None, n_clusters=5):
        if embeddings is None and self.embed_model:
            embeddings = self.embed_model.encode(texts, show_progress_bar=False)
        if embeddings is None:
            vec = TfidfVectorizer(max_df=0.85, stop_words="english")
            X = vec.fit_transform(texts)
            km = KMeans(n_clusters=min(n_clusters, X.shape[0]), random_state=42)
            labels = km.fit_predict(X)
            return labels, None
        km = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42)
        labels = km.fit_predict(embeddings)
        return labels, km

    def generate_summary(self, reflections: List[Dict[str, Any]], period_label: str = "weekly") -> Dict[str, Any]:

        if not reflections:
            return {
                "email": None,
                "empID": None,
                "Name": None,
                "period": period_label,
                "summary_text": "No reflections submitted during this period."
            }

        # Extract fields correctly from first reflection
        first = reflections[0]
        email = first.get("email")
        empID = first.get("empID") or first.get("empId") or first.get("EmpID")
        name = first.get("Name") or first.get("name")

        # Combine all reflection texts
        combined_text = " ".join([r["text"] for r in reflections])

        #summary_text = f"Summary for {period_label}: {combined_text}"
        summary_text = combined_text or "No reflections submitted during this period."

        # Final minimal summary
        return {
            "email": email,
            "empID": empID,
            "Name": name,
            "period": period_label,
            "summary_text": summary_text
        }



    def _top_tfidf_terms(self, texts: List[str], top_n=10):
        vec = TfidfVectorizer(max_df=0.85, stop_words="english", ngram_range=(1,2))
        X = vec.fit_transform(texts)
        if X.shape[1] == 0:
            return []
        sums = np.asarray(X.sum(axis=0)).ravel()
        terms = vec.get_feature_names_out()
        return [terms[i] for i in sums.argsort()[::-1][:top_n]]

    def _cluster_representatives(self, texts, labels, embeddings=None, cluster_model=None):
        reps = {}
        df = pd.DataFrame({"text": texts, "label": labels})

        # If no embeddings and no cluster model -> fallback: first per label
        if embeddings is None and cluster_model is None:
            for lbl in sorted(df.label.unique()):
                row = df[df.label == lbl].iloc[0]
                reps[int(lbl)] = row.text
            return reps

        # If embeddings missing but embed_model available -> compute embeddings
        if embeddings is None:
            if self.embed_model:
                embeddings = self.embed_model.encode(texts, show_progress_bar=False)
            else:
                # fallback to first text per label
                for lbl in sorted(df.label.unique()):
                    reps[int(lbl)] = df[df.label == lbl].iloc[0].text
                return reps

        # Ensure cluster_model has cluster centers (guard)
        centers = None
        if cluster_model is not None and hasattr(cluster_model, "cluster_centers_"):
            centers = cluster_model.cluster_centers_

        for lbl in sorted(df.label.unique()):
            # find indices (as numpy array) belonging to this cluster
            idxs = np.where(np.array(labels) == lbl)[0]
            if len(idxs) == 0:
                continue

            # If we have cluster centers and a valid center for this label, use it
            if centers is not None and lbl < len(centers):
                centroid = centers[lbl]
                subset = embeddings[idxs]

                # compute nearest index in subset
                try:
                    nearest_idxs, _ = pairwise_distances_argmin_min([centroid], subset)
                except Exception:
                    nearest_idxs = []

                if len(nearest_idxs) == 0:
                    # fallback — pick first in cluster
                    chosen_global_idx = int(idxs[0])
                else:
                    # nearest_idxs[0] is index inside subset -> map to global index
                    chosen_global_idx = int(idxs[int(nearest_idxs[0])])

                reps[int(lbl)] = texts[chosen_global_idx]

            else:
                # No centroids available — fallback to first text in cluster
                reps[int(lbl)] = texts[int(idxs[0])]

        return reps


    def _extract_action_items(self, texts: List[str]) -> List[str]:
        triggers = ["will ", "plan to ", "need to ", "to do ", "i'll ", "i will ", "we will ", "we should ", "follow up"]
        actions, seen = [], set()
        for doc in texts:
            for s in [s.strip() for s in doc.split('.') if s.strip()]:
                if any(t in s.lower() for t in triggers):
                    if s.lower() not in seen:
                        seen.add(s.lower())
                        actions.append(s)
        return actions[:50]

    def _sentiment_distribution(self, scores: List[float]):
        pos = sum(1 for s in scores if s > 0.05)
        neg = sum(1 for s in scores if s < -0.05)
        neu = len(scores) - pos - neg
        return {"positive": pos, "negative": neg, "neutral": neu}


def store_summary(summary: Dict[str, Any], period_start: datetime, period_end: datetime, period_label: str):
    client = get_mongo_client()
    db = client["myreflection"]
    coll = db["ml_reports"]

    # Make keys safe for Mongo (integers -> strings)
    safe_summary = _stringify_keys(summary)

    # Pull only the minimal summary content you want stored
    # (you asked to remove sentiment_distribution, top_keywords, topics, cluster_representatives, action_items,
    # and to also remove generated_at, n_reflections, avg_sentiment from stored summary)
    minimal_summary = {
        "summary_text": safe_summary.get("summary_text")
    }

    # If empID/Name are in summary use them; else try to fetch using email (best-effort)
    emp_id = safe_summary.get("empID")
    name = safe_summary.get("Name")

    # Best-effort MySQL lookup if empID/Name missing but email is present
    if (not emp_id or not name) and safe_summary.get("email"):
        try:
            from app.db import get_conn
            conn = get_conn()
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT empID, Name FROM employees WHERE EmailID=%s", (safe_summary.get("email"),))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                emp_id = emp_id or row.get("empID")
                name = name or row.get("Name")
        except Exception:
            pass


    payload = {
        "email": safe_summary.get("email"),
        "empID": emp_id,
        "Name": name,
        "period_label": period_label,
        "period_start": period_start,
        "period_end": period_end,
        "summary": minimal_summary,
        "generated_at": datetime.utcnow()
    }

    result = coll.insert_one(payload)
    return str(result.inserted_id)


def run_pipeline_for_period(start_date: datetime, end_date: datetime, period_label: str = "weekly", email: str = None):
    reflections = fetch_reflections(start_date, end_date, email=email)
    pipe = MLPipeline()
    summary = pipe.generate_summary(reflections, period_label=period_label)
    report_id = store_summary(summary, start_date, end_date, period_label)
    return {"report_id": report_id, "summary": summary}