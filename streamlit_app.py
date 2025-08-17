# inbox_triage_embeddings.py
import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
import re, ssl
from dateutil import tz
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

IMAP_HOST = "imap.gmail.com"

# -------------------- Utils --------------------
def _dh(s):
    if not s:
        return ""
    parts = decode_header(s)
    out = []
    for text, enc in parts:
        out.append(text.decode(enc or "utf-8", errors="replace") if isinstance(text, bytes) else text)
    return "".join(out)

def _clean_text(t: str) -> str:
    if not t:
        return ""
    return re.sub(r"\s+", " ", t).strip()

def _first_text_part(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and part.get_content_disposition() != "attachment":
                try:
                    return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                except Exception:
                    pass
        for part in msg.walk():
            if part.get_content_type() == "text/html" and part.get_content_disposition() != "attachment":
                try:
                    raw = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                    return re.sub(r"<[^>]+>", " ", raw)
                except Exception:
                    pass
    else:
        if msg.get_content_type() in ("text/plain", "text/html"):
            raw = msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="replace")
            return re.sub(r"<[^>]+>", " ", raw) if msg.get_content_type() == "text/html" else raw
    return ""

def _localize_date(dstr: str) -> str:
    try:
        dt = parsedate_to_datetime(dstr)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=tz.tzutc())
        return dt.astimezone(tz.tzlocal()).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return dstr or ""

# -------------------- IMAP --------------------
def imap_login(email_addr: str, app_password: str) -> imaplib.IMAP4_SSL:
    ctx = ssl.create_default_context()
    imap = imaplib.IMAP4_SSL(IMAP_HOST, 993, ssl_context=ctx)
    imap.login(email_addr, app_password)
    return imap

def fetch_latest_200(imap: imaplib.IMAP4_SSL, mailbox="INBOX") -> List[Dict]:
    typ, _ = imap.select(mailbox)
    if typ != "OK":
        raise RuntimeError("Unable to select INBOX")
    typ, data = imap.uid("search", None, "ALL")
    if typ != "OK":
        raise RuntimeError("IMAP search failed")
    uids = data[0].split()
    if not uids:
        return []
    latest_200 = uids[-200:]
    rows = []
    for uid in reversed(latest_200):  # newest first
        typ, msg_data = imap.uid("fetch", uid, "(RFC822)")
        if typ != "OK" or not msg_data or not msg_data[0]:
            continue
        msg = email.message_from_bytes(msg_data[0][1])
        subject = _dh(msg.get("Subject"))
        from_ = _dh(msg.get("From"))
        date = _localize_date(msg.get("Date", ""))
        body = _clean_text(_first_text_part(msg))
        snippet = (body[:300] + "…") if len(body) > 300 else body
        rows.append({"uid": uid.decode(), "subject": subject or "(no subject)", "from": from_, "date": date, "snippet": snippet})
    return rows

def archive_uids(imap: imaplib.IMAP4_SSL, uids: List[str]) -> Tuple[int, List[str]]:
    archived, errors = 0, []
    for uid in uids:
        typ, _ = imap.uid("store", uid, "-X-GM-LABELS", "(\\Inbox)")
        if typ == "OK":
            archived += 1
            continue
        typ1, _ = imap.uid("copy", uid, "[Gmail]/All Mail")
        typ2, _ = imap.uid("store", uid, "+FLAGS", "(\\Deleted)")
        if typ1 == "OK" and typ2 == "OK":
            archived += 1
        else:
            errors.append(uid)
    imap.expunge()
    return archived, errors

# -------------------- Embeddings + Clustering --------------------
@st.cache_resource(show_spinner=False)
def load_model(name="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_data(show_spinner=False)
def embed_texts(model_name: str, texts: List[str]) -> np.ndarray:
    model = load_model(model_name)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)

def choose_k_and_cluster(embeddings: np.ndarray, k_min=3, k_max=8, sample_for_sil=200):
    best = None
    for k in range(k_min, min(k_max, embeddings.shape[0]) + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) <= 1:
            continue
        score = silhouette_score(embeddings, labels, metric="cosine", sample_size=min(sample_for_sil, embeddings.shape[0]))
        if best is None or score > best[0]:
            best = (score, km, labels)
    if best is None:
        km = KMeans(n_clusters=k_min, n_init="auto", random_state=42).fit(embeddings)
        return km.labels_, km
    return best[2], best[1]

def name_clusters(texts: List[str], labels: np.ndarray, topn=4) -> List[str]:
    names = []
    df = pd.DataFrame({"text": texts, "label": labels})
    for cid in range(labels.max() + 1):
        ctexts = df[df.label == cid]["text"].tolist()
        if not ctexts:
            names.append(f"Cluster {cid+1}")
            continue
        vec = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1,2))
        X = vec.fit_transform(ctexts)
        terms = vec.get_feature_names_out()
        mean_scores = np.asarray(X.mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[-topn:][::-1]
        names.append(", ".join(terms[i] for i in top_idx))
    return names

# -------------------- UI --------------------
st.set_page_config(page_title="Inbox Triage (Embeddings)", page_icon="📬", layout="wide")
st.title("📬 Inbox Triage Assistant — Sentence Transformers")

with st.sidebar:
    st.header("Gmail IMAP Login")
    default_email = st.secrets.get("EMAIL", "")
    default_pass = st.secrets.get("APP_PASSWORD", "")
    email_addr = st.text_input("Gmail address", value=default_email, placeholder="you@gmail.com")
    app_password = st.text_input("App password", value=default_pass, type="password", placeholder="xxxx xxxx xxxx xxxx")
    st.caption("Use a Gmail **App Password** (Google Account → Security → App passwords).")
    st.divider()
    st.header("Clustering")
    model_name = st.selectbox("Embedding model", ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2"], index=0)
    k_min = st.slider("Min clusters (K)", 3, 8, 3)
    k_max = st.slider("Max clusters (K)", 3, 12, 6)
    fetch_btn = st.button("Fetch 200 emails")

if "emails" not in st.session_state:
    st.session_state.emails = []
if "labels" not in st.session_state:
    st.session_state.labels = None
if "cluster_names" not in st.session_state:
    st.session_state.cluster_names = []

def do_fetch():
    if not email_addr or not app_password:
        st.error("Enter your Gmail address and App Password in the sidebar.")
        return
    try:
        with st.spinner("Connecting to Gmail…"):
            imap = imap_login(email_addr, app_password)
            with st.spinner("Fetching latest 200 messages from INBOX…"):
                rows = fetch_latest_200(imap)
            imap.logout()
        st.session_state.emails = rows
        st.success(f"Loaded {len(rows)} messages.")
    except imaplib.IMAP4.error as e:
        st.error(f"IMAP auth/connection failed: {e}")
    except Exception as e:
        st.error(f"Error fetching messages: {e}")

def do_cluster():
    df = pd.DataFrame(st.session_state.emails)
    docs = [(s or "") + " " + (sn or "") for s, sn in zip(df["subject"], df["snippet"])]
    if not docs:
        st.warning("No emails to cluster.")
        return
    with st.spinner("Computing embeddings…"):
        E = embed_texts(model_name, docs)
    with st.spinner("Clustering…"):
        labels, _ = choose_k_and_cluster(E, k_min=k_min, k_max=k_max)
    st.session_state.labels = labels
    st.session_state.cluster_names = name_clusters(docs, labels, topn=4)

def do_archive(cluster_id: int):
    uids = [row["uid"] for row, lab in zip(st.session_state.emails, st.session_state.labels) if lab == cluster_id]
    if not uids:
        st.info("Nothing to archive in this cluster.")
        return
    try:
        imap = imap_login(email_addr, app_password)
        imap.select("INBOX")
        archived, errors = archive_uids(imap, uids)
        imap.logout()
        if errors:
            st.warning(f"Archived {archived}, {len(errors)} failed (sample: {errors[:5]}).")
        else:
            st.success(f"Archived {archived} messages from this cluster.")
        st.session_state.emails = [r for r in st.session_state.emails if r["uid"] not in set(uids)]
        if st.session_state.emails:
            do_cluster()
        else:
            st.session_state.labels = None
            st.session_state.cluster_names = []
    except Exception as e:
        st.error(f"Archiving failed: {e}")

if fetch_btn:
    do_fetch()

if st.session_state.emails:
    if st.button("Cluster emails (embeddings)"):
        do_cluster()

    if st.session_state.labels is not None:
        df = pd.DataFrame(st.session_state.emails)
        df["cluster"] = st.session_state.labels
        names = st.session_state.cluster_names

        st.subheader("Clusters")
        for cid in sorted(df["cluster"].unique()):
            cdf = df[df["cluster"] == cid].copy()
            title = names[cid] if cid < len(names) else f"Cluster {cid+1}"
            with st.expander(f"Cluster {cid+1}: {title} — {len(cdf)} emails", expanded=False):
                st.dataframe(cdf[["date", "from", "subject"]], use_container_width=True, hide_index=True)
                c1, c2 = st.columns([1,1])
                with c1:
                    if st.button(f"Archive cluster {cid+1}", key=f"arch_{cid}"):
                        do_archive(cid)
                with c2:
                    st.download_button("Export CSV", data=cdf.to_csv(index=False), file_name=f"cluster_{cid+1}.csv", mime="text/csv")
else:
    st.info("Enter credentials and click **Fetch 200 emails**.")
