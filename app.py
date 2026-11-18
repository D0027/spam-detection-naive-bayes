"""SMS Spam Detector - Streamlit app for the Naive Bayes project.

The model is trained at startup from data/spam.csv with the exact pipeline and
hyper-parameters found by GridSearchCV in naive_bayes_spam.ipynb, so every number
shown here is computed live, not hard-coded.
"""
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "data" / "spam.csv"
SPAM_COLOR, HAM_COLOR, ACCENT = "#ef4444", "#10b981", "#6366f1"

EXAMPLES = {   # written fresh, none of these are in the training data
    "🎁 Prize scam": "URGENT! Your mobile number has won a £2000 bonus caller prize. Call 09050000327 now to claim. T&Cs apply",
    "📦 Delivery fee": "Your package could not be delivered. Pay the £1.99 fee at http://royalmail-redelivery.co to reschedule",
    "🏦 Bank KYC": "Dear customer your bank account is suspended. Verify your KYC immediately at http://sbi-verify.in",
    "💬 Running late": "Hey, running late, start without me. Save me a seat!",
    "📚 Class notes": "Can you send me the notes from yesterday's class? Thanks",
}

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="wide")

st.markdown(
    f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
.hero {{
    padding: 2rem 2.2rem; border-radius: 20px; margin-bottom: 1.2rem; color: #fff;
    background: radial-gradient(circle at 15% 20%, #6366f1 0%, transparent 45%),
                radial-gradient(circle at 85% 80%, #ec4899 0%, transparent 45%), #0f172a;
}}
.hero h1 {{ margin: 0; font-size: 2.3rem; color: #fff; }}
.hero p {{ margin: .4rem 0 0; opacity: .85; font-size: 1.05rem; }}
.pill {{ display:inline-block; padding:.2rem .7rem; margin:.7rem .4rem 0 0; border-radius:999px;
         background: rgba(255,255,255,.14); font-size:.82rem; }}
.verdict {{ padding: 1.2rem 1.4rem; border-radius: 16px; border: 2px solid; margin-bottom: .6rem; }}
.verdict h2 {{ margin: 0; font-size: 1.7rem; }}
.verdict p {{ margin: .2rem 0 0; opacity: .85; }}
.spam {{ border-color: {SPAM_COLOR}; background: rgba(239,68,68,.12); }}
.ham {{ border-color: {HAM_COLOR}; background: rgba(16,185,129,.12); }}
.msgbox {{ padding: 1rem 1.2rem; border-radius: 12px; line-height: 2; font-size: 1.02rem;
           background: rgba(128,128,128,.10); border: 1px solid rgba(128,128,128,.25); }}
.w-spam {{ background: rgba(239,68,68,.35); padding: 2px 5px; border-radius: 6px; font-weight: 600; }}
.w-ham {{ background: rgba(16,185,129,.35); padding: 2px 5px; border-radius: 6px; font-weight: 600; }}
div[data-testid="stMetric"] {{ background: rgba(128,128,128,.10); border: 1px solid rgba(128,128,128,.22);
                               padding: .8rem 1rem; border-radius: 14px; }}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- model
@st.cache_resource(show_spinner="Training Naive Bayes model on 4,135 messages...")
def load_model():
    raw = pd.read_csv(DATA_PATH, encoding="latin-1")[["v1", "v2"]]
    raw.columns = ["label", "message"]
    n_raw = len(raw)
    df = raw.dropna().assign(label=lambda d: d["label"].str.lower().str.strip())
    df = df.drop_duplicates(subset="message").reset_index(drop=True)  # de-duplicate BEFORE splitting
    y = (df["label"] == "spam").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["message"], y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Best parameters from GridSearchCV in the notebook (CV F1 = 0.942)
    pipe = Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english", sublinear_tf=False)),
        ("nb", MultinomialNB(alpha=0.01)),
    ]).fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    vec, nb = pipe.named_steps["vec"], pipe.named_steps["nb"]
    return {
        "pipe": pipe,
        "terms": vec.get_feature_names_out(),
        "log_ratio": nb.feature_log_prob_[1] - nb.feature_log_prob_[0],   # spam vs ham, per term
        "prior": nb.class_log_prior_[1] - nb.class_log_prior_[0],
        "y_te": y_te.to_numpy(),
        "proba_te": proba,
        "n_raw": n_raw, "n_unique": len(df), "n_spam": int(y.sum()),
        "n_train": len(X_tr), "n_test": len(X_te),
        "vocab": len(vec.vocabulary_),
    }


def explain(model, text):
    """Per-term contribution to the spam log-odds. Sum + prior == model log-odds."""
    vec = model["pipe"].named_steps["vec"]
    x = vec.transform([text])
    idx, vals = x.indices, x.data
    contrib = vals * model["log_ratio"][idx]
    out = pd.DataFrame({"term": model["terms"][idx], "contribution": contrib})
    return out.sort_values("contribution", ascending=False).reset_index(drop=True)


def highlight(text, contrib_df):
    """Colour single words of the message by their pull toward spam (red) or ham (green)."""
    pull = dict(zip(contrib_df["term"], contrib_df["contribution"]))
    scale = max(1e-9, float(np.abs(contrib_df["contribution"]).max())) if len(contrib_df) else 1.0

    def paint(word):
        c = pull.get(word.lower())
        if c is None or abs(c) < 0.12 * scale:
            return html.escape(word)
        return f'<span class="{"w-spam" if c > 0 else "w-ham"}">{html.escape(word)}</span>'

    parts = re.split(r"([\w']{2,})", text)   # odd indices are words
    out = [paint(p) if i % 2 else html.escape(p) for i, p in enumerate(parts)]
    return "".join(out).replace("\n", "<br>")


def signals(text):
    letters = [c for c in text if c.isalpha()]
    return {
        "Characters": len(text),
        "Digits": sum(c.isdigit() for c in text),
        "Capitals": f"{(sum(c.isupper() for c in letters) / len(letters) if letters else 0):.0%}",
        "Has link": bool(re.search(r"(https?://|www\.|\.com|\.co\.uk)", text, re.I)),
        "Phone-like number": bool(re.search(r"\d{5,}", text)),
    }


def gauge(p, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Spam probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": SPAM_COLOR if p >= threshold else HAM_COLOR},
               "steps": [{"range": [0, threshold * 100], "color": "rgba(16,185,129,.18)"},
                         {"range": [threshold * 100, 100], "color": "rgba(239,68,68,.18)"}],
               "threshold": {"line": {"color": ACCENT, "width": 4}, "value": threshold * 100}}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def predict(model, texts, threshold):
    p = model["pipe"].predict_proba(list(texts))[:, 1]
    return p, p >= threshold


# --------------------------------------------------------------------------- UI
model = load_model()

with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Spam threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Messages with spam probability at or above this are flagged. "
                               "Higher = fewer false alarms but more missed spam.")
    st.caption("Default 0.50 is what the notebook evaluates. Raise it if blocking a real message is costly.")
    st.divider()
    st.markdown("**Model**  \nTF-IDF (1-2 grams, stop words removed) + Multinomial Naive Bayes, `alpha = 0.01`")
    st.markdown(f"**Data**  \n{model['n_raw']:,} SMS → {model['n_unique']:,} unique "
                f"({model['n_spam']} spam)  \nTrain {model['n_train']:,} · Test {model['n_test']:,}")
    st.divider()
    st.markdown("[GitHub](https://github.com/D0027/spam-detection-naive-bayes)")

st.markdown(
    """
<div class="hero">
  <h1>📩 SMS Spam Detector</h1>
  <p>Paste any text message and see whether it is spam, and exactly <b>which words</b> made the model decide.</p>
  <span class="pill">Naive Bayes</span><span class="pill">TF-IDF 1-2 grams</span>
  <span class="pill">Explainable</span><span class="pill">Live metrics</span>
</div>
""",
    unsafe_allow_html=True,
)

tab_check, tab_batch, tab_perf, tab_how = st.tabs(
    ["🔍 Check a message", "📦 Batch check", "📊 Model performance", "🧠 How it works"])

# ------------------------------------------------------------------ single message
with tab_check:
    st.session_state.setdefault("msg", "")
    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        col.button(label, width="stretch", key=f"ex_{label}",
                   on_click=lambda t=text: st.session_state.update(msg=t))

    message = st.text_area("Message", key="msg", height=130, placeholder="Type or paste an SMS here...")

    if message.strip():
        p, is_spam = (v[0] for v in predict(model, [message], threshold))
        contrib = explain(model, message)
        left, right = st.columns([1.15, 1])

        with left:
            cls, icon, title = ("spam", "🚨", "SPAM") if is_spam else ("ham", "✅", "NOT SPAM (ham)")
            st.markdown(
                f'<div class="verdict {cls}"><h2>{icon} {title}</h2>'
                f"<p>Spam probability <b>{p:.1%}</b> · threshold {threshold:.0%}</p></div>",
                unsafe_allow_html=True)
            st.markdown("**Words that drove the decision** "
                        '<span class="w-spam">toward spam</span> <span class="w-ham">toward ham</span>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="msgbox">{highlight(message, contrib)}</div>', unsafe_allow_html=True)
            if p > 0.97 or p < 0.03:
                st.caption("Naive Bayes is known to be over-confident, so treat extreme percentages as "
                           "'very likely', not as certainty.")
        with right:
            st.plotly_chart(gauge(p, threshold), width="stretch")

        sig = signals(message)
        m = st.columns(5)
        m[0].metric("Characters", sig["Characters"], help="Spam averages ~138 chars vs ~70 for ham")
        m[1].metric("Digits", sig["Digits"], help="Spam averages ~15 digits vs ~0.3 for ham")
        m[2].metric("Capitals", sig["Capitals"], help="Spam averages ~11% vs ~6% for ham")
        m[3].metric("Link", "Yes" if sig["Has link"] else "No")
        m[4].metric("Phone-like number", "Yes" if sig["Phone-like number"] else "No")

        st.subheader("Why this verdict?")
        if contrib.empty:
            st.info("None of the words in this message are in the model's vocabulary, so the decision "
                    "falls back to the prior (most SMS are ham).")
        else:
            top = pd.concat([contrib.head(6), contrib.tail(6)]).drop_duplicates("term") \
                    .sort_values("contribution")
            fig = go.Figure(go.Bar(
                x=top["contribution"], y=top["term"], orientation="h",
                marker_color=[SPAM_COLOR if v > 0 else HAM_COLOR for v in top["contribution"]]))
            fig.update_layout(height=max(260, 34 * len(top)), margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="Push toward spam →   (log-odds contribution)  ← push toward ham",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
            total = model["prior"] + contrib["contribution"].sum()
            st.caption(f"Log-odds of spam = prior ({model['prior']:+.2f}) + sum of word contributions "
                       f"({contrib['contribution'].sum():+.2f}) = **{total:+.2f}** → probability {p:.1%}. "
                       "This is the same Bayes calculation scikit-learn performs.")
    else:
        st.info("👆 Pick an example or paste a message to get started.")

# ------------------------------------------------------------------ batch
with tab_batch:
    st.write("Check many messages at once: paste one per line, or upload a CSV.")
    mode = st.radio("Input", ["Paste lines", "Upload CSV"], horizontal=True, label_visibility="collapsed")
    texts = []
    if mode == "Paste lines":
        raw_text = st.text_area("One message per line", height=180, key="batch_text")
        texts = [t.strip() for t in raw_text.splitlines() if t.strip()]
    else:
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            try:
                up_df = pd.read_csv(up, encoding="latin-1")
                guess = next((c for c in up_df.columns if c.lower() in {"message", "text", "v2", "sms"}),
                             up_df.columns[0])
                col = st.selectbox("Column containing the message", up_df.columns,
                                   index=list(up_df.columns).index(guess))
                texts = up_df[col].dropna().astype(str).tolist()
            except Exception as exc:
                st.error(f"Could not read that file: {exc}")
    if texts:
        probs, flags = predict(model, texts, threshold)
        res = pd.DataFrame({"message": texts, "spam_probability": probs.round(4),
                            "verdict": np.where(flags, "SPAM", "ham")})
        a, b, c = st.columns(3)
        a.metric("Messages", f"{len(res):,}")
        b.metric("Flagged as spam", f"{int(flags.sum()):,}")
        c.metric("Spam share", f"{flags.mean():.1%}")
        st.dataframe(res.sort_values("spam_probability", ascending=False), width="stretch",
                     hide_index=True,
                     column_config={"spam_probability": st.column_config.ProgressColumn(
                         "Spam probability", min_value=0, max_value=1, format="%.2f")})
        st.download_button("⬇️ Download results (CSV)", res.to_csv(index=False).encode(),
                           "spam_results.csv", "text/csv")

# ------------------------------------------------------------------ performance
with tab_perf:
    y_te, p_te = model["y_te"], model["proba_te"]
    pred = (p_te >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    st.markdown(f"Measured live on **{len(y_te):,} held-out messages** the model never saw during "
                f"training (threshold = {threshold:.2f}).")
    k = st.columns(5)
    k[0].metric("Accuracy", f"{accuracy_score(y_te, pred):.3f}")
    k[1].metric("Precision", f"{precision_score(y_te, pred):.3f}", help="Of messages flagged spam, how many were spam")
    k[2].metric("Recall", f"{recall_score(y_te, pred):.3f}", help="Of all real spam, how much was caught")
    k[3].metric("F1", f"{f1_score(y_te, pred):.3f}")
    k[4].metric("ROC-AUC", f"{roc_auc_score(y_te, p_te):.3f}")

    c1, c2 = st.columns(2)
    with c1:
        cm = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]], x=["Predicted ham", "Predicted spam"], y=["Actual ham", "Actual spam"],
            text=[[tn, fp], [fn, tp]], texttemplate="%{text}", textfont={"size": 22},
            colorscale=[[0, "rgba(99,102,241,.08)"], [1, ACCENT]], showscale=False))
        cm.update_layout(title="Confusion matrix", height=340, yaxis_autorange="reversed",
                         margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(cm, width="stretch")
        st.caption(f"**{fp}** real messages wrongly flagged · **{fn}** spam messages missed.")
    with c2:
        fpr, tpr, _ = roc_curve(y_te, p_te)
        prec, rec, _ = precision_recall_curve(y_te, p_te)
        curves = go.Figure()
        curves.add_scatter(x=fpr, y=tpr, name=f"ROC (AUC {roc_auc_score(y_te, p_te):.3f})",
                           line=dict(color=ACCENT, width=3))
        curves.add_scatter(x=rec, y=prec, name=f"Precision-Recall (AP {average_precision_score(y_te, p_te):.3f})",
                           line=dict(color=SPAM_COLOR, width=3))
        curves.add_scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color="gray", dash="dot"))
        curves.update_layout(title="Evaluation curves", height=340, xaxis_title="FPR / Recall",
                             yaxis_title="TPR / Precision", margin=dict(l=10, r=10, t=50, b=10),
                             legend=dict(orientation="h", y=-0.25), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(curves, width="stretch")

    st.subheader("Threshold trade-off")
    rows = []
    for t in [0.3, 0.5, 0.7, 0.9, 0.95]:
        pt = (p_te >= t).astype(int)
        _tn, _fp, _fn, _tp = confusion_matrix(y_te, pt).ravel()
        rows.append({"Threshold": t, "Precision": round(precision_score(y_te, pt), 3),
                     "Recall": round(recall_score(y_te, pt), 3),
                     "Ham flagged as spam": int(_fp), "Spam missed": int(_fn)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.warning(
        "**Read this before trusting the numbers.** They describe how the model does on messages from the "
        "same collection it was trained on (mostly UK/Singapore SMS, collected in the 2000s). Today's spam "
        "(delivery-fee and bank-phishing links, crypto, WhatsApp scams) uses different wording, so expect "
        "lower accuracy on modern messages. Retrain on fresh data before using this for anything important.")

# ------------------------------------------------------------------ how it works
with tab_how:
    st.markdown(
        f"""
### The pipeline
1. **Clean:** {model['n_raw']:,} messages → **{model['n_unique']:,} unique** (exact duplicates removed *before*
   splitting, otherwise the test set leaks into training).
2. **Vectorise:** TF-IDF over single words and word pairs (vocabulary of **{model['vocab']:,}** terms), English stop words removed.
3. **Classify:** Multinomial Naive Bayes with Laplace smoothing `alpha = 0.01`, tuned with 5-fold GridSearchCV on F1.
4. **Decide:** flag as spam when `P(spam | message) ≥ threshold`.

### The math in one line
`log-odds(spam) = log P(spam)/P(ham) + Σ  tfidf(word) × log P(word | spam)/P(word | ham)`

The bar chart in the first tab plots exactly those per-word terms, so the explanation is the real calculation,
not an approximation.

### Known limitations
- Word order and context are ignored (the "naive" independence assumption).
- Probabilities are over-confident.
- Short slang messages containing spam-flavoured words (e.g. *nokia*) can be false alarms.
- Vocabulary is frozen at the training data, so new spam tactics are missed until you retrain.
""")
