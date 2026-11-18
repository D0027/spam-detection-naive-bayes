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
INK, MUTED, LINE, NAVY = "#101828", "#667085", "#DDE2EA", "#0B1F3A"
SPAM_COLOR, HAM_COLOR = "#C4271B", "#067647"
SPAM_SOFT, HAM_SOFT = "#FDE8E6", "#DCF5E6"
FONT = "IBM Plex Sans, Segoe UI, sans-serif"

EXAMPLES = {   # written fresh, none of these are in the training data
    "Prize scam": "URGENT! Your mobile number has won a £2000 bonus caller prize. Call 09050000327 now to claim. T&Cs apply",
    "Delivery fee": "Your package could not be delivered. Pay the £1.99 fee at http://royalmail-redelivery.co to reschedule",
    "Bank KYC": "Dear customer your bank account is suspended. Verify your KYC immediately at http://sbi-verify.in",
    "Running late": "Hey, running late, start without me. Save me a seat!",
    "Class notes": "Can you send me the notes from yesterday's class? Thanks",
}

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="wide")

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{--ink:#101828;--muted:#667085;--line:#DDE2EA;--surface:#FFFFFF;--canvas:#F2F4F8;--navy:#0B1F3A;
--spam:#C4271B;--spam-soft:#FDE8E6;--ham:#067647;--ham-soft:#DCF5E6;
--display:'Bricolage Grotesque','Segoe UI',system-ui,sans-serif;--body:'IBM Plex Sans','Segoe UI',system-ui,sans-serif;}
.stApp{background:var(--canvas);}
.block-container{padding-top:2.2rem;max-width:1120px;}
footer,.stAppDeployButton,#MainMenu{display:none!important;}
.stApp,.stApp p,.stApp label,.stApp li,.stApp button,.stApp textarea,.stApp input,.stApp td,.stApp th{font-family:var(--body);}
.stApp h1,.stApp h2,.stApp h3{font-family:var(--display);letter-spacing:-.02em;color:var(--ink);}

.page-title{font-family:var(--display);font-weight:800;font-size:2.7rem;letter-spacing:-.035em;line-height:1.05;color:var(--ink);margin:0;}
.page-sub{font-size:1.12rem;color:var(--muted);margin:.55rem 0 1.5rem;max-width:36rem;line-height:1.5;}

button[data-baseweb="tab"] p{font-size:1rem;font-weight:600;}
.stButton>button{border-radius:999px;border:1px solid var(--line);background:var(--surface);color:var(--ink);font-weight:500;padding:.3rem .9rem;transition:border-color .15s,color .15s;}
.stButton>button:hover{border-color:var(--navy);color:var(--navy);background:var(--surface);}
.stTextArea textarea{border-radius:16px;font-size:1.05rem;line-height:1.5;background:var(--surface);border:1px solid var(--line);padding:1rem 1.1rem;}
.stTextArea textarea:focus{border-color:var(--navy);box-shadow:0 0 0 3px rgba(11,31,58,.12);}
div[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:.9rem 1rem;}
div[data-testid="stMetricValue"]{font-family:var(--display);font-weight:800;}
section[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--line);}

.result{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:1.1rem;margin:1rem 0 .8rem;}
@media(max-width:760px){.result{grid-template-columns:1fr;}}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:20px;padding:1.3rem 1.4rem;}
.panel-title{font-size:.88rem;color:var(--muted);margin:0 0 .8rem;font-weight:500;}
.bubble{background:#E9EBF0;color:var(--ink);border-radius:20px 20px 20px 6px;padding:.9rem 1.1rem;font-size:1.08rem;line-height:1.9;overflow-wrap:anywhere;}
.legend{margin-top:.9rem;font-size:.85rem;color:var(--muted);}
.w-spam,.w-ham{padding:1px 6px;border-radius:6px;font-weight:600;}
.w-spam{background:var(--spam-soft);color:var(--spam);box-shadow:inset 0 -2px 0 var(--spam);}
.w-ham{background:var(--ham-soft);color:var(--ham);box-shadow:inset 0 -2px 0 var(--ham);}

.stamp{display:inline-block;font-family:var(--display);font-weight:800;font-size:2rem;letter-spacing:.05em;padding:.15rem .9rem;border:3px solid;border-radius:10px;outline:2px solid;outline-offset:3px;transform:rotate(-4deg);margin:.3rem 0 1.3rem .3rem;animation:stamp .28s ease-out;}
.stamp.spam{color:var(--spam);}
.stamp.ham{color:var(--ham);}
@keyframes stamp{from{transform:rotate(-4deg) scale(1.35);opacity:0}to{transform:rotate(-4deg) scale(1);opacity:1}}
@media (prefers-reduced-motion:reduce){.stamp{animation:none;}}
.score{font-family:var(--display);font-weight:800;font-size:3rem;line-height:1;letter-spacing:-.03em;color:var(--ink);}
.score-l{color:var(--muted);font-size:.95rem;margin:.25rem 0 1.2rem;}
.meter{position:relative;width:100%;height:12px;border-radius:999px;background:linear-gradient(90deg,#7ED6A8 0%,#F4D675 50%,#F08A80 100%);margin:.4rem 0 .5rem;}
.dot{position:absolute;top:50%;width:22px;height:22px;border-radius:50%;background:var(--ink);border:3px solid #fff;transform:translate(-50%,-50%);box-shadow:0 1px 4px rgba(0,0,0,.3);}
.tick{position:absolute;top:-6px;bottom:-6px;width:2px;background:var(--ink);opacity:.5;transform:translateX(-50%);}
.scale{display:flex;justify-content:space-between;font-size:.8rem;color:var(--muted);}
.note{font-size:.85rem;color:var(--muted);margin:.9rem 0 0;line-height:1.5;}

.chips{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.7rem;margin:.4rem 0 1.2rem;}
@media(max-width:900px){.chips{grid-template-columns:repeat(2,minmax(0,1fr));}}
.chip{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:.75rem .9rem;}
.chip-v{font-family:var(--display);font-weight:800;font-size:1.5rem;color:var(--ink);line-height:1.1;}
.chip-l{font-size:.86rem;font-weight:600;color:var(--ink);margin-top:.2rem;}
.chip-h{font-size:.76rem;color:var(--muted);margin-top:.1rem;}

.callout{background:var(--surface);border:1px solid var(--line);border-left:5px solid var(--navy);border-radius:14px;padding:1rem 1.2rem;margin:.8rem 0 1rem;line-height:1.65;}
.steps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.8rem;margin:.6rem 0 1.2rem;}
@media(max-width:900px){.steps{grid-template-columns:repeat(2,minmax(0,1fr));}}
.step{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem;}
.step-n{font-family:var(--display);font-weight:800;font-size:1.6rem;color:var(--navy);line-height:1;}
.step-t{font-weight:600;margin:.4rem 0 .2rem;color:var(--ink);}
.step-d{font-size:.88rem;color:var(--muted);line-height:1.5;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


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


def predict(model, texts, threshold):
    p = model["pipe"].predict_proba(list(texts))[:, 1]
    return p, p >= threshold


# --------------------------------------------------------------------------- UI helpers
def style_fig(fig, height, **kw):
    margin = kw.pop("margin", dict(l=10, r=10, t=10, b=10))
    fig.update_layout(height=height, margin=margin, font=dict(family=FONT, color=INK, size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", **kw)
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=MUTED)
    fig.update_yaxes(gridcolor=LINE)
    return fig


def result_card(message_html, is_spam, p, threshold):
    cls, title = ("spam", "SPAM") if is_spam else ("ham", "NOT SPAM")
    why = (f"Flagged because the spam score is at or above your {threshold:.0%} threshold." if is_spam
           else f"Passed because the spam score is below your {threshold:.0%} threshold.")
    close = ('<p class="note">Close call: the score is near your threshold, so treat this verdict with care.</p>'
             if abs(p - threshold) <= 0.15 else "")
    dot, tick = min(98, max(2, p * 100)), threshold * 100
    return "".join([
        '<div class="result">',
        '<div class="panel"><p class="panel-title">Message as the model reads it</p>',
        f'<div class="bubble">{message_html}</div>',
        '<p class="legend"><span class="w-spam">red</span> words push toward spam, '
        '<span class="w-ham">green</span> words push toward normal. Unmarked words barely matter.</p></div>',
        '<div class="panel"><p class="panel-title">Verdict</p>',
        f'<div class="stamp {cls}">{title}</div>',
        f'<div class="score">{p:.1%}</div><div class="score-l">chance this message is spam</div>',
        f'<div class="meter"><div class="tick" style="left:{tick:.1f}%"></div>'
        f'<div class="dot" style="left:{dot:.1f}%"></div></div>',
        '<div class="scale"><span>Not spam</span><span>Spam</span></div>',
        f'<p class="note">{why} The dark line on the bar marks that threshold.</p>{close}</div>',
        '</div>'])


def chip(value, label, hint):
    return (f'<div class="chip"><div class="chip-v">{value}</div>'
            f'<div class="chip-l">{label}</div><div class="chip-h">{hint}</div></div>')


# --------------------------------------------------------------------------- UI
model = load_model()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Spam threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Messages with spam probability at or above this are flagged. "
                               "Higher = fewer false alarms but more missed spam.")
    st.caption("0.50 is what the notebook evaluates. Raise it if wrongly blocking a real message is costly.")
    st.divider()
    st.markdown("**Model**  \nTF-IDF (1-2 grams, stop words removed) + Multinomial Naive Bayes, `alpha = 0.01`")
    st.markdown(f"**Data**  \n{model['n_raw']:,} SMS, {model['n_unique']:,} unique "
                f"({model['n_spam']} spam)  \nTrain {model['n_train']:,}, test {model['n_test']:,}")
    st.divider()
    st.markdown("[View the code on GitHub](https://github.com/D0027/spam-detection-naive-bayes)")

st.markdown(
    '<h1 class="page-title">SMS Spam Detector</h1>'
    '<p class="page-sub">Paste a text message to see if it is spam, and which words gave it away.</p>',
    unsafe_allow_html=True)

tab_check, tab_batch, tab_perf, tab_how = st.tabs(
    ["Check a message", "Check many", "How well it works", "How it works"])

# ------------------------------------------------------------------ single message
with tab_check:
    st.session_state.setdefault("msg", "")
    st.caption("Start with a sample, or paste your own below.")
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        col.button(label, width="stretch", key=f"ex_{label}",
                   on_click=lambda t=text: st.session_state.update(msg=t))

    message = st.text_area("Message", key="msg", height=120, label_visibility="collapsed",
                           placeholder="Type or paste a text message here...")

    if message.strip():
        p, is_spam = (v[0] for v in predict(model, [message], threshold))
        contrib = explain(model, message)

        st.markdown(result_card(highlight(message, contrib), is_spam, p, threshold), unsafe_allow_html=True)
        if p > 0.97 or p < 0.03:
            st.caption("Naive Bayes is known to be over-confident, so read extreme percentages as "
                       "'very likely', not as certainty.")

        sig = signals(message)
        st.markdown('<div class="chips">' + "".join([
            chip(sig["Characters"], "Characters", "spam averages 138, normal 70"),
            chip(sig["Digits"], "Digits", "spam averages 15, normal 0.3"),
            chip(sig["Capitals"], "Capital letters", "spam 11%, normal 6%"),
            chip("Yes" if sig["Has link"] else "No", "Link", "web address in the text"),
            chip("Yes" if sig["Phone-like number"] else "No", "Phone-like number", "5 or more digits in a row"),
        ]) + "</div>", unsafe_allow_html=True)

        st.subheader("Why this verdict?")
        if contrib.empty:
            st.info("None of the words in this message are in the model's vocabulary, so the decision "
                    "falls back to the prior (most SMS are ham).")
        else:
            st.write("Each bar is one word or word pair. Red bars push the score toward spam, "
                     "green bars push it toward normal.")
            top = pd.concat([contrib.head(6), contrib.tail(6)]).drop_duplicates("term") \
                    .sort_values("contribution")
            fig = go.Figure(go.Bar(
                x=top["contribution"], y=top["term"], orientation="h",
                text=[f"{v:+.1f}" for v in top["contribution"]], textposition="outside", cliponaxis=False,
                marker_color=[SPAM_COLOR if v > 0 else HAM_COLOR for v in top["contribution"]]))
            style_fig(fig, max(260, 34 * len(top)),
                      xaxis_title="Pushes toward normal   |   pushes toward spam", yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
            with st.expander("Show the math"):
                total = model["prior"] + contrib["contribution"].sum()
                st.write(f"Log-odds of spam = prior ({model['prior']:+.2f}) + sum of word contributions "
                         f"({contrib['contribution'].sum():+.2f}) = **{total:+.2f}**, which is a probability of "
                         f"{p:.1%}. This is the same Bayes calculation scikit-learn performs.")
    else:
        st.info("Pick a sample above or paste a message to get a verdict.")

# ------------------------------------------------------------------ batch
with tab_batch:
    st.write("Check many messages at once: paste one per line, or upload a CSV.")
    mode = st.radio("Input", ["Paste lines", "Upload CSV"], horizontal=True, label_visibility="collapsed")
    texts = []
    if mode == "Paste lines":
        st.button("Fill with sample messages", key="batch_sample",
                  on_click=lambda: st.session_state.update(batch_text="\n".join(EXAMPLES.values())))
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
        st.download_button("Download results (CSV)", res.to_csv(index=False).encode(),
                           "spam_results.csv", "text/csv")
    else:
        st.info("Add some messages above to see a verdict for each one.")

# ------------------------------------------------------------------ performance
with tab_perf:
    y_te, p_te = model["y_te"], model["proba_te"]
    pred = (p_te >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    st.markdown(f"Measured live on **{len(y_te):,} held-out messages** the model never saw during "
                f"training (threshold {threshold:.2f}).")
    k = st.columns(5)
    k[0].metric("Accuracy", f"{accuracy_score(y_te, pred):.3f}")
    k[1].metric("Precision", f"{precision_score(y_te, pred):.3f}", help="Of messages flagged spam, how many were spam")
    k[2].metric("Recall", f"{recall_score(y_te, pred):.3f}", help="Of all real spam, how much was caught")
    k[3].metric("F1", f"{f1_score(y_te, pred):.3f}")
    k[4].metric("ROC-AUC", f"{roc_auc_score(y_te, p_te):.3f}")

    caught = 100 * tp / max(1, tp + fn)
    clean = 100 * tp / max(1, tp + fp)
    st.markdown(
        f'<div class="callout">In plain terms: out of every 100 real spam messages the model catches '
        f'<b>{caught:.0f}</b>, and out of every 100 messages it flags, <b>{clean:.0f}</b> really are spam. '
        f'On this test set that meant <b>{fp}</b> normal messages wrongly flagged and <b>{fn}</b> spam messages missed.</div>',
        unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        cm = go.Figure(go.Heatmap(
            z=[[1, 0], [0, 1]], x=["Predicted normal", "Predicted spam"], y=["Actually normal", "Actually spam"],
            text=[[tn, fp], [fn, tp]], texttemplate="%{text}", textfont={"size": 24, "color": INK},
            colorscale=[[0, SPAM_SOFT], [1, HAM_SOFT]], showscale=False, hoverinfo="skip", xgap=4, ygap=4))
        style_fig(cm, 340, title="Confusion matrix (green = correct, red = mistake)",
                  yaxis_autorange="reversed", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(cm, width="stretch")
    with c2:
        fpr, tpr, _ = roc_curve(y_te, p_te)
        prec, rec, _ = precision_recall_curve(y_te, p_te)
        curves = go.Figure()
        curves.add_scatter(x=fpr, y=tpr, name=f"ROC (AUC {roc_auc_score(y_te, p_te):.3f})",
                           line=dict(color=NAVY, width=3))
        curves.add_scatter(x=rec, y=prec, name=f"Precision-Recall (AP {average_precision_score(y_te, p_te):.3f})",
                           line=dict(color=SPAM_COLOR, width=3))
        curves.add_scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color=MUTED, dash="dot"))
        style_fig(curves, 340, title="Evaluation curves", xaxis_title="False positive rate / Recall",
                  yaxis_title="True positive rate / Precision", margin=dict(l=10, r=10, t=50, b=10),
                  legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(curves, width="stretch")

    st.subheader("What the threshold changes")
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
    def step(n, title, desc):
        return (f'<div class="step"><div class="step-n">{n}</div><div class="step-t">{title}</div>'
                f'<div class="step-d">{desc}</div></div>')

    st.markdown('<div class="steps">' + "".join([
        step(1, "Clean", f"{model['n_raw']:,} messages become {model['n_unique']:,} unique ones. Duplicates "
                         "are removed before splitting so the test set cannot leak into training."),
        step(2, "Vectorise", f"TF-IDF over words and word pairs, {model['vocab']:,} terms, English stop words removed."),
        step(3, "Classify", "Multinomial Naive Bayes with smoothing alpha = 0.01, tuned with 5-fold GridSearchCV on F1."),
        step(4, "Decide", "Flag the message as spam when P(spam | message) is at or above your threshold."),
    ]) + "</div>", unsafe_allow_html=True)

    st.subheader("The math in one line")
    st.code("log-odds(spam) = log P(spam)/P(ham)\n"
            "                + sum over words of  tfidf(word) * log P(word|spam)/P(word|ham)", language="text")
    st.write("The bar chart on the first tab plots exactly those per-word terms, so the explanation is the "
             "real calculation, not an approximation.")

    st.subheader("Known limitations")
    st.markdown(
        "- Word order and context are ignored (the \"naive\" independence assumption).\n"
        "- Probabilities are over-confident.\n"
        "- Short slang messages containing spam-flavoured words (e.g. *nokia*) can be false alarms.\n"
        "- Vocabulary is frozen at the training data, so new spam tactics are missed until you retrain.")
