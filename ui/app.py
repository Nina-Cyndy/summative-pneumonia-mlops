"""
PneumoScan: Streamlit console for the chest X-ray MLOps pipeline.

Pages: Diagnose (prediction) | Data Insights (visualisations) |
       Model Monitor (uptime + metrics) | Retrain Studio (bulk upload + retraining)

Run from the project root (API must be running):
    streamlit run ui/app.py
"""
import base64
import html
import inspect
import io
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

NAVY, INDIGO, TEAL, CORAL, AMBER, VIOLET, MUTED = (
    "#0B1F3A", "#4F46E5", "#14B8A6", "#F0605D", "#F59E0B", "#8B5CF6", "#64748B")
CLASS_COLORS = {"Normal": TEAL, "Pneumonia": CORAL}

st.set_page_config(page_title="PneumoScan · MLOps Console", page_icon="🫁",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(180deg,#F4F7FB 0%,#ECF1F8 100%); }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], .stDeployButton { display: none; }

/* sidebar */
[data-testid="stSidebar"] { background: linear-gradient(185deg,#0B1F3A 0%,#12385F 100%); }
[data-testid="stSidebar"] * { color: #DCE6F2; }
[data-testid="stSidebar"] .brand-name { color: #FFFFFF; }
[data-testid="stSidebar"] .brand-sub { color: #8FB0D6; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: 8px 10px; border-radius: 12px; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.08); }
.brand { display:flex; gap:12px; align-items:center; margin: 6px 0 20px; }
.brand-logo { width:44px; height:44px; border-radius:13px; display:flex; align-items:center;
  justify-content:center; font-size:23px; background: linear-gradient(135deg,#4F46E5,#14B8A6);
  box-shadow: 0 8px 18px rgba(20,184,166,.35); }
.brand-name { font-weight:800; font-size:18px; letter-spacing:-.01em; line-height:1.1; }
.brand-sub { font-size:11px; letter-spacing:.1em; text-transform:uppercase; }
.sbstat { background: rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.1);
  border-radius:14px; padding:12px 14px; margin-top:22px; font-size:13px; line-height:1.55; }
.sbnote { font-size:11.5px; color:#8FB0D6; margin-top:14px; line-height:1.5; }

/* hero */
.hero { display:flex; align-items:center; gap:18px; padding:26px 30px; border-radius:22px;
  background: linear-gradient(120deg,#0B1F3A 0%,#17406E 55%,#0E8F8D 100%); margin-bottom:24px;
  box-shadow: 0 14px 34px rgba(11,31,58,.22); position:relative; overflow:hidden; }
.hero:after { content:""; position:absolute; right:-70px; top:-70px; width:240px; height:240px;
  border-radius:50%; background: rgba(255,255,255,.07); }
.hero-icon { font-size:32px; width:62px; height:62px; flex:none; display:flex; align-items:center;
  justify-content:center; background: rgba(255,255,255,.14); border-radius:18px; }
.hero-title { font-size:28px; font-weight:800; letter-spacing:-.02em; color:#FFFFFF; line-height:1.15; }
.hero-sub { font-size:14.5px; color:#C9D8EA; margin-top:3px; }

/* section + step headers */
.sec-title { font-size:18px; font-weight:700; color:#0B1F3A; margin: 22px 0 2px; letter-spacing:-.01em; }
.sec-sub { font-size:13.5px; color:#64748B; margin-bottom:12px; }
.step { display:flex; align-items:center; gap:12px; margin: 26px 0 10px; }
.step-n { width:34px; height:34px; border-radius:11px; display:flex; align-items:center; justify-content:center;
  font-weight:800; color:#fff; background: linear-gradient(135deg,#4F46E5,#14B8A6); flex:none;
  box-shadow: 0 6px 14px rgba(79,70,229,.3); }
.step-t { font-size:18px; font-weight:700; color:#0B1F3A; line-height:1.2; }
.step-s { font-size:13px; color:#64748B; }

/* KPI tiles */
.kpi { background:#fff; border:1px solid #E6ECF4; border-top:4px solid var(--accent,#4F46E5);
  border-radius:16px; padding:15px 18px 14px; height:100%;
  box-shadow: 0 1px 2px rgba(11,31,58,.04), 0 8px 24px rgba(11,31,58,.05); }
.kpi-label { font-size:11.5px; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:#64748B; }
.kpi-value { font-size:27px; font-weight:800; letter-spacing:-.02em; color:#0B1F3A; margin-top:3px; line-height:1.15; }
.kpi-sub { font-size:12.5px; color:#64748B; margin-top:2px; }

/* generic bits */
.badge { display:inline-block; padding:4px 11px; border-radius:99px; font-size:12px; font-weight:700; white-space:nowrap; }
.b-ok { background:#DDF7F1; color:#0B7A6D; } .b-warn { background:#FEF0D5; color:#9A6200; }
.b-bad { background:#FDE3E2; color:#B3322F; } .b-info { background:#E5E9FF; color:#3730A3; }
.b-mute { background:#EDF1F6; color:#475569; }
.chip { display:inline-block; padding:5px 12px; border-radius:99px; background:#EEF2FF; color:#4338CA;
  font-size:12.5px; font-weight:600; margin: 6px 6px 0 0; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; background:#14B8A6;
  animation: pulse 2s infinite; }
.dot.red { background:#F0605D; animation:none; } .dot.amber { background:#F59E0B; }
@keyframes pulse { 0% { box-shadow:0 0 0 0 rgba(20,184,166,.55); } 70% { box-shadow:0 0 0 10px rgba(20,184,166,0); }
  100% { box-shadow:0 0 0 0 rgba(20,184,166,0); } }
.panel { background:#fff; border:1px solid #E6ECF4; border-radius:18px; padding:18px 22px;
  box-shadow: 0 1px 2px rgba(11,31,58,.04), 0 10px 28px rgba(11,31,58,.05); }
.banner { display:flex; align-items:center; justify-content:space-between; gap:12px; background:#fff;
  border:1px solid #E6ECF4; border-radius:16px; padding:14px 20px; margin-bottom:14px; font-weight:600;
  box-shadow: 0 1px 2px rgba(11,31,58,.04), 0 8px 24px rgba(11,31,58,.05); }
.insight { font-size:13.5px; line-height:1.55; color:#334155; background:#F8FAFD; border-left:3px solid #4F46E5;
  padding:10px 13px; border-radius:8px; margin-top:6px; }
.story { background: linear-gradient(160deg,#F5F3FF,#FFFFFF); border:1px solid #DDD6FE; border-radius:16px;
  padding:18px 22px; line-height:1.65; color:#1E293B; font-size:14.5px; }
.hint { border:2px dashed #C9D6E6; border-radius:16px; padding:34px 20px; text-align:center; color:#64748B;
  background: rgba(255,255,255,.6); font-size:14px; }
.hint .big { font-size:36px; margin-bottom:6px; }
.frame { border-radius:18px; overflow:hidden; background:#0B1F3A; margin:14px 0 12px;
  box-shadow: 0 12px 30px rgba(11,31,58,.18); }
.frame img { width:100%; display:block; max-height:460px; object-fit:contain; }

/* prediction result */
.result { border-radius:20px; padding:24px; border:1px solid; box-shadow: 0 12px 30px rgba(11,31,58,.08); }
.result.bad { background: linear-gradient(160deg,#FFF1F0,#FFFFFF); border-color:#F7C9C6; }
.result.good { background: linear-gradient(160deg,#E8FAF6,#FFFFFF); border-color:#BDEBE4; }
.result.warn { background: linear-gradient(160deg,#FFF8EA,#FFFFFF); border-color:#F6DFAE; }
.result-eyebrow { font-size:12px; text-transform:uppercase; letter-spacing:.12em; font-weight:700; color:#64748B; }
.result-label { font-size:31px; font-weight:800; letter-spacing:-.02em; margin:4px 0 0; }
.result.bad .result-label { color:#C23B38; } .result.good .result-label { color:#0B8A7B; }
.result.warn .result-label { color:#B26B00; }
.result-conf { font-size:15px; color:#475569; margin-bottom:14px; }
.prow { display:flex; align-items:center; gap:12px; margin:11px 0; }
.pname { width:92px; font-weight:600; font-size:14px; color:#0B1F3A; }
.ptrack { flex:1; height:12px; background:#E4EBF4; border-radius:99px; overflow:hidden; }
.pfill { height:100%; border-radius:99px; }
.pval { width:58px; text-align:right; font-weight:700; font-size:14px; color:#0B1F3A; }
.fine { font-size:12px; color:#64748B; margin-top:14px; line-height:1.5; }

/* tables */
.tbl-wrap { overflow-x:auto; }
.tbl { width:100%; border-collapse:collapse; font-size:13.5px; background:#fff; border-radius:16px; }
.tbl th { text-align:left; font-size:11.5px; text-transform:uppercase; letter-spacing:.08em; color:#64748B;
  padding:11px 12px; border-bottom:1px solid #E6ECF4; white-space:nowrap; }
.tbl td { padding:12px; border-bottom:1px solid #EFF3F8; vertical-align:top; color:#1E293B; }
.tbl .note { color:#64748B; font-size:12.5px; max-width:340px; }

/* widgets */
.stButton > button { width:100%; border-radius:12px; font-weight:600; padding:.65rem 1rem;
  border:1px solid #D2DCE9; background:#fff; color:#0B1F3A; transition: all .15s ease; }
.stButton > button:hover { border-color:#4F46E5; color:#4F46E5; transform: translateY(-1px); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg,#4F46E5,#0EA5A4); color:#fff; border:none;
  box-shadow: 0 10px 22px rgba(79,70,229,.30); }
.stButton > button[kind="primary"] p, .stButton > button[data-testid="stBaseButton-primary"] p { color:#fff; }
.stButton > button[kind="primary"]:hover { color:#fff; filter: brightness(1.06); }
.stButton > button:disabled { opacity:.5; transform:none; }
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"] {
  border:2px dashed #B8C6DA; border-radius:16px; background:#FBFDFF; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:#4F46E5; background:#F7F8FF; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def md(s: str) -> None:
    """Render an HTML snippet. Lines are stripped and joined so markdown never
    mistakes indented HTML for a code block (keep prose on ONE line)."""
    st.markdown("".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)


def fmt_uptime(sec) -> str:
    sec = int(sec)
    d, sec = divmod(sec, 86400)
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    parts = ([f"{d}d"] if d else []) + ([f"{h}h"] if (h or d) else []) + [f"{m}m"] + ([] if d else [f"{s}s"])
    return " ".join(parts)


def parse_ts(iso):
    try:
        return datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None


def fmt_ts(iso) -> str:
    dt = parse_ts(iso)
    return dt.strftime("%d %b · %H:%M UTC") if dt else "–"


def pct(v) -> str:
    return f"{v * 100:.1f}%"


def api(method: str, path: str, timeout: int = 8, **kw):
    """Returns (status_code, json). (None, None) when the API is unreachable."""
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kw)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {}
    except requests.RequestException:
        return None, None


@st.cache_data(ttl=15, show_spinner=False)
def get_json(path: str):
    code, data = api("GET", path)
    return data if code == 200 else None


def live(run_every: str):
    """Auto-refreshing fragment (works across Streamlit versions)."""
    frag = getattr(st, "fragment", None) or getattr(st, "experimental_fragment")
    return frag(run_every=run_every)


_PC = inspect.signature(st.plotly_chart).parameters
_PC_KW = {"width": "stretch"} if "width" in _PC else ({"use_container_width": True} if "use_container_width" in _PC else {})


def show(fig) -> None:
    st.plotly_chart(fig, config={"displayModeBar": False}, **_PC_KW)


def style_fig(fig, height=340):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=34, b=8), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=NAVY, size=13),
        legend=dict(orientation="h", y=1.14, x=0), hoverlabel=dict(font_family="Inter"))
    fig.update_xaxes(showgrid=False, linecolor="#D7E0EB")
    fig.update_yaxes(gridcolor="#E8EDF4", zeroline=False)
    return fig


def header(title, subtitle, icon):
    md(f'<div class="hero"><div class="hero-icon">{icon}</div><div><div class="hero-title">{esc(title)}</div>'
       f'<div class="hero-sub">{esc(subtitle)}</div></div></div>')


def section(title, sub=""):
    md(f'<div class="sec-title">{esc(title)}</div>' + (f'<div class="sec-sub">{esc(sub)}</div>' if sub else ""))


def step(n, title, sub=""):
    md(f'<div class="step"><div class="step-n">{n}</div><div><div class="step-t">{esc(title)}</div>'
       f'<div class="step-s">{esc(sub)}</div></div></div>')


def kpi(label, value, sub="", accent=INDIGO) -> str:
    return (f'<div class="kpi" style="--accent:{accent}"><div class="kpi-label">{esc(label)}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def kpi_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            md(kpi(*item))


def badge(text, kind="info") -> str:
    return f'<span class="badge b-{kind}">{esc(text)}</span>'


def preview_html(data: bytes) -> str:
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((900, 900))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'<div class="frame"><img src="data:image/jpeg;base64,{b64}"/></div>'
    except Exception:
        return '<div class="hint">Preview unavailable for this file.</div>'


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def effect_label(d: float) -> str:
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


NAV = ["🩻  Diagnose", "📊  Data Insights", "🛰️  Model Monitor", "🔁  Retrain Studio"]

code, health = api("GET", "/health", timeout=3)
online = code == 200

with st.sidebar:
    md('<div class="brand"><div class="brand-logo">🫁</div><div><div class="brand-name">PneumoScan</div>'
       '<div class="brand-sub">MLOps Console</div></div></div>')
    page = st.radio("Navigation", NAV, label_visibility="collapsed")
    if online:
        md(f'<div class="sbstat"><span class="dot"></span><b>API online</b><br>Up for {fmt_uptime(health["uptime_seconds"])}'
           f'<br>Model: {"loaded" if health["model_loaded"] else "not loaded"}</div>')
    else:
        md('<div class="sbstat"><span class="dot red"></span><b>API offline</b></div>')
    md('<div class="sbnote">Educational project using a public dataset. Not a medical device and not a substitute for a clinician.</div>')

if not online:
    header("API not reachable", f"The console could not connect to {API_URL}", "🔌")
    md('<div class="panel">Start the backend in another terminal from the project root, then refresh this page:'
       '<br><br><code>python -m uvicorn api.main:app --port 8000</code></div>')
    st.stop()


def result_html(out: dict) -> str:
    label = out["label"]
    bad = label.lower() == "pneumonia"
    headline = "Signs of pneumonia" if bad else "Looks normal"
    conf = out["confidence"]
    rows = ""
    for name, p in sorted(out["probabilities"].items(), key=lambda kv: -kv[1]):
        color = CLASS_COLORS.get(name, INDIGO)
        rows += (f'<div class="prow"><div class="pname">{esc(name)}</div><div class="ptrack">'
                 f'<div class="pfill" style="width:{p * 100:.1f}%;background:{color}"></div></div>'
                 f'<div class="pval">{p * 100:.1f}%</div></div>')
    chips = f'<span class="chip">⚡ {out["latency_ms"]:.0f} ms</span>'
    chips += f'<span class="chip">🧠 Model {esc(fmt_ts(out.get("model_version")))}</span>'
    if conf < 0.70:
        chips += '<span class="chip" style="background:#FEF0D5;color:#9A6200">⚠ Low confidence</span>'
    return (f'<div class="result {"bad" if bad else "good"}"><div class="result-eyebrow">Result</div>'
            f'<div class="result-label">{headline}</div><div class="result-conf">{pct(conf)} confidence</div>'
            f'{rows}<div>{chips}</div>'
            '<div class="fine">Educational model trained on a public dataset. This is not a medical diagnosis.</div></div>')


def page_diagnose():
    header("Diagnose", "Upload a chest X-ray and get an instant AI assessment", "🩻")
    left, right = st.columns(2, gap="large")
    with left:
        step(1, "Upload an X-ray", "JPG or PNG, up to 10 MB")
        up = st.file_uploader("X-ray image", type=["jpg", "jpeg", "png"], key="pred_upload",
                              label_visibility="collapsed")
        data, key = None, None
        if up is not None:
            data = up.getvalue()
            key = (up.name, len(data))
            md(preview_html(data))
            if st.button("Analyse X-ray", type="primary", key="analyse"):
                with st.spinner("Analysing…"):
                    c, out = api("POST", "/predict", timeout=60,
                                 files={"file": (up.name, data, up.type or "image/jpeg")})
                if c == 200:
                    st.session_state["pred"] = {"key": key, "out": out}
                else:
                    detail = (out or {}).get("detail", "The service did not respond.")
                    st.session_state["pred"] = {"key": key, "error": str(detail)}
    with right:
        step(2, "AI assessment", "Probability for each class")
        pred = st.session_state.get("pred")
        if key is not None and pred and pred["key"] == key:
            if "error" in pred:
                md(f'<div class="panel"><span class="badge b-bad">Error</span><p style="margin:10px 0 0">{esc(pred["error"])}</p></div>')
            else:
                md(result_html(pred["out"]))
        else:
            md('<div class="hint"><div class="big">🔬</div>Upload an X-ray and press <b>Analyse</b>.<br>'
               'The result will appear here.</div>')


FEATURE_INFO = {
    "brightness": ("Overall brightness", "Mean pixel intensity of the whole image. It also depends on the X-ray machine, patient size and positioning, so it is a noisy signal."),
    "contrast": ("Contrast", "Spread of light and dark pixels. Healthy lungs are dark, air-filled regions next to bright bones; fluid can flatten that difference."),
    "centre_brightness": ("Central-lung brightness", "Mean intensity of the middle of the image, where the lungs sit. Fluid and inflammation show up as brighter, cloudier patches."),
}


def describe_feature(df: pd.DataFrame, col: str, pos="Pneumonia", neg="Normal"):
    a, b = df.loc[df["class"] == pos, col], df.loc[df["class"] == neg, col]
    if a.empty or b.empty:
        return "Not enough data to compare the classes.", 0.0
    d = cohens_d(a, b)
    direction = "brighter / higher" if d > 0 else "darker / lower"
    text = (f"{pos} images average <b>{a.mean():.1f}</b> vs <b>{b.mean():.1f}</b> for {neg} ({direction}). "
            f"Effect size d = {d:+.2f} ({effect_label(d)}), so the two groups "
            f"{'overlap heavily' if abs(d) < 0.5 else 'are clearly different but still overlap'}.")
    return text, d


def page_insights():
    header("Data Insights", "What the X-ray dataset looks like, and what a model can learn from it", "📊")
    summ, ins = get_json("/data/summary"), get_json("/insights")
    if not summ or not ins:
        md('<div class="panel">Dataset information is not available yet. Check that <code>data/train</code> exists.</div>')
        return

    train, test, ups = summ["train"], summ["test"], summ["uploads"]
    n_tr, n_te, n_up = sum(train.values()), sum(test.values()), sum(ups.values())
    ratio = train.get("Pneumonia", 0) / max(train.get("Normal", 1), 1)
    kpi_row([("Training images", f"{n_tr:,}", "used to train the model", INDIGO),
             ("Test images", f"{n_te:,}", "never seen during training", TEAL),
             ("Uploaded by users", f"{n_up:,}", "added for retraining", AMBER),
             ("Pneumonia : Normal", f"{ratio:.2f} : 1", "class ratio in training data", CORAL)])

    section("1 · Class balance", "How many images of each class are in each part of the data")
    classes = sorted(set(train) | set(test) | set(ups))
    fig = go.Figure()
    for name, d, color in (("Train", train, INDIGO), ("Test", test, TEAL), ("Uploaded", ups, AMBER)):
        vals = [d.get(c, 0) for c in classes]
        fig.add_bar(name=name, x=classes, y=vals, marker_color=color, text=vals, textposition="outside")
    fig.update_layout(barmode="group", bargap=0.35)
    show(style_fig(fig, 330))
    md(f'<div class="insight"><b>What it means:</b> pneumonia images outnumber normal ones ({ratio:.2f} to 1 in training). '
       'A model that always answered "Pneumonia" would already look decent on accuracy alone, so accuracy is not enough. '
       'We use class weights while training and judge the model with precision, recall, F1 and AUC.</div>')

    df = pd.DataFrame(ins["features"])
    section("2 · What the images look like in numbers",
            f"Three simple measurements on a random sample of {ins['sample_per_class']} images per class")
    cols = st.columns(3, gap="medium")
    effects = {}
    for col_ui, feat in zip(cols, FEATURE_INFO):
        title, meaning = FEATURE_INFO[feat]
        with col_ui:
            fig = go.Figure()
            for cls in sorted(df["class"].unique()):
                color = CLASS_COLORS.get(cls, INDIGO)
                fig.add_trace(go.Violin(y=df.loc[df["class"] == cls, feat], name=cls, box_visible=True,
                                        meanline_visible=True, line_color=color, fillcolor=color, opacity=0.6))
            fig.update_layout(showlegend=False, title=dict(text=title, font=dict(size=15)))
            show(style_fig(fig, 300))
            text, d = describe_feature(df, feat)
            effects[title] = d
            md(f'<div class="insight">{text}<br><span style="color:#64748B">{esc(meaning)}</span></div>')

    best, d_best = max(effects.items(), key=lambda kv: abs(kv[1]))
    if abs(d_best) < 1.0:
        verdict = ("No single measurement separates the two classes cleanly: the strongest signal is "
                   f"<b>{esc(best)}</b> (d = {d_best:+.2f}, {effect_label(d_best)}), and the distributions overlap. ")
    else:
        verdict = (f"<b>{esc(best)}</b> separates the classes fairly well (d = {d_best:+.2f}, {effect_label(d_best)}), "
                   "but it still overlaps and cannot be used alone. ")
    section("3 · The story")
    md('<div class="story"><b>What the data tells us.</b> ' + verdict +
       'Simple summary statistics also depend on equipment and exposure, so they are unreliable. '
       'The real difference lies in <i>where</i> and in <i>what shape</i> the cloudy patches appear inside the lungs, '
       'which is a spatial pattern. That is exactly what a convolutional neural network learns, and why we use '
       'transfer learning with MobileNetV2 instead of a classic model on summary numbers.</div>')


def page_monitor():
    header("Model Monitor", "Live service health and how well the model in production performs", "🛰️")

    @live("5s")
    def live_health():
        c, h = api("GET", "/health", timeout=3)
        if c != 200:
            md('<div class="banner"><span><span class="dot red"></span>Service unreachable</span></div>')
            return
        busy = h["retraining"]
        md(f'<div class="banner"><span><span class="dot {"amber" if busy else ""}"></span>'
           f'{"Online · retraining in progress" if busy else "Online · all systems operational"}</span>'
           f'<span style="color:#64748B;font-weight:500">Started {esc(fmt_ts(h["started_at"]))}</span></div>')
        avg = h["avg_latency_ms"]
        mv = parse_ts(h["model_version"])
        mv_val = mv.strftime("%d %b") if mv else "–"
        mv_sub = (mv.strftime("%H:%M UTC") + " · " if mv else "") + ("loaded" if h["model_loaded"] else "not loaded")
        kpi_row([("Uptime", fmt_uptime(h["uptime_seconds"]), "since the API last started", TEAL),
                 ("Predictions served", f'{h["predictions_served"]:,}', f'{h["prediction_errors"]} errors · {h.get("replicas_online", 1)} replica(s)', INDIGO),
                 ("Avg latency", f"{avg:.0f} ms" if avg else "–", "per prediction", AMBER),
                 ("Model version", mv_val, mv_sub, VIOLET)])
        md('<div class="fine">Prediction counters are cumulative across all inference replicas. Uptime refers to the control service.</div>')

    live_health()

    m = get_json("/metrics")
    section("Model performance", "Measured on the held-out test set. These numbers describe the model currently in production.")
    if not m:
        md('<div class="panel">No evaluation saved yet.</div>')
        return
    kpi_row([("Accuracy", pct(m["accuracy"]), "share of correct predictions", INDIGO),
             ("Precision", pct(m["precision"]), "predicted pneumonia that was right", TEAL),
             ("Recall", pct(m["recall"]), "real pneumonia that was caught", CORAL)])
    st.write("")
    kpi_row([("F1 score", pct(m["f1"]), "balance of precision and recall", VIOLET),
             ("AUC", f'{m["auc"]:.3f}', "ranking quality (1.0 is perfect)", AMBER),
             ("Loss", f'{m["loss"]:.3f}', "lower is better", MUTED)])

    names, cm = m["class_names"], np.array(m["confusion_matrix"])
    pos = names.index(m.get("positive_class", "Pneumonia"))
    neg = 1 - pos
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        section("Confusion matrix")
        text = [[f"{v}<br>{v / max(row.sum(), 1):.0%}" for v in row] for row in cm]
        fig = go.Figure(go.Heatmap(z=cm, x=[f"Predicted {n}" for n in names], y=[f"Actual {n}" for n in names],
                                   colorscale=[[0, "#EEF2FF"], [1, INDIGO]], showscale=False,
                                   text=text, texttemplate="%{text}", textfont=dict(size=17)))
        fig.update_yaxes(autorange="reversed")
        show(style_fig(fig, 330))
    with right:
        section("In plain words")
        tp, fn, fp, tn = cm[pos][pos], cm[pos][neg], cm[neg][pos], cm[neg][neg]
        md(f'<div class="story">Of <b>{tp + fn}</b> real {esc(names[pos])} cases the model caught <b>{tp}</b> '
           f'and missed <b>{fn}</b>.<br><br>Of <b>{tn + fp}</b> {esc(names[neg])} X-rays it correctly cleared <b>{tn}</b> '
           f'and raised <b>{fp}</b> false alarms.<br><br>For screening, missing a case is the costlier mistake, '
           'so a model that leans cautious (high recall, more false alarms) is the safer trade-off.</div>')


def outcome_badge(run: dict) -> str:
    if run["status"] == "running":
        return badge("Running", "info")
    if run["status"] == "finished":
        return badge("Promoted", "ok") if run["promoted"] else badge("Kept old model", "warn")
    return badge(run["status"].capitalize(), "bad" if run["status"] == "failed" else "mute")


def runs_table(runs: list) -> str:
    rows = ""
    for r in runs:
        old, new = r.get("old_metrics"), r.get("new_metrics")
        f1 = f'{old["f1"]:.3f} → {new["f1"]:.3f}' if old and new else "–"
        imgs = f'{r["n_new"]} + {r["n_replay"]}' if r["n_new"] is not None else "–"
        rows += (f'<tr><td><b>#{r["id"]}</b></td><td>{esc(fmt_ts(r["started_at"]))}</td><td>{esc(r["trigger_type"] or "–")}</td>'
                 f'<td>{imgs}</td><td>{r["epochs"] if r["epochs"] is not None else "–"}</td><td>{f1}</td>'
                 f'<td>{outcome_badge(r)}</td><td class="note">{esc(r["message"] or "")}</td></tr>')
    head = "".join(f"<th>{h}</th>" for h in ["Run", "Started", "Trigger", "Images (new + replay)", "Epochs",
                                              "F1 before → after", "Outcome", "Note"])
    return f'<div class="tbl-wrap"><table class="tbl"><tr>{head}</tr>{rows}</table></div>'


def compare_chart(run: dict):
    order = ["accuracy", "precision", "recall", "f1", "auc"]
    old, new = run["old_metrics"], run["new_metrics"]
    fig = go.Figure()
    fig.add_bar(name="Previous model", x=[k.upper() if k == "auc" else k.capitalize() for k in order],
                y=[old[k] for k in order], marker_color="#A5B4FC",
                text=[f"{old[k]:.3f}" for k in order], textposition="outside")
    fig.add_bar(name="Retrained model", x=[k.upper() if k == "auc" else k.capitalize() for k in order],
                y=[new[k] for k in order], marker_color=TEAL if run["promoted"] else CORAL,
                text=[f"{new[k]:.3f}" for k in order], textposition="outside")
    fig.update_traces(textfont=dict(size=11))
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.06)
    fig.update_yaxes(range=[0, 1.12])
    return style_fig(fig, 330)


def bulk_upload(files, label):
    totals = {"saved": 0, "duplicate": 0, "invalid": 0}
    started, pending, chunk_size = False, None, 25
    prog = st.progress(0.0, text="Uploading…")
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        payload = [("files", (f.name, f.getvalue(), f.type or "image/jpeg")) for f in chunk]
        c, out = api("POST", "/upload", timeout=180, data={"label": label}, files=payload)
        if c != 200:
            prog.empty()
            return {"error": str((out or {}).get("detail", "Upload failed."))}
        for k in totals:
            totals[k] += out.get(k, 0)
        started = started or out.get("auto_retrain_started", False)
        pending = out.get("pending_for_retraining")
        prog.progress(min(1.0, (i + len(chunk)) / len(files)), text=f"Uploaded {min(i + chunk_size, len(files))} of {len(files)}")
    prog.empty()
    return {**totals, "label": label, "pending": pending, "auto_started": started}


def page_retrain():
    header("Retrain Studio", "Add new labelled X-rays and teach the model with them", "🔁")

    step(1, "Upload new training images", "All files in one upload must belong to the same class")
    c, summ = api("GET", "/data/summary")
    classes = sorted((summ or {}).get("train", {}).keys()) or ["Normal", "Pneumonia"]
    label = st.radio("Class of these images", classes, horizontal=True)
    md('<div class="fine" style="margin:-4px 0 8px">The label must be correct: a wrongly labelled image teaches the model the wrong thing.</div>')
    key = f"bulk_{st.session_state.get('bulk_key', 0)}"
    files = st.file_uploader("Training images", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                             key=key, label_visibility="collapsed")
    if files:
        md(f'<div class="fine">{len(files)} file(s) selected as <b>{esc(label)}</b></div>')
    if st.button(f"Upload {len(files)} image(s)" if files else "Upload images", type="primary",
                 disabled=not files, key="do_upload"):
        st.session_state["upload_result"] = bulk_upload(files, label)
        st.session_state["bulk_key"] = st.session_state.get("bulk_key", 0) + 1
        st.rerun()
    res = st.session_state.pop("upload_result", None)
    if res:
        if "error" in res:
            md(f'<div class="panel"><span class="badge b-bad">Upload failed</span><p style="margin:10px 0 0">{esc(res["error"])}</p></div>')
        else:
            kpi_row([("Saved", res["saved"], f'as {esc(res["label"])}', TEAL),
                     ("Duplicates skipped", res["duplicate"], "already uploaded before", AMBER),
                     ("Invalid files", res["invalid"], "not readable as images", CORAL),
                     ("Waiting for retraining", res["pending"], "pending uploads", INDIGO)])
            if res["auto_started"]:
                md('<div class="insight">The automatic trigger fired: enough new images arrived, so retraining has started.</div>')

    @live("3s")
    def studio_status():
        cs, state = api("GET", "/retrain/status", timeout=5)
        ch, hist = api("GET", "/retrain/history?limit=10", timeout=5)
        if cs != 200:
            md('<div class="panel">Could not read the retraining status.</div>')
            return
        running = state["running"]
        pending, thr = state["pending_uploads"], state["auto_retrain_threshold"]

        step(2, "Trigger retraining", "Starts automatically at a threshold, or press the button")
        a, b, c3 = st.columns(3, gap="medium")
        with a:
            md(kpi("Pending uploads", pending, "not yet used for training", INDIGO))
        with b:
            share = min(100, pending / thr * 100) if thr else 0
            md(f'<div class="kpi" style="--accent:{AMBER}"><div class="kpi-label">Auto-trigger</div>'
               f'<div class="kpi-value">{pending} / {thr}</div><div class="ptrack" style="margin-top:8px">'
               f'<div class="pfill" style="width:{share:.0f}%;background:linear-gradient(90deg,{INDIGO},{TEAL})"></div></div>'
               f'<div class="kpi-sub">{"enabled" if state["auto_retrain"] else "disabled"}</div></div>')
        with c3:
            if running:
                started = parse_ts(state.get("started_at"))
                el = fmt_uptime((datetime.now(timezone.utc) - started).total_seconds()) if started else ""
                md(kpi("Status", '<span class="dot amber"></span>Retraining', f"running for {el}", AMBER))
            elif state.get("last_error"):
                md(kpi("Status", '<span class="dot red"></span>Failed', esc(state["last_error"])[:80], CORAL))
            else:
                md(kpi("Status", '<span class="dot"></span>Idle', "ready for a new run", TEAL))
        st.write("")
        if st.button("Start retraining now", type="primary", key="start_retrain", disabled=running):
            c, out = api("POST", "/retrain", timeout=10)
            if c == 200:
                st.toast("Retraining started", icon="🚀")
            else:
                md(f'<div class="insight" style="border-color:{CORAL}">{esc((out or {}).get("detail", "Could not start retraining."))}</div>')

        step(3, "Review the result", "The new model replaces the old one only if it is not worse on the test set")
        runs = hist if ch == 200 and hist else []
        if not runs:
            md('<div class="hint"><div class="big">🧪</div>No retraining runs yet.<br>Upload some images and start one.</div>')
            return
        last = runs[0]
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            tone = ("bad" if last["status"] == "failed"
                    else "good" if (last["promoted"] or last["status"] == "running") else "warn")
            title = {"running": "Retraining in progress", "failed": "Retraining failed"}.get(
                last["status"], "New model promoted" if last["promoted"] else "Old model kept")
            md(f'<div class="result {tone}"><div class="result-eyebrow">Latest run #{last["id"]}</div>'
               f'<div class="result-label" style="font-size:24px">{esc(title)}</div>'
               f'<div class="result-conf" style="margin-top:8px">{esc(last["message"] or "")}</div>'
               f'<span class="chip">🆕 {last["n_new"] if last["n_new"] is not None else "–"} new images</span>'
               f'<span class="chip">♻ {last["n_replay"] if last["n_replay"] is not None else "–"} replayed</span>'
               f'<span class="chip">⏱ {last["epochs"] if last["epochs"] is not None else "–"} epochs</span></div>')
        with right:
            if last.get("old_metrics") and last.get("new_metrics"):
                show(compare_chart(last))
        section("Run history")
        md(runs_table(runs))

    studio_status()


{NAV[0]: page_diagnose, NAV[1]: page_insights, NAV[2]: page_monitor, NAV[3]: page_retrain}[page]()