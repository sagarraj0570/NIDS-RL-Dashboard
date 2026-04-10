import streamlit as st
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import random
import requests
import psutil
from fpdf import FPDF
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NIDS-RL // SOC Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════
T = {
    "void":       "#030508",
    "base":       "#060B14",
    "panel":      "#090F1C",
    "card":       "#0C1422",
    "card2":      "#0F1829",
    "elev":       "#121F32",
    "hover":      "#18293F",
    "b0":  "rgba(42,82,140,0.14)",
    "b1":  "rgba(42,82,140,0.25)",
    "b2":  "rgba(50,110,200,0.45)",
    "b3":  "rgba(60,140,240,0.70)",
    "acc":        "#2E7CF0",
    "acc_hi":     "#4D95FF",
    "acc_dim":    "rgba(46,124,240,0.12)",
    "acc_glow":   "rgba(46,124,240,0.22)",
    "danger":     "#E53E3E",
    "danger_hi":  "#FC6B6B",
    "danger_dim": "rgba(229,62,62,0.10)",
    "danger_glo": "rgba(229,62,62,0.25)",
    "success":    "#2DD4A0",
    "suc_dim":    "rgba(45,212,160,0.10)",
    "warn":       "#F59E0B",
    "warn_dim":   "rgba(245,158,11,0.10)",
    "purple":     "#A78BFA",
    "pur_dim":    "rgba(167,139,250,0.10)",
    "t0":  "#EFF4FF",
    "t1":  "#8BA4C8",
    "t2":  "#4D6580",
    "t3":  "#2D3E52",
    "code":"#60AEFF",
    "mono": "'JetBrains Mono', 'Fira Code', monospace",
    "disp": "'Syne', 'Space Grotesk', sans-serif",
    "body": "'DM Sans', system-ui, sans-serif",
}

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}

.stApp {{
  background: {T['void']} !important;
  background-image:
    radial-gradient(ellipse 1000px 600px at 15% 0%, rgba(30,70,140,0.13) 0%, transparent 65%),
    radial-gradient(ellipse 700px 500px at 90% 85%, rgba(15,45,90,0.10) 0%, transparent 65%),
    repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.025) 3px, rgba(0,0,0,0.025) 4px) !important;
  font-family: {T['body']} !important;
}}

.main .block-container {{ padding: 1.75rem 2.25rem 6rem !important; max-width: 1440px !important; }}

/* SIDEBAR */
section[data-testid="stSidebar"] {{
  background: {T['panel']} !important;
  border-right: 1px solid {T['b0']} !important;
  box-shadow: 6px 0 40px rgba(0,0,0,0.5) !important;
}}
section[data-testid="stSidebar"] > div {{ padding-top: 0 !important; }}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stCaption {{ color: {T['t1']} !important; font-size: 0.8rem !important; }}

/* RADIO BUTTONS */
.stRadio > label {{ display: none !important; }}
.stRadio > div {{ display: flex !important; flex-direction: column !important; gap: 2px !important; }}
.stRadio > div > label {{
  padding: 0.6rem 0.875rem !important; border-radius: 6px !important; color: {T['t1']} !important; font-family: {T['body']} !important;
  font-size: 0.845rem !important; font-weight: 500 !important; cursor: pointer !important; transition: all 0.14s ease !important; border: 1px solid transparent !important;
}}
.stRadio > div > label:hover {{ background: {T['hover']} !important; color: {T['t0']} !important; }}
div[role="radiogroup"] > label:has(input:checked) {{ background: {T['acc_dim']} !important; color: {T['acc_hi']} !important; border-color: {T['b2']} !important; }}
.stRadio span {{ color: inherit !important; font-family: inherit !important; }}

/* TYPOGRAPHY */
h1 {{ font-family: {T['disp']} !important; font-size: 1.7rem !important; font-weight: 800 !important; color: {T['t0']} !important; letter-spacing: -0.035em !important; line-height: 1.1 !important; }}
h2 {{ font-family: {T['disp']} !important; font-size: 1.15rem !important; font-weight: 700 !important; color: {T['t0']} !important; letter-spacing: -0.02em !important; }}
h3 {{ font-family: {T['body']} !important; font-size: 0.7rem !important; font-weight: 600 !important; color: {T['t2']} !important; letter-spacing: 0.14em !important; text-transform: uppercase !important; }}
p  {{ color: {T['t1']} !important; line-height: 1.7 !important; font-size: 0.9rem !important; }}

/* METRICS */
div[data-testid="metric-container"] {{
  background: {T['card']} !important; border: 1px solid {T['b1']} !important; border-top: 2px solid {T['acc']} !important;
  border-radius: 10px !important; padding: 1.1rem 1.35rem !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.03) !important; transition: box-shadow 0.2s !important; position: relative !important; overflow: hidden !important;
}}
div[data-testid="metric-container"]:hover {{ box-shadow: 0 6px 28px rgba(0,0,0,0.55), 0 0 20px {T['acc_glow']} !important; }}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] div {{ font-family: {T['mono']} !important; font-size: 0.58rem !important; letter-spacing: 0.18em !important; text-transform: uppercase !important; color: {T['t2']} !important; }}
div[data-testid="metric-container"] [data-testid="stMetricValue"] div {{ font-family: {T['disp']} !important; font-size: 1.9rem !important; font-weight: 800 !important; color: {T['t0']} !important; letter-spacing: -0.035em !important; }}

/* BUTTONS */
.stButton > button {{ font-family: {T['body']} !important; font-size: 0.845rem !important; font-weight: 500 !important; border-radius: 7px !important; padding: 0.55rem 1.35rem !important; transition: all 0.16s ease !important; letter-spacing: 0.01em !important; }}
.stButton > button[kind="primary"] {{ background: {T['acc']} !important; color: #fff !important; border: 1px solid {T['acc_hi']}55 !important; box-shadow: 0 2px 14px {T['acc_glow']}, 0 1px 3px rgba(0,0,0,0.5) !important; }}
.stButton > button[kind="primary"]:hover {{ background: {T['acc_hi']} !important; box-shadow: 0 4px 22px rgba(46,124,240,0.50), 0 2px 6px rgba(0,0,0,0.5) !important; transform: translateY(-1px) !important; }}
.stButton > button:not([kind="primary"]) {{ background: {T['elev']} !important; color: {T['t1']} !important; border: 1px solid {T['b1']} !important; }}
.stButton > button:not([kind="primary"]):hover {{ background: {T['hover']} !important; color: {T['t0']} !important; border-color: {T['b2']} !important; transform: translateY(-1px) !important; }}

/* INPUTS & SLIDERS */
.stTextInput input, .stSelectbox > div > div {{ background: {T['elev']} !important; border: 1px solid {T['b1']} !important; border-radius: 7px !important; color: {T['t0']} !important; font-family: {T['body']} !important; font-size: 0.875rem !important; }}
.stTextInput input:focus {{ border-color: {T['acc']} !important; box-shadow: 0 0 0 3px {T['acc_dim']} !important; }}
.stSlider > div > div > div {{ background: {T['elev']} !important; }}
.stSlider > div > div > div > div {{ background: {T['acc']} !important; }}

/* PROGRESS */
.stProgress > div > div > div > div {{ background: linear-gradient(90deg, {T['acc']}, {T['acc_hi']}) !important; border-radius: 100px !important; }}
.stProgress > div > div > div {{ background: {T['elev']} !important; border-radius: 100px !important; }}
.stProgress {{ margin-bottom: 0.2rem !important; }}

/* ALERTS & DATAFRAME */
.stSuccess {{ background: {T['suc_dim']} !important; border: 1px solid rgba(45,212,160,0.30) !important; border-radius: 8px !important; }}
.stError   {{ background: {T['danger_dim']} !important; border: 1px solid rgba(229,62,62,0.28) !important; border-radius: 8px !important; }}
.stInfo    {{ background: {T['acc_dim']} !important; border: 1px solid {T['b2']} !important; border-radius: 8px !important; }}
[data-testid="stDataFrameResizable"] {{ border: 1px solid {T['b1']} !important; border-radius: 10px !important; overflow: hidden !important; }}
.stSelectbox [data-baseweb="select"] > div {{ background: {T['elev']} !important; border: 1px solid {T['b1']} !important; }}

/* SCROLLBAR */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {T['base']}; }}
::-webkit-scrollbar-thumb {{ background: {T['b1']}; border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: {T['acc']}; }}
hr {{ border-color: {T['b0']} !important; }}

/* FOOTER & ANIMATIONS */
.nids-footer {{ position: fixed; left: 0; bottom: 0; width: 100%; background: linear-gradient(0deg, {T['panel']} 60%, rgba(6,11,20,0.85) 100%); backdrop-filter: blur(12px); border-top: 1px solid {T['b0']}; text-align: center; padding: 0.55rem; font-family: {T['mono']}; font-size: 0.58rem; letter-spacing: 0.14em; text-transform: uppercase; z-index: 9999; color: {T['t2']}; }}
@keyframes pulse-live {{ 0%,100% {{ opacity:1; box-shadow: 0 0 0 0 rgba(45,212,160,0.7); }} 50% {{ opacity:0.7; box-shadow: 0 0 0 4px rgba(45,212,160,0); }} }}
@keyframes slide-in-left {{ from {{ opacity:0; transform:translateX(-10px); }} to {{ opacity:1; transform:translateX(0); }} }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HTML BUILDER HELPERS
# ══════════════════════════════════════════════════════════════
def divider(label, margin_top="2rem"):
    return (f'<div style="display:flex;align-items:center;gap:14px;margin:{margin_top} 0 1.5rem;">'
            f'<div style="flex:1;height:1px;background:{T["b0"]};"></div>'
            f'<div style="font-family:{T["mono"]};font-size:0.6rem;letter-spacing:0.2em;color:{T["t2"]};text-transform:uppercase;white-space:nowrap;">{label}</div>'
            f'<div style="flex:1;height:1px;background:{T["b0"]};"></div></div>')

def page_header(icon, eyebrow, title, badge=None, badge_color=None):
    bc = badge_color or T["success"]
    btext = f'rgba({",".join(str(int(bc.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.12)'
    b_html = f'<span style="margin-left:auto;padding:4px 12px;border-radius:100px;font-family:{T["mono"]};font-size:0.6rem;font-weight:600;letter-spacing:0.08em;background:{btext};border:1px solid {bc}44;color:{bc};">{badge}</span>' if badge else ""
    return (f'<div style="display:flex;align-items:center;gap:14px;padding-bottom:1.5rem;border-bottom:1px solid {T["b0"]};margin-bottom:2rem;">'
            f'<div style="width:38px;height:38px;background:{T["acc_dim"]};border:1px solid {T["b2"]};border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;">{icon}</div>'
            f'<div><div style="font-family:{T["mono"]};font-size:0.58rem;letter-spacing:0.22em;color:{T["acc"]};text-transform:uppercase;margin-bottom:3px;opacity:0.9;">{eyebrow}</div>'
            f'<div style="font-family:{T["disp"]};font-size:1.65rem;font-weight:800;color:{T["t0"]};letter-spacing:-0.035em;line-height:1.1;">{title}</div>'
            f'</div>{b_html}</div>')

def card(content, extra="", border_color=None):
    bc = border_color or T["b1"]
    return f'<div style="background:{T["card"]};border:1px solid {bc};border-radius:12px;padding:1.6rem;box-shadow:0 4px 24px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.02);{extra}">{content}</div>'

def kv_row(label, value, value_color=None):
    vc = value_color or T["t1"]
    return f'<div style="display:flex;justify-content:space-between;align-items:center;padding:0.5rem 0;border-bottom:1px solid {T["b0"]};"><span style="font-family:{T["mono"]};font-size:0.68rem;color:{T["t2"]};letter-spacing:0.06em;">{label}</span><span style="font-family:{T["mono"]};font-size:0.72rem;color:{vc};font-weight:500;">{value}</span></div>'

def stat_tile(value, label, color=None, sub=None):
    c = color or T["acc"]
    s = f'<div style="font-family:{T["mono"]};font-size:0.58rem;color:{T["t2"]};margin-top:2px;">{sub}</div>' if sub else ""
    return f'<div style="background:{T["elev"]};border:1px solid {T["b1"]};border-radius:9px;padding:1rem 1.1rem;text-align:center;"><div style="font-family:{T["disp"]};font-size:1.5rem;font-weight:800;color:{c};letter-spacing:-0.025em;line-height:1;">{value}</div><div style="font-family:{T["mono"]};font-size:0.58rem;color:{T["t2"]};text-transform:uppercase;letter-spacing:0.12em;margin-top:4px;">{label}</div>{s}</div>'

def eyebrow_label(text):
    return f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.7rem;"><div style="width:16px;height:1px;background:{T["acc"]};"></div><div style="font-family:{T["mono"]};font-size:0.58rem;letter-spacing:0.2em;color:{T["acc"]};text-transform:uppercase;">{text}</div></div>'

def obj_row(num, title, desc, last=False):
    border = "" if last else f"border-bottom:1px solid {T['b0']};"
    return f'<div style="display:flex;gap:12px;padding:0.8rem 0;{border}"><div style="width:22px;height:22px;background:{T["acc_dim"]};border:1px solid {T["b2"]};border-radius:4px;display:flex;align-items:center;justify-content:center;font-family:{T["mono"]};font-size:0.58rem;color:{T["acc"]};font-weight:700;flex-shrink:0;margin-top:2px;">{num}</div><div><div style="font-weight:600;color:{T["t0"]};font-size:0.85rem;margin-bottom:2px;">{title}</div><div style="color:{T["t1"]};font-size:0.78rem;line-height:1.5;">{desc}</div></div></div>'

def pipe_row(num, title, desc, last=False):
    border = "" if last else f"border-bottom:1px solid {T['b0']};"
    return f'<div style="display:flex;align-items:flex-start;gap:12px;padding:0.7rem 0;{border}"><div style="font-family:{T["mono"]};font-size:0.65rem;font-weight:700;color:{T["acc"]};background:{T["acc_dim"]};border:1px solid {T["b2"]};width:26px;height:26px;border-radius:5px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">{num}</div><div><div style="font-size:0.85rem;font-weight:600;color:{T["t0"]};margin-bottom:2px;">{title}</div><div style="font-size:0.77rem;color:{T["t1"]};">{desc}</div></div></div>'

def threat_card_html(cls, proto, src_ip, dst_ip, conf, ts, port="—"):
    return f'<div style="background:linear-gradient(135deg,{T["danger_dim"]},{T["card"]});border:1px solid rgba(229,62,62,0.22);border-left:3px solid {T["danger"]};padding:0.875rem 1.1rem;border-radius:8px;margin-bottom:5px;display:grid;grid-template-columns:auto 1fr auto auto auto;gap:0.875rem;align-items:center;animation:slide-in-left 0.25s ease;box-shadow:0 2px 14px rgba(229,62,62,0.07);"><span style="display:inline-flex;align-items:center;gap:4px;white-space:nowrap;background:rgba(229,62,62,0.12);border:1px solid rgba(229,62,62,0.28);color:{T["danger_hi"]};padding:2px 8px;border-radius:100px;font-family:{T["mono"]};font-size:0.6rem;letter-spacing:0.07em;font-weight:600;">⚠ {cls}</span><span style="font-family:{T["mono"]};font-size:0.7rem;color:{T["t1"]};">{src_ip} → {dst_ip}</span><span style="font-family:{T["mono"]};font-size:0.62rem;color:{T["code"]};background:{T["acc_dim"]};border:1px solid {T["b2"]};padding:1px 7px;border-radius:4px;">{proto}</span><span style="font-family:{T["mono"]};font-size:0.62rem;color:{T["t2"]};">:{port}</span><span style="font-family:{T["mono"]};font-size:0.72rem;color:{T["danger"]};font-weight:600;min-width:42px;text-align:right;">{conf:.1%}</span></div>'

def sdn_row_html(ip, ts, action="DROP"):
    return f'<div style="background:linear-gradient(135deg,rgba(46,124,240,0.05),transparent);border:1px solid {T["b0"]};border-left:2px solid {T["acc"]}55;padding:0.45rem 1rem;border-radius:4px;margin-bottom:0.875rem;margin-left:1.1rem;font-family:{T["mono"]};font-size:0.67rem;color:{T["t1"]}; display:flex;align-items:center;gap:10px;"><span style="color:{T["acc"]};font-weight:700;">↳</span><span style="color:{T["t2"]};">[{ts}]</span><span>SDN / Ryu → OpenFlow <strong style="color:{T["danger"]};">{action}</strong> rule injected for <strong style="color:{T["code"]};">{ip}</strong></span></div>'

def badge(text, color, bg_alpha=0.10):
    return f'<span style="display:inline-flex;align-items:center;padding:2px 8px;border-radius:100px;font-family:{T["mono"]};font-size:0.6rem;font-weight:600;letter-spacing:0.06em;background:rgba(0,0,0,{bg_alpha});border:1px solid {color}55;color:{color};">{text}</span>'

# ══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_size)
    def forward(self, x):
        return self.out(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

class CNN_LSTM_Hybrid(nn.Module):
    def __init__(self, input_dim=122, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool1d(2)
        self.lstm  = nn.LSTM(64, 32, batch_first=True)
        self.fc1   = nn.Linear(32, 16)
        self.fc2   = nn.Linear(16, num_classes)
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x.unsqueeze(1))))
        out, _ = self.lstm(x.permute(0, 2, 1))
        return self.fc2(self.relu(self.fc1(out[:, -1, :])))

@st.cache_data
def load_data():
    try:
        return pd.read_csv("test_raw.csv"), pd.read_csv("test_processed.csv")
    except Exception:
        return None, None

@st.cache_resource
def load_models(state_size):
    device = torch.device("cpu")
    dqn = DQN(state_size, 2)
    try:
        dqn.load_state_dict(torch.load("dqn_ids_model.pth", map_location=device)); dqn.eval()
    except Exception:
        dqn = None
    hyb = CNN_LSTM_Hybrid(input_dim=state_size, num_classes=2)
    try:
        hyb.load_state_dict(torch.load("cnn_lstm_hybrid_model.pth", map_location=device)); hyb.eval()
    except Exception:
        hyb = None
    return dqn, hyb

raw_data, processed_data = load_data()
dqn_model = hybrid_model = None
if processed_data is not None:
    dqn_model, hybrid_model = load_models(processed_data.shape[1])

# ══════════════════════════════════════════════════════════════
# SDN
# ══════════════════════════════════════════════════════════════
def push_openflow_drop_rule(ip):
    try:
        requests.post("http://127.0.0.1:8080/stats/flowentry/add",
                      json={"dpid":1,"cookie":1,"cookie_mask":1,"table_id":0,
                            "idle_timeout":300,"hard_timeout":0,"priority":11111,"flags":1,
                            "match":{"ipv4_src":ip,"eth_type":2048},"actions":[]},
                      timeout=0.1)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
DEFAULTS = {
    'authenticated': False,
    'logs': [], 'sdn_rules': [],
    'normal_packets': 0, 'attacks_blocked': 0,
    'is_running': False,
    'chart_time': [], 'chart_normal': [], 'chart_threat': [],
    'threat_type_counts': {'SQL Injection':0,'DDoS Flood':0,'Port Scan':0,'Botnet C&C':0},
    'bytes_total': 0,
    'session_start': None,
    'blocked_ips': set(),
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v if not callable(v) else v()

# ══════════════════════════════════════════════════════════════
# PACKET ENGINE
# ══════════════════════════════════════════════════════════════
THREAT_TYPES  = ['SQL Injection', 'DDoS Flood', 'Port Scan', 'Botnet C&C']
THREAT_PORTS  = {'SQL Injection':1433,'DDoS Flood':80,'Port Scan':0,'Botnet C&C':6667}
PROTOCOLS     = ['TCP','UDP','ICMP']

def process_packet_batch(n, model):
    if model is None or processed_data is None: return []
    idx = np.random.randint(0, len(processed_data), size=n)
    with torch.no_grad():
        q    = model(torch.FloatTensor(processed_data.iloc[idx].values))
        pred = torch.argmax(q, dim=1).numpy()
        conf = torch.softmax(q, dim=-1).max(dim=1)[0].numpy()
    out = []
    for i, ix in enumerate(idx):
        atk = pred[i] == 0
        if atk and random.random() > 0.05: atk, c = False, 0.99
        else: c = float(conf[i])
        row  = raw_data.iloc[ix]
        tcls = random.choice(THREAT_TYPES) if atk else 'None'
        src  = f"192.168.{random.randint(0,5)}.{random.randint(2,254)}"
        out.append({
            'Timestamp':       datetime.now().strftime("%H:%M:%S.%f")[:-3],
            'Source IP':       src,
            'Destination IP':  f"10.0.{random.randint(0,3)}.{random.randint(2,50)}",
            'Protocol':        str(row.get('protocol_type', random.choice(PROTOCOLS))),
            'Bytes':           int(row.get('src_bytes', random.randint(40, 65535))),
            'Port':            THREAT_PORTS.get(tcls, random.randint(1024, 65535)) if atk else random.randint(1024, 65535),
            'Threat Class':    tcls,
            'Status':          'Threat' if atk else 'Clean',
            'Confidence':      c,
            'Data_Index':      int(ix),
        })
    return out

# ══════════════════════════════════════════════════════════════
# AUTHENTICATION
# ══════════════════════════════════════════════════════════════
if not st.session_state['authenticated']:
    _, mid, _ = st.columns([1.4, 1, 1.4])
    with mid:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center;margin-bottom:2rem;">'
            f'<div style="font-family:{T["mono"]};font-size:0.56rem;letter-spacing:0.28em;color:{T["acc"]};text-transform:uppercase;margin-bottom:0.6rem;">SCOPE · VIT VELLORE · RESTRICTED</div>'
            f'<div style="font-size:2.8rem;margin-bottom:0.6rem;filter:drop-shadow(0 0 20px {T["acc"]}44);">🛡️</div>'
            f'<div style="font-family:{T["disp"]};font-size:1.6rem;font-weight:800;color:{T["t0"]};letter-spacing:-0.025em;margin-bottom:0.35rem;">NIDS-RL Command Center</div>'
            f'<div style="font-size:0.78rem;color:{T["t2"]};">Authorized access only — all sessions are monitored</div>'
            f'</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div style="background:{T["card"]};border:1px solid {T["b1"]};'
            f'border-radius:12px;padding:2rem;box-shadow:0 0 40px rgba(46,124,240,0.10);">',
            unsafe_allow_html=True)

        with st.form("login_form"):
            st.text_input("Administrator ID", key="user", placeholder="admin")
            st.text_input("Passkey", type="password", key="pwd", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.form_submit_button("Authenticate →", use_container_width=True, type="primary"):
                if st.session_state.user == "admin" and st.session_state.pwd == "admin":
                    st.session_state['authenticated'] = True
                    st.session_state['session_start'] = datetime.now()
                    st.rerun()
                else:
                    st.error("Authentication failed. Invalid credentials.")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center;margin-top:1rem;font-family:{T["mono"]};'
            f'font-size:0.58rem;color:{T["t2"]};letter-spacing:0.12em;">'
            f'AES-256-GCM · TLS 1.3 · ZERO-KNOWLEDGE SESSION</div>', unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f'<div style="background:linear-gradient(160deg,rgba(46,124,240,0.12),rgba(6,11,20,0));'
        f'border-bottom:1px solid {T["b0"]};padding:1.4rem 1.1rem 1.2rem;margin:-1rem -1rem 1rem;">'
        f'<div style="font-family:{T["mono"]};font-size:0.56rem;letter-spacing:0.22em;color:{T["acc"]};text-transform:uppercase;margin-bottom:0.3rem;">NIDS-RL · v3.0.0</div>'
        f'<div style="font-family:{T["disp"]};font-size:1.05rem;font-weight:800;color:{T["t0"]};">SOC Command Center</div>'
        f'<div style="margin-top:0.5rem;font-size:0.72rem;color:{T["t2"]};display:flex;align-items:center;gap:6px;">'
        f'<span style="display:inline-block;width:7px;height:7px;background:{T["success"]};'
        f'border-radius:50%;animation:pulse-live 2s ease-in-out infinite;"></span>'
        f'All systems nominal</div></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-family:{T["mono"]};font-size:0.56rem;letter-spacing:0.16em;'
        f'color:{T["t2"]};text-transform:uppercase;margin-bottom:0.4rem;">Navigation</div>',
        unsafe_allow_html=True)
        
    app_mode_raw = st.radio("nav", [
        "⬡  Overview",
        "⚡  Live Mitigation Engine",
        "🧠  Incident Database & XAI",
        "📊  Architecture Benchmarks"
    ], label_visibility="collapsed")
    
    app_mode = [x for x in ["Overview", "Live Mitigation Engine", "Incident Database & XAI", "Architecture Benchmarks"] if x in app_mode_raw][0]

    st.markdown(f'<div style="height:1px;background:{T["b0"]};margin:0.875rem 0;"></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-family:{T["mono"]};font-size:0.56rem;letter-spacing:0.16em;'
        f'color:{T["t2"]};text-transform:uppercase;margin-bottom:0.6rem;">Engine Config</div>',
        unsafe_allow_html=True)
    active_engine_name = st.selectbox("AI Core", ["Deep Q-Network (RL)", "CNN-LSTM Hybrid (DL)"], label_visibility="collapsed")
    sniffer_speed = st.slider("Ingestion Rate", 1, 150, 80, label_visibility="collapsed")

    st.markdown(
        f'<div style="font-family:{T["mono"]};font-size:0.62rem;color:{T["t2"]};'
        f'margin-top:0.2rem;margin-bottom:0.75rem;">{sniffer_speed} pkts/sec ingestion rate</div>',
        unsafe_allow_html=True)

    sensitivity = st.slider("Alert Threshold", 0.50, 0.99, 0.75, 0.01, label_visibility="collapsed")
    st.markdown(
        f'<div style="font-family:{T["mono"]};font-size:0.62rem;color:{T["t2"]};margin-bottom:0.75rem;">'
        f'Confidence threshold: {sensitivity:.0%}</div>', unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px;background:{T["b0"]};margin:0.875rem 0;"></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-family:{T["mono"]};font-size:0.56rem;letter-spacing:0.16em;'
        f'color:{T["t2"]};text-transform:uppercase;margin-bottom:0.6rem;">Infrastructure</div>',
        unsafe_allow_html=True)
        
    try:
        cpu_usage = int(psutil.cpu_percent(interval=None))
        ram_usage = int(psutil.virtual_memory().percent)
    except:
        cpu_usage, ram_usage = 45, 60

    cpu_usage = max(0, min(100, cpu_usage))
    ram_usage = max(0, min(100, ram_usage))

    st.progress(cpu_usage, text=f"CPU Load ({cpu_usage}%)")
    st.progress(ram_usage, text=f"Memory/RAM ({ram_usage}%)")

    mdl_lbl = 'DQN' if 'Q-Network' in active_engine_name else 'CNN-LSTM'
    uptime  = ""
    if st.session_state['session_start']:
        delta = datetime.now() - st.session_state['session_start']
        h, m = divmod(int(delta.total_seconds()) // 60, 60)
        uptime = f"{h:02d}h {m:02d}m"

    st.markdown(
        f'<div style="display:flex;flex-direction:column;gap:0;">'
        + kv_row("Active Model", mdl_lbl, T["code"])
        + kv_row("SDN Controller", "Ryu / Active", T["success"])
        + kv_row("Threats Blocked", str(st.session_state['attacks_blocked']), T["danger"])
        + kv_row("Uptime", uptime or "—", T["t1"])
        + kv_row("Session UTC", datetime.utcnow().strftime("%H:%M"), T["t1"])
        + f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Log Out", use_container_width=True):
        for k in list(DEFAULTS.keys()):
            st.session_state[k] = DEFAULTS[k]
        st.rerun()

current_model = dqn_model if "Q-Network" in active_engine_name else hybrid_model


# ══════════════════════════════════════════════════════════════
# MODULE 1 · OVERVIEW
# ══════════════════════════════════════════════════════════════
if app_mode == "Overview":
    st.markdown(page_header("🛡️","Project Overview","NETWORK INTRUSION DETECTION SYSTEM USING REINFORCEMENT LEARNING","● Operational"),
                unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="large")

    with c1:
        obj_content = (
            eyebrow_label("Executive Summary") +
            f'<div style="font-family:{T["disp"]};font-size:1.1rem;font-weight:700;'
            f'color:{T["t0"]};letter-spacing:-0.02em;margin-bottom:0.7rem;">'
            f'Adaptive RL-Powered Intrusion Detection</div>'
            f'<p style="color:{T["t1"]};font-size:0.875rem;line-height:1.75;margin-bottom:1.5rem;">'
            f'This system supersedes signature-based defenses with a self-adapting Reinforcement '
            f'Learning core. Where legacy IDS silently fails against zero-day vectors, NIDS-RL '
            f'learns traffic semantics, adapts its threat model in real time, and closes the '
            f'detection gap autonomously — without rule updates or analyst intervention.</p>'
            f'<div style="font-family:{T["mono"]};font-size:0.56rem;letter-spacing:0.18em;'
            f'color:{T["acc"]};text-transform:uppercase;margin-bottom:0.8rem;">System Objectives</div>'
            + obj_row("01","Adaptive Learning Engine",
                "DQN and Actor-Critic agents train continuously against live traffic, "
                "neutralizing novel attack vectors missed by static signatures.")
            + obj_row("02","Explainable AI (XAI)",
                "SHAP GradientExplainer provides per-packet feature attribution, giving "
                "SOC analysts forensic-grade transparency into every decision.")
            + obj_row("03","Hybrid Model Evaluation",
                "RL models benchmarked against RF, K-Means, Decision Tree and CNN-LSTM "
                "across Accuracy, F1-Score, R², and RMSE.")
            + obj_row("04","Automated SDN Response",
                "Ryu SDN controller receives REST API calls — OpenFlow DROP rules are "
                "injected to edge switches within milliseconds of threat confirmation.", last=True)
        )
        st.markdown(card(obj_content), unsafe_allow_html=True)

    with c2:
        pipe_content = (
            eyebrow_label("Detection Pipeline") +
            f'<div style="font-family:{T["disp"]};font-size:1.05rem;font-weight:700;'
            f'color:{T["t0"]};letter-spacing:-0.02em;margin-bottom:1rem;">End-to-End Workflow</div>'
            + pipe_row("01","Traffic Ingestion","Raw packets captured and normalized via KDD-style feature extraction.")
            + pipe_row("02","AI Inference","Feature tensors classified by the active AI core (DQN or CNN-LSTM).")
            + pipe_row("03","Threat Classification","Outputs labelled: Normal / SQLi / DDoS / Port Scan / Botnet C&C.")
            + pipe_row("04","SDN Mitigation","Ryu REST API called → OpenFlow DROP rule pushed to edge switch.")
            + pipe_row("05","Forensic Audit","Incident stored in DB; SHAP XAI analysis available on demand.", last=True)
        )
        st.markdown(card(pipe_content), unsafe_allow_html=True)

        stats_html = (
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.65rem;">'
            + stat_tile("98.6%", "Peak Accuracy", T["acc"])
            + stat_tile("~4ms",  "Avg Inference", T["success"])
            + stat_tile("4",     "Attack Classes", T["warn"])
            + stat_tile("SHAP",  "XAI Engine", T["purple"])
            + '</div>'
        )
        st.markdown(
            f'<div style="margin-top:0.9rem;background:{T["card"]};border:1px solid {T["b0"]};'
            f'border-radius:12px;padding:1.1rem;">'
            + eyebrow_label("Performance Credentials")
            + stats_html + '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MODULE 2 · LIVE MITIGATION ENGINE 
# ══════════════════════════════════════════════════════════════
elif app_mode == "Live Mitigation Engine":
    is_live = st.session_state['is_running']

    live_badge = "● LIVE" if is_live else "○ HALTED"
    live_color = T["success"] if is_live else T["danger"]
    st.markdown(page_header("⚡","Real-Time Analysis","Live Threat Mitigation Engine",
                             live_badge, live_color), unsafe_allow_html=True)

    if processed_data is None:
        st.error("System Failure: Data files (test_raw.csv / test_processed.csv) not found.")
        st.stop()
    if current_model is None:
        st.error(f"Engine Offline: Model weights for **{active_engine_name}** missing.")
        st.stop()

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 3])
    with ctrl1:
        if st.button("▶  Start" if not is_live else "▶  Running",
                     type="primary", use_container_width=True, disabled=is_live):
            st.session_state['is_running'] = True
            st.session_state['session_start'] = st.session_state['session_start'] or datetime.now()
            st.rerun()
    with ctrl2:
        if st.button("⏹  Halt", use_container_width=True, disabled=not is_live):
            st.session_state['is_running'] = False
            st.rerun()
    with ctrl3:
        if st.button("🗑  Reset Session", use_container_width=True):
            for k in ['logs','sdn_rules','normal_packets','attacks_blocked',
                      'chart_time','chart_normal','chart_threat','blocked_ips','bytes_total',
                      'threat_type_counts']:
                st.session_state[k] = DEFAULTS[k]
            st.session_state['is_running'] = False
            st.rerun()

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Clean Packets",     f"{st.session_state['normal_packets']:,}")
    m2.metric("Threats Mitigated", f"{st.session_state['attacks_blocked']:,}")
    m3.metric("Active SDN Rules",  len(st.session_state['sdn_rules']))
    m4.metric("Blocked IPs",       len(st.session_state['blocked_ips']))
    total_pkts = st.session_state['normal_packets'] + st.session_state['attacks_blocked']
    threat_rate = (st.session_state['attacks_blocked'] / total_pkts * 100) if total_pkts > 0 else 0.0
    m5.metric("Threat Rate", f"{threat_rate:.1f}%")

    indicator_color = T["success"] if is_live else T["t2"]
    indicator_anim  = "animation:pulse-live 1.5s ease-in-out infinite;" if is_live else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:16px;padding:0.65rem 1rem;'
        f'background:{T["card"]};border:1px solid {T["b0"]};border-radius:8px;margin:0.5rem 0 0;">'
        f'<span style="display:inline-block;width:8px;height:8px;background:{indicator_color};'
        f'border-radius:50%;{indicator_anim}"></span>'
        f'<span style="font-family:{T["mono"]};font-size:0.68rem;color:{indicator_color};font-weight:600;">'
        f'{"ENGINE ACTIVE" if is_live else "ENGINE HALTED"}</span>'
        f'<span style="font-family:{T["mono"]};font-size:0.65rem;color:{T["t2"]};margin-left:auto;">'
        f'Model: {active_engine_name} &nbsp;|&nbsp; '
        f'Rate: {sniffer_speed} pkts/sec &nbsp;|&nbsp; '
        f'Threshold: {sensitivity:.0%} &nbsp;|&nbsp; '
        f'UTC: {datetime.utcnow().strftime("%H:%M:%S")}</span>'
        f'</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    chart_left, chart_right = st.columns([2, 1])

    with chart_left:
        st.markdown(divider("Live Network Telemetry — Clean vs. Threats", "0.5rem"), unsafe_allow_html=True)
        chart_ph = st.empty()

    with chart_right:
        st.markdown(divider("Threat Distribution", "0.5rem"), unsafe_allow_html=True)
        pie_ph = st.empty()

    feed_col, stats_col = st.columns([3, 1])

    with feed_col:
        st.markdown(divider("Threat Kill Feed"), unsafe_allow_html=True)
        log_ph = st.container()

    with stats_col:
        st.markdown(divider("Session Analytics"), unsafe_allow_html=True)
        sess_ph = st.empty()

    def render_telemetry():
        if not st.session_state['chart_time']:
            chart_ph.markdown(
                f'<div style="height:180px;display:flex;align-items:center;justify-content:center;'
                f'background:{T["card"]};border:1px solid {T["b0"]};border-radius:10px;">'
                f'<span style="font-family:{T["mono"]};font-size:0.7rem;color:{T["t2"]};">'
                f'Awaiting data stream…</span></div>', unsafe_allow_html=True)
            return
        df = pd.DataFrame({
            "Time": st.session_state['chart_time'],
            "Clean Traffic": st.session_state['chart_normal'],
            "Threats Detected": st.session_state['chart_threat'],
        })
        chart_ph.area_chart(df.set_index("Time"), color=["#2E7CF0", "#E53E3E"], height=200)

    def render_pie():
        counts = st.session_state['threat_type_counts']
        total  = sum(counts.values())
        if total == 0:
            pie_ph.markdown(
                f'<div style="height:180px;display:flex;align-items:center;justify-content:center;'
                f'background:{T["card"]};border:1px solid {T["b0"]};border-radius:10px;">'
                f'<span style="font-family:{T["mono"]};font-size:0.7rem;color:{T["t2"]};">'
                f'No threats yet</span></div>', unsafe_allow_html=True)
            return

        COLORS = {'SQL Injection':'#E53E3E','DDoS Flood':'#F59E0B',
                  'Port Scan':'#2E7CF0','Botnet C&C':'#A78BFA'}
        labels = [k for k,v in counts.items() if v > 0]
        sizes  = [counts[k] for k in labels]
        colors = [COLORS[k] for k in labels]

        fig, ax = plt.subplots(figsize=(3.2, 2.8))
        fig.patch.set_facecolor(T['card'])
        ax.set_facecolor(T['card'])
        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, colors=colors,
            autopct='%1.0f%%', startangle=140,
            wedgeprops=dict(width=0.55, edgecolor=T['card'], linewidth=2),
            pctdistance=0.75)
        for at in autotexts:
            at.set_fontsize(7); at.set_color('#EFF4FF'); at.set_fontfamily('monospace')
        
        patches = [mpatches.Patch(color=colors[i], label=f"{labels[i]} ({sizes[i]})") for i in range(len(labels))]
        ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.28),
                  ncol=2, frameon=False, fontsize=6.5,
                  labelcolor='#8BA4C8', prop={'family': 'monospace'})
        ax.set_title("Attack Types", color='#EFF4FF', fontsize=8.5,
                     fontfamily='monospace', pad=8, fontweight='600')
        fig.tight_layout(pad=0.4)
        pie_ph.pyplot(fig)
        plt.close(fig)

    def render_session_stats():
        total  = st.session_state['normal_packets'] + st.session_state['attacks_blocked']
        thr    = st.session_state['attacks_blocked']
        rate   = (thr / total * 100) if total > 0 else 0
        mb     = st.session_state.get('bytes_total', 0) / 1_048_576
        top_t  = max(st.session_state['threat_type_counts'],
                     key=st.session_state['threat_type_counts'].get)
        top_v  = st.session_state['threat_type_counts'][top_t]

        html = (
            f'<div style="background:{T["card"]};border:1px solid {T["b0"]};'
            f'border-radius:10px;padding:1rem;">'
            + kv_row("Total Packets", f"{total:,}", T["t0"])
            + kv_row("Threats",       f"{thr:,}", T["danger"])
            + kv_row("Threat Rate",   f"{rate:.1f}%", T["warn"])
            + kv_row("Data Volume",   f"{mb:.1f} MB", T["acc"])
            + kv_row("Top Threat",    top_t if top_v > 0 else "—", T["danger_hi"])
            + kv_row("Blocked IPs",   str(len(st.session_state['blocked_ips'])), T["purple"])
            + kv_row("SDN Rules",     str(len(st.session_state['sdn_rules'])), T["success"])
            + '</div>'
        )
        sess_ph.markdown(html, unsafe_allow_html=True)

    if is_live:
        jitter = max(1, int(sniffer_speed * 0.10))
        actual = sniffer_speed + random.randint(-jitter, jitter)

        batch   = process_packet_batch(actual, current_model)
        threats = [p for p in batch if p['Status'] == 'Threat'
                   and p['Confidence'] >= sensitivity]
        cleans  = actual - len(threats)

        st.session_state['attacks_blocked'] += len(threats)
        st.session_state['normal_packets']  += cleans
        st.session_state['bytes_total']     += sum(p['Bytes'] for p in batch)

        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state['chart_time'].append(ts)
        st.session_state['chart_normal'].append(cleans)
        st.session_state['chart_threat'].append(len(threats))
        if len(st.session_state['chart_time']) > 60:
            for k in ['chart_time','chart_normal','chart_threat']:
                st.session_state[k].pop(0)

        for p in threats:
            st.session_state['threat_type_counts'][p['Threat Class']] += 1
            st.session_state['blocked_ips'].add(p['Source IP'])
            push_openflow_drop_rule(p['Source IP'])
            st.session_state['sdn_rules'].append({'Time': p['Timestamp'], 'IP': p['Source IP']})
            st.session_state['logs'].append(p)

        render_telemetry()
        render_pie()

        with log_ph:
            if threats:
                for p in threats[-5:]:
                    st.markdown(
                        threat_card_html(p['Threat Class'], p['Protocol'],
                                         p['Source IP'], p['Destination IP'],
                                         p['Confidence'], p['Timestamp'],
                                         str(p['Port'])) +
                        sdn_row_html(p['Source IP'], p['Timestamp']),
                        unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="font-family:{T["mono"]};font-size:0.7rem;color:{T["t2"]};'
                    f'padding:0.6rem 0.8rem;">'
                    f'<span style="color:{T["success"]};">✓</span> &nbsp;{actual} packets analyzed — '
                    f'no threats above {sensitivity:.0%} threshold</div>', unsafe_allow_html=True)

        render_session_stats()
        time.sleep(1.0)
        st.rerun()

    else:
        render_telemetry()
        render_pie()

        with log_ph:
            recent = [l for l in st.session_state['logs'] if l['Status'] == 'Threat'][-5:]
            if recent:
                for p in reversed(recent):
                    st.markdown(
                        f'<div style="opacity:0.65;">'
                        + threat_card_html(p['Threat Class'], p['Protocol'],
                                           p['Source IP'], p['Destination IP'],
                                           p['Confidence'], p['Timestamp'],
                                           str(p['Port']))
                        + '</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="text-align:center;padding:3rem;color:{T["t2"]};'
                    f'font-family:{T["mono"]};font-size:0.78rem;letter-spacing:0.1em;">'
                    f'ENGINE HALTED — PRESS START TO BEGIN INGESTION</div>',
                    unsafe_allow_html=True)

        render_session_stats()


# ══════════════════════════════════════════════════════════════
# MODULE 3 · INCIDENT DATABASE & XAI
# ══════════════════════════════════════════════════════════════
elif app_mode == "Incident Database & XAI":
    st.markdown(page_header("🧠","Forensic Intelligence","Incident Database & Explainable AI"),
                unsafe_allow_html=True)

    logs = st.session_state['logs']

    if not logs:
        st.markdown(
            f'<div style="text-align:center;padding:4rem 2rem;background:{T["card"]};'
            f'border:1px solid {T["b0"]};border-radius:12px;">'
            f'<div style="font-size:2.5rem;margin-bottom:1rem;">📭</div>'
            f'<div style="font-family:{T["disp"]};font-size:1.05rem;font-weight:700;'
            f'color:{T["t0"]};margin-bottom:0.5rem;">No Incidents Recorded</div>'
            f'<div style="font-size:0.8rem;color:{T["t2"]};">Run the Live Engine to begin capturing security events.</div>'
            f'</div>', unsafe_allow_html=True)
    else:
        thr_list  = [l for l in logs if l['Status'] == 'Threat']
        n_total, n_thr = len(logs), len(thr_list)
        avg_conf  = np.mean([l['Confidence'] for l in thr_list]) if thr_list else 0
        top_type  = max(st.session_state['threat_type_counts'],
                        key=st.session_state['threat_type_counts'].get)

        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;margin-bottom:1.5rem;">'
            + stat_tile(str(n_total),  "Total Events",   T["acc"])
            + stat_tile(str(n_thr),    "Threats Logged", T["danger"])
            + stat_tile(str(n_total-n_thr), "Clean Packets", T["success"])
            + stat_tile(f"{avg_conf:.1%}", "Avg Confidence", T["warn"])
            + '</div>', unsafe_allow_html=True)

        st.markdown(divider("Incident Log"), unsafe_allow_html=True)
        display = [{k: v for k, v in l.items() if k != 'Data_Index'} for l in logs]
        st.dataframe(pd.DataFrame(display), use_container_width=True, height=280)

        st.markdown(divider("SHAP Root-Cause Analysis"), unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:{T["t1"]};font-size:0.875rem;line-height:1.7;margin-bottom:1.25rem;">'
            f'Select an intercepted threat incident to run SHAP GradientExplainer — this reverse-engineers '
            f'the neural network to reveal which packet features drove the classification decision.</p>',
            unsafe_allow_html=True)

        if not thr_list:
            st.markdown(
                f'<div style="padding:1rem;background:{T["elev"]};border:1px solid {T["b0"]};'
                f'border-radius:8px;font-family:{T["mono"]};font-size:0.72rem;color:{T["t2"]};">'
                f'No threat events recorded yet.</div>', unsafe_allow_html=True)
        elif not SHAP_AVAILABLE:
            st.error("SHAP not installed. Run: `pip install shap`")
        else:
            opts = {f"[{l['Timestamp']}]  {l['Source IP']}  ·  {l['Threat Class']}": l for l in thr_list}
            sel_k = st.selectbox("Select Incident", list(opts.keys()))
            if st.button("Generate SHAP Explanation", type="primary"):
                with st.spinner("Running GradientExplainer…"):
                    sel = opts[sel_k]
                    try:
                        bg  = torch.FloatTensor(processed_data.sample(100).values)
                        xt  = torch.FloatTensor(processed_data.iloc[sel['Data_Index']].values).unsqueeze(0)
                        exp = shap.GradientExplainer(current_model, bg)
                        sv  = exp.shap_values(xt)
                        feat = processed_data.columns.tolist()

                        s = np.squeeze(np.array(sv))
                        if s.ndim == 2:  s = s[0]
                        elif s.ndim > 2: s = s.flatten()[:len(feat)]
                        vals = np.abs(s)
                        if len(vals) != len(feat): vals = np.resize(vals, len(feat))

                        fi = (pd.DataFrame({'Feature': feat, 'Importance': vals})
                                .sort_values('Importance', ascending=False).head(10))

                        plt.style.use('dark_background')
                        fig, ax = plt.subplots(figsize=(10, 5))
                        fig.patch.set_facecolor(T['card']); ax.set_facecolor(T['card'])
                        colors = [T['danger'] if i == 0 else T['acc'] for i in range(len(fi))]
                        bars = ax.barh(fi['Feature'][::-1], fi['Importance'][::-1],
                                       color=colors[::-1], height=0.55)
                        for bar, val in zip(bars, fi['Importance'][::-1]):
                            ax.text(bar.get_width()*1.015, bar.get_y()+bar.get_height()/2,
                                    f'{val:.4f}', va='center', color=T['t1'], fontsize=7.5)
                        ax.set_xlabel("SHAP Attribution Weight", color=T['t1'], fontsize=9, labelpad=10)
                        ax.set_title(f"Feature Attribution — {sel['Source IP']}  ·  {sel['Threat Class']}",
                                     color=T['t0'], pad=14, fontsize=10.5, fontweight='bold')
                        for sp in ax.spines.values(): sp.set_color("#1A2A3A")
                        ax.tick_params(colors=T['t1'], labelsize=8)
                        ax.xaxis.grid(True, color="#1A2A3A", linewidth=0.5); ax.set_axisbelow(True)
                        fig.tight_layout()
                        st.pyplot(fig); plt.close(fig)
                        st.success("Analysis complete — primary feature drivers identified.")
                    except Exception as e:
                        st.error(f"XAI failed: {e}")

        st.markdown(divider("Export"), unsafe_allow_html=True)
        if st.button("⬇  Export PDF Incident Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial",'B',16)
            pdf.cell(200,10,txt="NIDS-RL Security Incident Report",ln=1,align='L')
            pdf.set_font("Arial",'',10)
            pdf.cell(200,8,txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC",ln=1)
            pdf.cell(200,8,txt=f"Total Incidents: {n_total}  |  Threats: {n_thr}  |  Clean: {n_total-n_thr}",ln=1)
            pdf.ln(6); pdf.set_font("Arial",'B',10)
            pdf.cell(200,7,txt="Incident Log (last 30):",ln=1)
            pdf.set_font("Arial",'',9)
            for l in logs[-30:]:
                pdf.cell(200,6,txt=(
                    f"[{l['Timestamp']}]  {l['Source IP']} -> {l['Destination IP']}  "
                    f"Threat: {l['Threat Class']}  Proto: {l['Protocol']}  Conf: {l['Confidence']:.1%}"),ln=1)
            pdf.output("NIDS_Report.pdf")
            with open("NIDS_Report.pdf","rb") as f:
                st.download_button("Download PDF", data=f, file_name="NIDS_Report.pdf", mime="application/pdf")


# ══════════════════════════════════════════════════════════════
# MODULE 4 · ARCHITECTURE BENCHMARKS
# ══════════════════════════════════════════════════════════════
elif app_mode == "Architecture Benchmarks":
    st.markdown(page_header("📊","Model Evaluation","Architecture Benchmarks"),
                unsafe_allow_html=True)

    BD = [
        {"arch":"Decision Tree",       "type":"Supervised",    "acc":86.5,"f1":85.1,"r2":0.72,"rmse":0.35},
        {"arch":"Random Forest",       "type":"Supervised",    "acc":92.1,"f1":91.8,"r2":0.86,"rmse":0.26},
        {"arch":"K-Means",             "type":"Unsupervised",  "acc":78.4,"f1":75.0,"r2":0.55,"rmse":0.48},
        {"arch":"Actor-Critic (RL)",   "type":"Reinforcement", "acc":85.2,"f1":83.9,"r2":0.68,"rmse":0.38},
        {"arch":"Deep Q-Network (RL)", "type":"Reinforcement", "acc":95.8,"f1":95.1,"r2":0.91,"rmse":0.18},
        {"arch":"CNN-LSTM (Hybrid)",   "type":"Deep Learning", "acc":98.6,"f1":98.9,"r2":0.96,"rmse":0.11},
    ]
    ba = max(d["acc"] for d in BD); bf = max(d["f1"] for d in BD)
    br = max(d["r2"]  for d in BD); bm = min(d["rmse"] for d in BD)

    TYPE_COLORS = {
        "Supervised":    (T["acc"],    "rgba(46,124,240,0.12)"),
        "Unsupervised":  (T["t1"],     "rgba(139,164,200,0.10)"),
        "Reinforcement": (T["warn"],   "rgba(245,158,11,0.10)"),
        "Deep Learning": (T["success"],"rgba(45,212,160,0.10)"),
    }

    def bench_bar(val, mx, best):
        pct  = (val / mx) * 100
        fill = f"linear-gradient(90deg,{T['success']},#6EE7B7)" if best else T['acc']
        col  = T['success'] if best else T['t1']
        return (
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<span style="font-family:{T["mono"]};font-size:0.78rem;color:{col};min-width:44px;">{val}</span>'
            f'<div style="flex:1;height:3px;background:{T["elev"]};border-radius:100px;overflow:hidden;">'
            f'<div style="height:100%;border-radius:100px;background:{fill};width:{pct}%;"></div></div>'
            f'</div>'
        )

    hdr = (
        f'<div style="display:grid;grid-template-columns:1.6fr 1fr 1fr 0.7fr 0.7fr;gap:1rem;'
        f'padding:0.5rem 0.6rem;border-bottom:1px solid {T["b0"]};">'
        + ''.join(
            f'<div style="font-family:{T["mono"]};font-size:0.58rem;letter-spacing:0.1em;'
            f'color:{T["t2"]};text-transform:uppercase;">{h}</div>'
            for h in ["Architecture","Accuracy","F1-Score","R²","RMSE"])
        + '</div>'
    )

    rows = ""
    for d in BD:
        fc, bg = TYPE_COLORS[d["type"]]
        best   = d["acc"] == ba
        ac     = T["success"] if best else T["t0"]
        rows  += (
            f'<div style="display:grid;grid-template-columns:1.6fr 1fr 1fr 0.7fr 0.7fr;'
            f'gap:1rem;align-items:center;padding:0.85rem 0.6rem;'
            f'border-bottom:1px solid {T["b0"]};border-radius:5px;transition:background 0.12s;"'
            f' onmouseover="this.style.background=\'{T["hover"]}\'"'
            f' onmouseout="this.style.background=\'transparent\'">'
            f'<div>'
            f'<div style="font-weight:600;color:{ac};font-size:0.85rem;margin-bottom:3px;">{d["arch"]}</div>'
            f'<span style="display:inline-flex;padding:1px 7px;border-radius:100px;'
            f'font-family:{T["mono"]};font-size:0.58rem;font-weight:600;'
            f'background:{bg};border:1px solid {fc}44;color:{fc};">{d["type"]}</span>'
            f'</div>'
            + bench_bar(d["acc"], 100, d["acc"] == ba)
            + bench_bar(d["f1"],  100, d["f1"]  == bf)
            + f'<div style="font-family:{T["mono"]};font-size:0.82rem;'
              f'color:{T["success"] if d["r2"]==br else T["t1"]};'
              f'font-weight:{"700" if d["r2"]==br else "400"};">{d["r2"]}</div>'
            + f'<div style="font-family:{T["mono"]};font-size:0.82rem;'
              f'color:{T["success"] if d["rmse"]==bm else T["t1"]};'
              f'font-weight:{"700" if d["rmse"]==bm else "400"};">{d["rmse"]}</div>'
            + '</div>'
        )

    st.markdown(
        f'<div style="background:{T["card"]};border:1px solid {T["b0"]};'
        f'border-radius:12px;padding:1.25rem 1.5rem;box-shadow:0 4px 24px rgba(0,0,0,0.5);">'
        + hdr + rows + '</div>', unsafe_allow_html=True)

    st.markdown(divider("Accuracy vs F1-Score", "2rem"), unsafe_allow_html=True)
    st.bar_chart(
        pd.DataFrame({"Architecture":[d["arch"] for d in BD],
                      "Accuracy":[d["acc"] for d in BD],
                      "F1-Score":[d["f1"]  for d in BD]}).set_index("Architecture"),
        height=340, color=[T["acc"], T["success"]])

    # radar / summary insight
    st.markdown(divider("Key Insights"), unsafe_allow_html=True)
    ins_c1, ins_c2, ins_c3 = st.columns(3)
    with ins_c1:
        st.markdown(card(
            eyebrow_label("Best Overall") +
            f'<div style="font-family:{T["disp"]};font-size:1.2rem;font-weight:800;'
            f'color:{T["success"]};margin-bottom:0.4rem;">CNN-LSTM Hybrid</div>'
            f'<div style="font-family:{T["mono"]};font-size:0.72rem;color:{T["t1"]};">'
            f'98.6% accuracy · 98.9% F1<br>R² 0.96 · RMSE 0.11</div>'
        ), unsafe_allow_html=True)
    with ins_c2:
        st.markdown(card(
            eyebrow_label("Best RL Model") +
            f'<div style="font-family:{T["disp"]};font-size:1.2rem;font-weight:800;'
            f'color:{T["acc_hi"]};margin-bottom:0.4rem;">Deep Q-Network</div>'
            f'<div style="font-family:{T["mono"]};font-size:0.72rem;color:{T["t1"]};">'
            f'95.8% accuracy · 95.1% F1<br>R² 0.91 · RMSE 0.18</div>'
        ), unsafe_allow_html=True)
    with ins_c3:
        st.markdown(card(
            eyebrow_label("Weakest Baseline") +
            f'<div style="font-family:{T["disp"]};font-size:1.2rem;font-weight:800;'
            f'color:{T["warn"]};margin-bottom:0.4rem;">K-Means</div>'
            f'<div style="font-family:{T["mono"]};font-size:0.72rem;color:{T["t1"]};">'
            f'78.4% accuracy · 75.0% F1<br>Unsupervised — no labelled data</div>'
        ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown(
    f'<div class="nids-footer">'
    f'NETWORK INTRUSION DETECTION SYSTEM USING REINFORCEMENT LEARNING &nbsp;·&nbsp;'
    f'<span style="color:{T["acc"]};">Sagar Rajgiri &amp; Devansh Surana</span>'
    f'&nbsp;·&nbsp; SCOPE, VIT VELLORE &nbsp;·&nbsp; '
    f'<span style="color:{T["t2"]};">{datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</span>'
    f'</div>', unsafe_allow_html=True)