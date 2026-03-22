import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="IndoStockBro",
    page_icon="🇮🇩",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS (sama seperti sebelumnya, dipotong agar tidak terlalu panjang)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0a0a12;}
.block-container{padding:1.5rem 1rem 4rem;max-width:720px;}
[data-testid="stSidebar"]{display:none;}
.kartu-beli{background:#0d1f0f;border:1px solid #1a3a1e;border-left:4px solid #00d4aa;border-radius:12px;padding:18px 20px;margin-bottom:14px;}
.kartu-tunggu{background:#141410;border:1px solid #2a2a18;border-left:4px solid #e3b341;border-radius:12px;padding:18px 20px;margin-bottom:14px;}
.saham-kode{color:#e6edf3;font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace;}
.saham-nama{color:#484f58;font-size:11px;margin-top:2px;}
.badge-beli{background:#1a3a1e;color:#00d4aa;font-size:11px;font-weight:600;padding:4px 12px;border-radius:20px;border:1px solid #00d4aa44;}
.badge-tunggu{background:#2a2a18;color:#e3b341;font-size:11px;font-weight:600;padding:4px 12px;border-radius:20px;border:1px solid #e3b34144;}
.divider{border:none;border-top:1px solid #21262d;margin:12px 0;}
.row-data{display:flex;justify-content:space-between;padding:4px 0;font-size:13px;}
.row-label{color:#484f58;}
.row-val{color:#e6edf3;font-weight:500;font-family:'JetBrains Mono',monospace;}
.row-green{color:#00d4aa;font-weight:600;font-family:'JetBrains Mono',monospace;}
.row-red{color:#ff6b6b;font-weight:600;font-family:'JetBrains Mono',monospace;}
.conf-bar-bg{background:#21262d;border-radius:4px;height:6px;margin:4px 0 8px;}
.ai-box{background:#0d1117;border-left:2px solid #4da6ff;border-radius:0 8px 8px 0;padding:8px 12px;margin-top:10px;font-size:13px;color:#8b949e;line-height:1.6;}
.stButton>button{background:linear-gradient(135deg,#00d4aa,#0099cc)!important;color:#060a0f!important;font-weight:700!important;font-size:14px!important;border:none!important;border-radius:10px!important;padding:12px 24px!important;width:100%!important;}
.stSelectbox>div>div{background:#161b22!important;border:1px solid #30363d!important;border-radius:8px!important;color:#e6edf3!important;}
.stTextInput>div>div>input{background:#161b22!important;border:1px solid #30363d!important;border-radius:8px!important;color:#e6edf3!important;font-family:'JetBrains Mono',monospace!important;font-size:15px!important;}
.stProgress>div>div>div>div{background:linear-gradient(90deg,#00d4aa,#4da6ff)!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid #21262d!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#484f58!important;font-size:13px!important;}
.stTabs [aria-selected="true"]{color:#00d4aa!important;border-bottom:2px solid #00d4aa!important;}
[data-testid="stMetricValue"]{color:#00d4aa!important;font-family:'JetBrains Mono',monospace!important;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Groq AI (untuk summary)
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")

def ai_summary(prompt):
    if not GROQ_KEY:
        return ""
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 120,
                "temperature": 0.4
            },
            timeout=20
        )
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        return ""

# ──────────────────────────────────────────────────────────────────────────────
# Fungsi ambil data yfinance dengan cache
@st.cache_data(ttl=3600, show_spinner=False)
def ambil_data(kode):
    try:
        t = yf.Ticker(f"{kode}.JK")
        df = t.history(period="1y")
        info = t.info
        if df is None or df.empty or len(df) < 30:  # kurangi menjadi 30 hari
            return None, None
        return df, info
    except Exception:
        return None, None

# ──────────────────────────────────────────────────────────────────────────────
# Core engine: analisis lengkap
def analisis_lengkap(kode):
    try:
        df, info = ambil_data(kode)
        if df is None or df.empty or len(df) < 30:
            return None
    except Exception:
        return None

    try:
        c = df['Close']
        h = df['High']
        l = df['Low']
        v = df['Volume']

        harga = round(c.iloc[-1], 0)
        perubahan = round(((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2]) * 100, 2)
        vol_ratio = round(v.iloc[-1] / (v.tail(20).mean() + 1), 2)

        # RSI Wilder
        d = c.diff()
        gain = d.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
        loss = (-d.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
        rsi = round(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-10)), 1)

        ma20 = round(c.tail(20).mean(), 0)
        ma50 = round(c.tail(50).mean(), 0) if len(df) >= 50 else None
        supp = round(l.tail(20).min(), 0)
        res = round(h.tail(20).max(), 0)

        # ATR
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = round(tr.rolling(14).mean().iloc[-1], 0)

        ma5 = c.tail(5).mean()
        trend = "UPTREND ↑" if ma5 > ma20 else "DOWNTREND ↓" if ma5 < ma20 else "SIDEWAYS →"

        # Behavioral pattern
        hi_hari = df.iloc[-1]
        rng_hi = hi_hari['High'] - hi_hari['Low']
        cr_today = (hi_hari['Close'] - hi_hari['Low']) / (rng_hi + 1) if rng_hi > 0 else 0.5

        cond_mirip = 0
        ol_besok = 0
        naik_list = []
        turun_list = []

        for i in range(1, len(df) - 1):
            hari = df.iloc[i]
            besok = df.iloc[i+1]
            rng_h = hari['High'] - hari['Low']
            if rng_h == 0:
                continue
            cr = (hari['Close'] - hari['Low']) / rng_h
            if abs(cr - cr_today) < 0.25:
                cond_mirip += 1
                rng_b = besok['High'] - besok['Low']
                if rng_b > 0 and abs(besok['Open'] - besok['Low']) / rng_b <= 0.05:
                    ol_besok += 1
                if besok['Open'] > 0:
                    k = ((besok['Close'] - besok['Open']) / besok['Open']) * 100
                    naik_list.append(k) if k > 0 else turun_list.append(abs(k))

        if cond_mirip == 0:
            return None

        prob = round((ol_besok / cond_mirip) * 100, 1)
        avg_naik = round(np.mean(naik_list), 2) if naik_list else 0
        max_naik = round(np.max(naik_list), 2) if naik_list else 0
        win_rate = round(len(naik_list) / (len(naik_list) + len(turun_list) + 0.001) * 100, 1)

        # Hari terbaik
        df2 = df.copy()
        df2['dow'] = df2.index.dayofweek
        df2['ol'] = False
        for i in range(len(df2) - 1):
            rb = df2.iloc[i+1]['High'] - df2.iloc[i+1]['Low']
            if rb > 0 and abs(df2.iloc[i+1]['Open'] - df2.iloc[i+1]['Low']) / rb <= 0.05:
                df2.iloc[i, df2.columns.get_loc('ol')] = True
        bd = df2.groupby('dow')['ol'].mean()
        hn = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat'}
        hari_terbaik = hn.get(bd.idxmax() if not bd.empty else 0, 'N/A')

        # Bandarmologi
        clv = ((c - l) - (h - c)) / (h - l + 1e-10)
        cmf = (clv * v).tail(20).sum() / (v.tail(20).sum() + 1e-10)
        obv = (np.sign(c.diff()) * v).cumsum()
        obv_up = obv.iloc[-1] > obv.iloc[-10]

        if cmf > 0.05 and obv_up:
            bandar, b_ikon = "AKUMULASI", "🟢"
        elif cmf < -0.05 and not obv_up:
            bandar, b_ikon = "DISTRIBUSI", "🔴"
        elif cmf > 0 and obv_up:
            bandar, b_ikon = "AKUMULASI LEMAH", "🟡"
        else:
            bandar, b_ikon = "NETRAL", "🟡"

        # Confidence Score
        s_prob = min(prob * 2.5, 40)
        s_winrate = min(win_rate * 0.25, 20)
        s_bandar = 25 if 'AKUMULASI' in bandar else 10 if bandar == 'NETRAL' else 0
        s_volume = min(vol_ratio * 5, 15)
        confidence = round(s_prob + s_winrate + s_bandar + s_volume, 1)

        # Target & SL
        targets = [t for t in [round(harga + atr, 0), res, ma50 if ma50 and harga < ma50 else ma20] if harga < t < harga * 1.25]
        if not targets:
            targets = [round(harga + atr, 0)]
        sl = round(harga - atr * 1.5, 0)
        sl_pct = round(((sl - harga) / harga) * 100, 2)
        t_med = round(float(np.median(targets)), 0)
        t_pct = round(((t_med - harga) / harga) * 100, 2)
        rr = abs(round(t_pct / (abs(sl_pct) + 0.001), 1))

        lulus = (cond_mirip >= 5 and prob >= 10 and rsi < 75 and harga >= 50 and 'DISTRIBUSI' not in bandar)

        nama = info.get('longName', kode) if info else kode

        return dict(
            kode=kode, nama=nama, harga=harga, perubahan=perubahan,
            vol_ratio=vol_ratio, rsi=rsi, trend=trend,
            ma20=ma20, ma50=ma50, supp=supp, res=res,
            cond_mirip=cond_mirip, ol_besok=ol_besok,
            prob=prob, avg_naik=avg_naik, max_naik=max_naik,
            win_rate=win_rate, hari_terbaik=hari_terbaik,
            bandar=bandar, b_ikon=b_ikon, cmf=round(cmf, 3),
            confidence=confidence, lulus=lulus,
            target=t_med, t_pct=t_pct, sl=sl, sl_pct=sl_pct, rr=rr,
        )
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Daftar saham IDX (sudah dibersihkan)
SAHAM_IDX = [
    "AALI","ACES","ADHI","ADRO","AGII","AKRA","AMRT","ANTM","ARNA","ASII",
    "ASRI","AUTO","BBCA","BBNI","BBRI","BBTN","BFIN","BJBR","BJTM","BKSL",
    "BMRI","BMTR","BNII","BREN","BRIS","BRMS","BSDE","BYAN","CASS","CPIN",
    "CTRA","DMAS","DOID","DSNG","EKAD","ELSA","EMTK","ERAA","ESSA","EXCL",
    "FILM","GGRM","GOOD","GOTO","HEAL","HRUM","ICBP","INCO","INDF","INDY",
    "INKP","INTP","ISAT","ITMG","JPFA","JSMR","KAEF","KLBF","LSIP","MAPI",
    "MARK","MBMA","MDKA","MEDC","MIKA","MNCN","MTEL","MTDL","MYOR","NISP",
    "PGAS","PGEO","PNBN","PTBA","PTPP","PTRO","PWON","RAJA","RALS","ROTI",
    "SIDO","SIMP","SMGR","SMRA","SMSM","SRTG","SSMS","TINS","TLKM","TOWR",
    "TPIA","TRIM","TSPC","ULTJ","UNTR","UNVR","WIKA","WSKT","ADCP","ADMF",
    "AGRO","AKPI","AKSI","ALDO","ALMI","AMFG","ANJT","APEX","APII","APLN",
    "ASDM","ASGR","ASMI","ATPK","AYLS","BACA","BAJA","BATA","BAYU","BBHI",
    "BBKP","BBMD","BBSI","BCIP","BEEF","BELL","BHIT","BIKA","BKDP","BLTA",
    "BNBA","BNLI","BOLT","BOSS","BPII","BRAM","BRNA","BRPT","BSSR","BTON",
    "BTPN","BUDI","BUVA","BVIC","CAKK","CARE","CARS","CASH","CEKA","CFIN",
    "CINT","CITA","CLAY","CLEO","COAL","COCO","CPRI","CSAP","DART","DEAL",
    "DEWA","DIGI","DILD","DKFT","DNAR","DNET","DPNS","DRMA","DSFI","DSSA",
    "DUCK","DUTI","DVLA","EMDE","ENRG","EPMT","ESTA","FAST","FISH","FORU",
    "GAMA","GDST","GDYR","GEMS","GJTL","GPRA","HADE","HERO","HEXA","HITS",
    "HMSP","HOKI","HRTA","IBST","ICON","IGAR","IKBI","IMJS","IMPC","INAF",
    "INAI","INCF","IPCC","IPCM","IPOL","IRRA","ISSP","JAWA","JECC","JIHD",
    "JKON","JRPT","JSPT","JTPE","KEEN","KIJA","KINO","KKGI","KOIN","KPIG",
    "KRAS","LCGP","LEAD","LINK","LION","LMSH","LPCK","LPGI","LPIN","LPKR",
    "LTLS","MAIN","MAYA","MBAP","MDLN","MEGA","MERK","META","MICE","MIRA",
    "MITI","MKPI","MLBI","MLIA","MLPL","MPPA","MRAT","MREI","MSKY","MTFN",
    "MTLA","MYOH","NELY","NIKL","NIRO","NOBU","NRCA","OKAS","OMRE","PADI",
    "PANR","PANS","PBRX","PJAA","PLAN","PLIN","PNLF","PSAB","PTIS","PTSP",
    "PUDP","PYFA","RANC","RBMS","RDTX","RELI","RICY","RIGS","RODA","RUIS",
    "SAFE","SAME","SCMA","SDRA","SGRO","SHID","SHIP","SIPD","SKBM","SKLT",
    "SMBR","SMCB","SMDR","SMMA","SMMT","SMRU","SOCI","SONA","SPMA","SQMI",
    "SRIL","SRSN","SSIA","STAR","STTP","SUGI","SUPR","TALF","TARA","TBIG",
    "TBLA","TCID","TINS","TKIM","TMAS","TOBA","TOTO","TPMA","TRIO","TRIS",
    "TRST","UNIC","UNIT","UNSP","VICO","VIVA","VOKS","WEGE","WIIM","WINS",
    "WOOD","WTON","BTPS","PZZA","TUGU","MAPA","NFCX","NUSA","MOLI","CITY",
    "YELO","LUCK","URBN","FOOD","CLAY","ITIC","BLUE","EAST","FUJI","TFAS",
    "OPMS","PURE","IRRA","SINI","TEBE","KEJU","PSGO","REAL","UCID","GLVA",
    "AMAR","AMOR","PURA","TAMA","SAMF","KBAG","RONY","CSMI","UANG","SOHO",
    "HOMI","ROCK","PLAN","ATAP","BANK","EDGE","TAPG","HOPE","LABA","ARCI",
    "MASB","BUKA","OILS","MCOL","RUNS","CMNT","IDEA","BOBA","MTEL","DEPO",
    "CMRY","AVIA","NASI","DRMA","ADMR","NETV","ENAK","STAA","NANO","BIKE",
    "TLDN","WINR","TRGU","CHEM","DEWI","GULA","AMMS","RAFI","EURO","KLIN",
    "BUAH","MEDS","COAL","PRAY","BELI","NINE","SOUL","ELIT","BEER","SUNI",
    "WINE","PEVE","LAJU","PACK","VAST","CHIP","HALO","KING","PGEO","FUTR",
    "HILL","SAGE","CUAN","JATI","SMIL","KLAS","MAXI","VKTR","AMMN","CRSN",
    "WIDI","INET","MAHA","CNMA","FOLK","GRIA","ERAL","MUTU","RSCH","BABY",
    "BREN","STRK","LOPI","RGAS","AYAM","SURI","ASLI","SMGA","UNTD","ALII",
    "MEJA","LIVE","HYGN","BAIK","VISI","AREA","ATLA","DATA","SOLA","BATR",
    "PART","GOLF","GUNA","LABS","DOSS","NEST","VERN","BOAT","NAIK","AADI",
    "MDIY","RATU","YOII","BRRC","MINE","ASPR","PSAT","MERI","CHEK","EMAS",
    "RLCO","SUPA","YUPI","FORE","MDLA","AYLS","DADA","ASPI","ESTA","BESS",
    "AMAN","CARE","PIPA","NCKL","MENN","AWAN","MBMA","RAAM","CGAS","NICE",
    "SMLE","ACRO","MANG","WIFI","FAPA","DCII","DGNS","ADMG","AGRS","PNSE",
    "POLY","POOL","PPRO"
]
SAHAM_IDX = sorted(set(SAHAM_IDX))

# ──────────────────────────────────────────────────────────────────────────────
# Render kartu
def render_kartu(rank, r):
    beli = 'AKUMULASI' in r['bandar'] and r['prob'] >= 10 and r['confidence'] >= 40
    cls = "kartu-beli" if beli else "kartu-tunggu"
    badge = '<span class="badge-beli">✅ BELI BESOK</span>' if beli else '<span class="badge-tunggu">⏳ PANTAU</span>'
    p_col = "row-green" if r['perubahan'] >= 0 else "row-red"
    p_sgn = "+" if r['perubahan'] >= 0 else ""
    cw = min(int(r['confidence']), 100)
    cc = "#00d4aa" if cw >= 60 else "#e3b341" if cw >= 40 else "#ff6b6b"

    st.markdown(f"""
    <div class="{cls}">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div>
          <div class="saham-kode">#{rank} &nbsp;{r['kode']}</div>
          <div class="saham-nama">{r['nama'][:50]}</div>
        </div>
        {badge}
      </div>
      <div class="divider"></div>
      <div class="row-data"><span class="row-label">💰 Harga</span>
        <span class="row-val">Rp {r['harga']:,.0f} &nbsp;<span class="{p_col}">({p_sgn}{r['perubahan']}%)</span></span></div>
      <div class="row-data"><span class="row-label">📊 Prob Open=Low besok</span>
        <span class="row-green">{r['prob']}% &nbsp;<span style="color:#484f58;font-size:11px;">({r['ol_besok']}/{r['cond_mirip']} historis)</span></span></div>
      <div class="row-data"><span class="row-label">🏆 Confidence</span>
        <span class="row-val">{r['confidence']}/100</span></div>
      <div class="conf-bar-bg"><div style="width:{cw}%;background:{cc};height:6px;border-radius:4px;"></div></div>
      <div class="row-data"><span class="row-label">🏦 Bandar</span>
        <span class="row-val">{r['b_ikon']} {r['bandar']} &nbsp;<span style="color:#484f58;font-size:11px;">CMF {r['cmf']}</span></span></div>
      <div class="divider"></div>
      <div class="row-data"><span class="row-label">📥 Entry</span>
        <span class="row-val">Rp {r['harga']:,.0f} &nbsp;<span style="color:#484f58;font-size:11px;">(beli saat open)</span></span></div>
      <div class="row-data"><span class="row-label">🎯 Target</span>
        <span class="row-green">Rp {r['target']:,.0f} &nbsp;(+{r['t_pct']}%)</span></div>
      <div class="row-data"><span class="row-label">🚀 Maks historis</span>
        <span class="row-green">+{r['max_naik']}% &nbsp;<span style="color:#484f58;font-size:11px;">win rate {r['win_rate']}%</span></span></div>
      <div class="row-data"><span class="row-label">🛑 Stop Loss</span>
        <span class="row-red">Rp {r['sl']:,.0f} &nbsp;({r['sl_pct']}%)</span></div>
      <div class="row-data"><span class="row-label">⚖️ R/R &nbsp;·&nbsp; RSI &nbsp;·&nbsp; Trend</span>
        <span class="row-val" style="font-size:12px;">1:{r['rr']} &nbsp;·&nbsp; {r['rsi']} &nbsp;·&nbsp; {r['trend']}</span></div>
      <div class="row-data"><span class="row-label">📅 Hari terbaik</span>
        <span class="row-val">{r['hari_terbaik']} &nbsp;<span style="color:#484f58;font-size:11px;">Vol {r['vol_ratio']}x</span></span></div>
    </div>
    """, unsafe_allow_html=True)

    if GROQ_KEY:
        with st.spinner(""):
            ai = ai_summary(
                f"Analis saham IDX. 2 kalimat SINGKAT untuk trader scalping besok.\n"
                f"{r['kode']} Rp {r['harga']:,.0f} | RSI {r['rsi']} | {r['trend']} | "
                f"Bandar: {r['bandar']} CMF:{r['cmf']} | "
                f"Prob OL: {r['prob']}% ({r['ol_besok']}/{r['cond_mirip']} historis) | "
                f"Target +{r['t_pct']}% | Max +{r['max_naik']}% | Win rate {r['win_rate']}%\n"
                f"Indonesia. 2 kalimat. Langsung ke point. Tidak perlu sapaan."
            )
        if ai:
            st.markdown(f'<div class="ai-box">💬 {ai}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Main UI
def main():
    st.markdown(f"""
    <div style="text-align:center;padding:24px 0 8px;">
        <div style="font-size:32px;margin-bottom:4px;">🇮🇩</div>
        <h1 style="color:#e6edf3;font-size:24px;font-weight:700;margin:0;">IndoStockBro</h1>
        <p style="color:#484f58;font-size:12px;margin:4px 0 0;font-family:'JetBrains Mono',monospace;">
            AI · Open=Low Scanner · IDX · {datetime.now().strftime('%d %b %Y %H:%M WIB')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    tab1, tab2 = st.tabs(["🏆 Scan Open=Low Besok", "🔍 Cek Saham"])

    with tab1:
        st.markdown("""
        <div style="background:#161b22;border-radius:10px;padding:12px 16px;margin-bottom:16px;border:1px solid #21262d;">
            <p style="color:#8b949e;font-size:13px;margin:0;line-height:1.6;">
                💡 <b style="color:#e6edf3;">Cara pakai:</b>
                Jalankan sore <b style="color:#00d4aa;">15:30–17:00</b> setelah market tutup.
                Sistem belajar kebiasaan historis 1 tahun tiap saham untuk prediksi besok.
            </p>
        </div>
        """, unsafe_allow_html=True)

        jumlah = st.select_slider("Jumlah saham di-scan", options=[50, 100, 150, 200, 300], value=150)
        scan_btn = st.button("🚀 Mulai Scan", type="primary")

        if scan_btn:
            saham_scan = random.sample(SAHAM_IDX, min(jumlah, len(SAHAM_IDX)))
            total = len(saham_scan)
            prog = st.progress(0)
            info_txt = st.empty()
            kandidat = []
            diproses = 0

            for i, kode in enumerate(saham_scan):
                prog.progress((i + 1) / total)
                info_txt.markdown(f'<p style="color:#484f58;font-size:12px;font-family:monospace;">⏳ [{i+1}/{total}] Menganalisis {kode}...</p>', unsafe_allow_html=True)
                r = analisis_lengkap(kode)
                time.sleep(0.05)
                if r is None:
                    continue
                diproses += 1
                if r['lulus']:
                    kandidat.append(r)

            prog.empty()
            info_txt.empty()
            kandidat.sort(key=lambda x: x['confidence'], reverse=True)
            top3 = kandidat[:3]

            st.markdown(f"""
            <div style="background:#0d1117;border-radius:10px;padding:12px 16px;
                        margin:8px 0 20px;border:1px solid #21262d;
                        display:flex;justify-content:space-between;">
                <div><span style="color:#484f58;font-size:11px;font-family:'JetBrains Mono',monospace;">DIANALISIS</span><br>
                     <span style="color:#e6edf3;font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace;">{diproses}</span></div>
                <div style="text-align:center;"><span style="color:#484f58;font-size:11px;font-family:'JetBrains Mono',monospace;">KANDIDAT</span><br>
                     <span style="color:#00d4aa;font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace;">{len(kandidat)}</span></div>
                <div style="text-align:right;"><span style="color:#484f58;font-size:11px;font-family:'JetBrains Mono',monospace;">DITAMPILKAN</span><br>
                     <span style="color:#e6edf3;font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace;">TOP {len(top3)}</span></div>
            </div>
            """, unsafe_allow_html=True)

            if not top3:
                st.warning("⚠️ Tidak ada kandidat hari ini. Pasar mungkin sedang konsolidasi. Coba scan lebih banyak saham.")
            else:
                st.markdown("### 🏆 Rekomendasi Open=Low Besok")
                st.caption(f"📅 {datetime.now().strftime('%d %B %Y, %H:%M WIB')} · Urut: Confidence Score")
                st.write("")
                for i, r in enumerate(top3, 1):
                    render_kartu(i, r)

            st.markdown("""
            <div style="text-align:center;padding:16px;color:#484f58;font-size:11px;
                        font-family:'JetBrains Mono',monospace;">
                ⚠️ Bukan saran investasi · Selalu gunakan stop loss · DYOR
            </div>""", unsafe_allow_html=True)

    with tab2:
        if 'kode_cek' not in st.session_state:
            st.session_state['kode_cek'] = ''

        kode_input = st.text_input(
            "", placeholder="Ketik kode saham, contoh: BBCA",
            label_visibility="collapsed",
            value=st.session_state['kode_cek']
        ).upper().strip().replace(".JK", "")

        st.caption("Populer:")
        cols = st.columns(5)
        populer = ['BBCA', 'BBRI', 'BMRI', 'TLKM', 'ASII', 'GOTO', 'ADRO', 'ANTM', 'PTBA', 'INDF']
        for i, s in enumerate(populer):
            if cols[i % 5].button(s, key=f"p_{s}", use_container_width=True):
                st.session_state['kode_cek'] = s
                st.rerun()

        cek_btn = st.button("🔍 Analisis", type="primary", use_container_width=True)

        if (kode_input or st.session_state['kode_cek']) and cek_btn:
            kode_final = kode_input or st.session_state['kode_cek']
            with st.spinner(f"🧠 Menganalisis kebiasaan {kode_final}..."):
                r = analisis_lengkap(kode_final)
            if r is None:
                st.error(f"❌ {kode_final} tidak ditemukan atau data tidak cukup.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Harga", f"Rp {r['harga']:,.0f}", f"{r['perubahan']:+.1f}%")
                c2.metric("Prob OL", f"{r['prob']}%")
                c3.metric("Confidence", f"{r['confidence']}/100")
                st.write("")
                render_kartu(1, r)

if __name__ == "__main__":
    main()
