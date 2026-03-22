import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import warnings
import random
warnings.filterwarnings('ignore')

st.set_page_config(page_title="IndoStockBro", page_icon="🇮🇩", layout="centered")

st.markdown("""
<style>
.stApp{background:#0a0a12;}
.block-container{padding:1.5rem 1rem 4rem;max-width:720px;}
[data-testid="stSidebar"]{display:none;}
.kartu{border-radius:12px;padding:18px 20px;margin-bottom:14px;}
.row-data{display:flex;justify-content:space-between;padding:4px 0;font-size:13px;}
.row-label{color:#484f58;}
.row-val{color:#e6edf3;font-weight:500;font-family:monospace;}
.row-green{color:#00d4aa;font-weight:600;font-family:monospace;}
.row-red{color:#ff6b6b;font-weight:600;font-family:monospace;}
.divider{border:none;border-top:1px solid #21262d;margin:12px 0;}
.ai-box{background:#0d1117;border-left:2px solid #4da6ff;border-radius:0 8px 8px 0;padding:8px 12px;margin-top:10px;font-size:13px;color:#8b949e;line-height:1.6;}
.stButton>button{background:linear-gradient(135deg,#00d4aa,#0099cc)!important;color:#060a0f!important;font-weight:700!important;border:none!important;border-radius:10px!important;width:100%!important;}
.stProgress>div>div>div>div{background:linear-gradient(90deg,#00d4aa,#4da6ff)!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid #21262d!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#484f58!important;font-size:13px!important;}
.stTabs [aria-selected="true"]{color:#00d4aa!important;border-bottom:2px solid #00d4aa!important;}
[data-testid="stMetricValue"]{color:#00d4aa!important;font-family:monospace!important;}
</style>
""", unsafe_allow_html=True)

# ── Groq ──
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")

def ai_call(prompt):
    if not GROQ_KEY: return ""
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization":f"Bearer {GROQ_KEY}","Content-Type":"application/json"},
            json={"model":"llama-3.3-70b-versatile",
                  "messages":[{"role":"user","content":prompt}],
                  "max_tokens":120,"temperature":0.4},
            timeout=20)
        return r.json()["choices"][0]["message"]["content"]
    except: return ""

# ── Core Engine ──
def scan(kode):
    """Analisis lengkap satu saham"""
    try:
        t    = yf.Ticker(f"{kode}.JK")
        df   = t.history(period="1y")
        info = t.info

        if df is None or df.empty or len(df) < 60:
            return None

        # Reset timezone
        df.index = df.index.tz_localize(None)

        c=df['Close']; h=df['High']; l=df['Low']; v=df['Volume']

        # Harga
        harga     = round(float(c.iloc[-1]), 0)
        perubahan = round(((float(c.iloc[-1])-float(c.iloc[-2]))/float(c.iloc[-2]))*100, 2)
        vol_ratio = round(float(v.iloc[-1])/(float(v.tail(20).mean())+1), 2)

        # RSI Wilder
        d    = c.diff()
        gain = d.clip(lower=0).ewm(alpha=1/14,min_periods=14).mean()
        loss = (-d.clip(upper=0)).ewm(alpha=1/14,min_periods=14).mean()
        rsi  = round(100-100/(1+float(gain.iloc[-1])/(float(loss.iloc[-1])+1e-10)), 1)

        # MA
        ma20 = round(float(c.tail(20).mean()), 0)
        ma50 = round(float(c.tail(50).mean()), 0) if len(df)>=50 else None
        supp = round(float(l.tail(20).min()), 0)
        res  = round(float(h.tail(20).max()), 0)

        # ATR
        tr  = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atr = round(float(tr.rolling(14).mean().iloc[-1]), 0)

        # Trend
        ma5   = float(c.tail(5).mean())
        trend = "UPTREND ↑" if ma5>ma20 else "DOWNTREND ↓" if ma5<ma20 else "SIDEWAYS →"

        # ── Behavioral Pattern ──
        hi_row   = df.iloc[-1]
        rng_hi   = float(hi_row['High']) - float(hi_row['Low'])
        cr_today = (float(hi_row['Close'])-float(hi_row['Low']))/(rng_hi+1) if rng_hi>0 else 0.5

        cond_mirip=ol_besok=0
        naik_list=[]; turun_list=[]

        for i in range(1, len(df)-1):
            hari=df.iloc[i]; besok=df.iloc[i+1]
            rng_h=float(hari['High'])-float(hari['Low'])
            if rng_h==0: continue
            cr=(float(hari['Close'])-float(hari['Low']))/rng_h
            if abs(cr-cr_today)<0.25:
                cond_mirip+=1
                rng_b=float(besok['High'])-float(besok['Low'])
                if rng_b>0 and abs(float(besok['Open'])-float(besok['Low']))/rng_b<=0.05:
                    ol_besok+=1
                if float(besok['Open'])>0:
                    k=((float(besok['Close'])-float(besok['Open']))/float(besok['Open']))*100
                    naik_list.append(k) if k>0 else turun_list.append(abs(k))

        if cond_mirip==0: return None
        prob     = round((ol_besok/cond_mirip)*100, 1)
        avg_naik = round(float(np.mean(naik_list)), 2) if naik_list else 0
        max_naik = round(float(np.max(naik_list)), 2) if naik_list else 0
        win_rate = round(len(naik_list)/(len(naik_list)+len(turun_list)+0.001)*100, 1)

        # Hari terbaik
        df2=df.copy(); df2['dow']=df2.index.dayofweek; df2['ol']=False
        for i in range(len(df2)-1):
            rb=float(df2.iloc[i+1]['High'])-float(df2.iloc[i+1]['Low'])
            if rb>0 and abs(float(df2.iloc[i+1]['Open'])-float(df2.iloc[i+1]['Low']))/rb<=0.05:
                df2.iloc[i, df2.columns.get_loc('ol')]=True
        bd=df2.groupby('dow')['ol'].mean()
        hn={0:'Senin',1:'Selasa',2:'Rabu',3:'Kamis',4:'Jumat'}
        hari_terbaik=hn.get(int(bd.idxmax()) if not bd.empty else 0,'N/A')

        # ── Bandarmologi ──
        clv=((c-l)-(h-c))/(h-l+1e-10)
        cmf=float((clv*v).tail(20).sum()/(v.tail(20).sum()+1e-10))
        obv=(np.sign(c.diff())*v).cumsum()
        obv_up=float(obv.iloc[-1])>float(obv.iloc[-10])

        if cmf>0.05 and obv_up:        bandar,b_ikon="AKUMULASI","🟢"
        elif cmf<-0.05 and not obv_up:  bandar,b_ikon="DISTRIBUSI","🔴"
        elif cmf>0 and obv_up:          bandar,b_ikon="AKUMULASI LEMAH","🟡"
        else:                            bandar,b_ikon="NETRAL","🟡"

        # ── Confidence Score ──
        s1=min(prob*2.5,40); s2=min(win_rate*0.25,20)
        s3=25 if 'AKUMULASI' in bandar else 10 if bandar=='NETRAL' else 0
        s4=min(vol_ratio*5,15)
        confidence=round(s1+s2+s3+s4, 1)

        # Target & SL
        tlist=[x for x in [round(harga+atr,0), res,
                             ma50 if ma50 and harga<ma50 else ma20]
               if harga<x<harga*1.25]
        if not tlist: tlist=[round(harga+atr,0)]
        sl     = round(harga-atr*1.5, 0)
        sl_pct = round(((sl-harga)/harga)*100, 2)
        t_med  = round(float(np.median(tlist)), 0)
        t_pct  = round(((t_med-harga)/harga)*100, 2)
        rr     = abs(round(t_pct/(abs(sl_pct)+0.001), 1))

        lulus=(cond_mirip>=5 and prob>=10 and
               rsi<75 and harga>=50 and 'DISTRIBUSI' not in bandar)

        nama=info.get('longName',kode) if info else kode

        return dict(
            kode=kode, nama=nama, harga=harga, perubahan=perubahan,
            vol_ratio=vol_ratio, rsi=rsi, trend=trend,
            cond_mirip=cond_mirip, ol_besok=ol_besok,
            prob=prob, avg_naik=avg_naik, max_naik=max_naik,
            win_rate=win_rate, hari_terbaik=hari_terbaik,
            bandar=bandar, b_ikon=b_ikon, cmf=round(cmf,3),
            confidence=confidence, lulus=lulus,
            target=t_med, t_pct=t_pct, sl=sl, sl_pct=sl_pct, rr=rr,
        )
    except:
        return None

# ── Daftar saham ──
SAHAM = list(set([
    "AALI","ACES","ADHI","ADRO","AGII","AKRA","AMRT","ANTM","ASII","ASRI",
    "AUTO","BBCA","BBNI","BBRI","BBTN","BFIN","BJBR","BKSL","BMRI","BMTR",
    "BNII","BREN","BRIS","BRMS","BSDE","BYAN","CASS","CPIN","CTRA","DMAS",
    "DOID","DSNG","EKAD","ELSA","EMTK","ERAA","ESSA","EXCL","GGRM","GOOD",
    "GOTO","HEAL","HRUM","ICBP","INCO","INDF","INDY","INKP","INTP","ISAT",
    "ITMG","JPFA","JSMR","KAEF","KLBF","LSIP","MAPI","MARK","MBMA","MDKA",
    "MEDC","MIKA","MNCN","MTEL","MYOR","NISP","PGAS","PGEO","PNBN","PTBA",
    "PTPP","PTRO","PWON","RALS","ROTI","SIDO","SIMP","SMGR","SMRA","SMSM",
    "SRTG","SSMS","TINS","TLKM","TOWR","TPIA","TRIM","TSPC","ULTJ","UNTR",
    "UNVR","WIKA","WSKT","AKPI","ALMI","AMFG","ANJT","ASDM","ASGR","ATPK",
    "BACA","BAJA","BATA","BAYU","BBHI","BBKP","BBMD","BCIP","BEEF","BELL",
    "BHIT","BIKA","BOLT","BOSS","BPII","BRAM","BRNA","BRPT","BSSR","BTON",
    "BTPN","BUDI","CEKA","CFIN","CINT","CITA","CLAY","CLEO","COAL","COCO",
    "CPRI","CSAP","DART","DEWA","DILD","DKFT","DNAR","DPNS","DRMA","DSFI",
    "DSSA","DUCK","DUTI","DVLA","EKAD","EMDE","ENRG","EPMT","ERAA","FAST",
    "FISH","GDST","GDYR","GEMS","GJTL","GPRA","HERO","HEXA","HITS","HMSP",
    "HOKI","HRTA","IBST","ICON","IGAR","IKBI","IMPC","INAF","INAI","IPCC",
    "IPOL","IRRA","ISSP","JAWA","JECC","JKON","JRPT","JSPT","KEEN","KIJA",
    "KINO","KLBF","KOIN","KPIG","KRAS","LEAD","LINK","LION","LMSH","LPCK",
    "LPIN","LPKR","LTLS","MAIN","MAYA","MBAP","MDLN","MEGA","MERK","MICE",
    "MIRA","MITI","MKPI","MLBI","MLIA","MLPL","MPPA","MRAT","MREI","MTFN",
    "MTLA","MYOH","NELY","NIKL","NIRO","NOBU","NRCA","OKAS","PADI","PANR",
    "PANS","PJAA","PLAN","PLIN","PNLF","PSAB","PTIS","PTSP","PUDP","PYFA",
    "RANC","RDTX","RELI","RICY","RIGS","RODA","RUIS","SAFE","SAME","SCMA",
    "SDRA","SGRO","SHID","SHIP","SIPD","SKBM","SKLT","SMBR","SMCB","SMDR",
    "SMMA","SMRU","SOCI","SONA","SPMA","SQMI","SRIL","SRSN","SSIA","STAR",
    "STTP","SUGI","SUPR","TBIG","TBLA","TCID","TINS","TKIM","TMAS","TOBA",
    "TOTO","TPMA","TRIO","TRIS","TRST","UNIC","UNIT","VICO","VIVA","VOKS",
    "WEGE","WIIM","WINS","WOOD","WTON","BTPS","PZZA","TUGU","NFCX","CITY",
    "YELO","LUCK","URBN","FOOD","ITIC","BLUE","EAST","TFAS","OPMS","PURE",
    "SINI","TEBE","KEJU","REAL","UCID","AMAR","AMOR","PURA","TAMA","SAMF",
    "SOHO","HOMI","ROCK","ATAP","BANK","TAPG","LABA","ARCI","BUKA","OILS",
    "CMNT","IDEA","BOBA","DEPO","CMRY","AVIA","NASI","ADMR","ENAK","STAA",
    "NANO","GOTO","WINR","TRGU","CHEM","GULA","EURO","BUAH","MEDS","COAL",
    "PRAY","NINE","SOUL","BEER","SUNI","PACK","VAST","CHIP","HALO","KING",
    "PGEO","HILL","SAGE","CUAN","SMIL","KLAS","VKTR","AMMN","CRSN","INET",
    "MAHA","CNMA","GRIA","ERAL","MUTU","RSCH","BREN","LOPI","RGAS","AYAM",
    "ASLI","SMGA","ALII","MEJA","LIVE","HYGN","BAIK","AREA","ATLA","DATA",
    "SOLA","PART","GOLF","GUNA","LABS","DOSS","NEST","VERN","BOAT","NAIK",
    "AADI","RATU","BRRC","MINE","ASPR","MERI","CHEK","EMAS","SUPA","YUPI",
    "FORE","MDLA","DADA","ASPI","ESTA","BESS","AMAN","CARE","PIPA","NCKL",
    "MENN","AWAN","MBMA","RAAM","CGAS","NICE","SMLE","ACRO","MANG","WIFI",
    "FAPA","DCII","DGNS","ADMG","AGRS","POLY","POOL","PPRO","FILM","PBSA",
]))

# ── Render kartu ──
def kartu(rank, r):
    beli  = 'AKUMULASI' in r['bandar'] and r['prob']>=10 and r['confidence']>=40
    warna = "#00d4aa" if beli else "#e3b341"
    badge = "✅ BELI BESOK" if beli else "⏳ PANTAU"
    bg    = "#0d1f0f" if beli else "#141410"
    bdr   = "#1a3a1e" if beli else "#2a2a18"
    p_col = "#00d4aa" if r['perubahan']>=0 else "#ff6b6b"
    cw    = min(int(r['confidence']),100)
    cc    = "#00d4aa" if cw>=60 else "#e3b341" if cw>=40 else "#ff6b6b"
    p_sgn = "+" if r['perubahan']>=0 else ""

    st.markdown(f"""
    <div style="background:{bg};border:1px solid {bdr};border-left:4px solid {warna};
                border-radius:12px;padding:18px 20px;margin-bottom:14px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div>
          <div style="color:#e6edf3;font-size:20px;font-weight:700;font-family:monospace;">
            #{rank} &nbsp;{r['kode']}
          </div>
          <div style="color:#484f58;font-size:11px;margin-top:2px;">{r['nama'][:50]}</div>
        </div>
        <span style="background:{'#1a3a1e' if beli else '#2a2a18'};
                     color:{warna};font-size:11px;font-weight:600;
                     padding:4px 12px;border-radius:20px;">{badge}</span>
      </div>
      <div style="border-top:1px solid #21262d;margin:12px 0;"></div>
      <div class="row-data" style="display:flex;justify-content:space-between;padding:4px 0;font-size:13px;">
        <span style="color:#484f58;">💰 Harga</span>
        <span style="color:#e6edf3;font-family:monospace;">Rp {r['harga']:,.0f}
          <span style="color:{p_col};">({p_sgn}{r['perubahan']}%)</span></span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;font-size:13px;">
        <span style="color:#484f58;">📊 Prob Open=Low besok</span>
        <span style="color:#00d4aa;font-weight:600;font-family:monospace;">{r['prob']}%
          <span style="color:#484f58;font-size:11px;">({r['ol_besok']}/{r['cond_mirip']} historis)</span></span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;font-size:13px;">
        <span style="color:#484f58;">🏆 Confidence</span>
        <span style="color:#e6edf3;font-family:monospace;">{r['confidence']}/100</span>
      </div>
      <div style="background:#21262d;border-radius:4px;height:6px;margin:4px 0 8px;">
        <div style="width:{cw}%;background:{cc};height:6px;border-radius:4px;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;font-size:13px;">
        <span style="color:#484f58;">🏦 Bandar</span>
        <span style="color:#e6edf3;font-family:monospace;">{r['b_ikon']} {r['bandar']}
          <span style="color:#484f58;font-size:11px;">CMF {r['cmf']}</span></span>
      </div>
      <div style="border-top:1px solid #21262d;margin:12px 0;"></div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;font-size:13px;">
        <span style="color:#484f58;">📥 Entry</span>
        <span style="color:#e6edf3;font-family:monospace;">Rp {r['harga']:,.0f}
          <span style="color:#484f58;font-size:11px;">(beli saat open)</span></span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;font-size:13px;">
        <span style="color:#484f58;">🎯 Target</span>
        <span style="color:#00d4aa;font-weight:600;font-family:monospace;">
          Rp {r['target']:,.0f} (+{r['t_pct']}%)</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;font-size:13px;">
        <span style="color:#484f58;">🚀 Maks historis</span>
        <span style="color:#00d4aa;font-family:monospace;">+{r['max_naik']}%
          <span style="color:#484f58;font-size:11px;">win rate {r['win_rate']}%</span></span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;font-size:13px;">
        <span style="color:#484f58;">🛑 Stop Loss</span>
        <span style="color:#ff6b6b;font-weight:600;font-family:monospace;">
          Rp {r['sl']:,.0f} ({r['sl_pct']}%)</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;font-size:13px;">
        <span style="color:#484f58;">⚖️ R/R · RSI · Trend</span>
        <span style="color:#e6edf3;font-size:12px;font-family:monospace;">
          1:{r['rr']} · {r['rsi']} · {r['trend']}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;font-size:13px;">
        <span style="color:#484f58;">📅 Hari terbaik</span>
        <span style="color:#e6edf3;font-size:12px;">{r['hari_terbaik']}
          <span style="color:#484f58;font-size:11px;">Vol {r['vol_ratio']}x</span></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if GROQ_KEY:
        with st.spinner(""):
            ai = ai_call(
                f"Analis saham IDX. 2 kalimat SINGKAT untuk trader scalping besok.\n"
                f"{r['kode']} Rp {r['harga']:,.0f} RSI:{r['rsi']} {r['trend']} "
                f"Bandar:{r['bandar']} CMF:{r['cmf']} "
                f"Prob OL:{r['prob']}% ({r['ol_besok']}/{r['cond_mirip']} historis) "
                f"Target:+{r['t_pct']}% Max:+{r['max_naik']}% WinRate:{r['win_rate']}%\n"
                f"Bahasa Indonesia. 2 kalimat. Langsung ke point."
            )
        if ai:
            st.markdown(f'<div class="ai-box">💬 {ai}</div>', unsafe_allow_html=True)

# ── Main ──
def main():
    st.markdown(f"""
    <div style="text-align:center;padding:24px 0 8px;">
        <div style="font-size:32px;">🇮🇩</div>
        <h1 style="color:#e6edf3;font-size:24px;font-weight:700;margin:4px 0;">IndoStockBro</h1>
        <p style="color:#484f58;font-size:12px;margin:0;font-family:monospace;">
            AI · Open=Low Scanner · IDX · {datetime.now().strftime('%d %b %Y %H:%M WIB')}
        </p>
    </div>
    <hr style="border-color:#21262d;">
    """, unsafe_allow_html=True)

    menu = st.radio(
        "",
        ["🏆 Scan Open=Low Besok", "🔍 Cek Saham"],
        horizontal=True,
        label_visibility="collapsed"
    )
    st.divider()

    if "Scan" in menu:
        st.markdown("""
        <div style="background:#161b22;border-radius:10px;padding:12px 16px;
                    margin-bottom:16px;border:1px solid #21262d;">
            <p style="color:#8b949e;font-size:13px;margin:0;line-height:1.6;">
                💡 <b style="color:#e6edf3;">Cara pakai:</b>
                Jalankan sore <b style="color:#00d4aa;">15:30–17:00</b> setelah market tutup.
                Sistem belajar kebiasaan historis 1 tahun tiap saham untuk prediksi open=low besok.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<p style='color:#8b949e;font-size:13px;margin:0 0 6px 0;'>Jumlah saham di-scan</p>", unsafe_allow_html=True)
        jumlah   = st.select_slider("jumlah_scan", label_visibility="collapsed",
                                     options=[50,100,150,200,300], value=150)
        scan_btn = st.button("🚀 Mulai Scan", type="primary", key="scan")

        if scan_btn:
            saham_scan = random.sample(SAHAM, min(jumlah, len(SAHAM)))
            total      = len(saham_scan)
            prog       = st.progress(0)
            info_txt   = st.empty()
            kandidat   = []
            diproses   = 0

            for i, kode in enumerate(saham_scan):
                prog.progress((i+1)/total)
                info_txt.caption(f"⏳ [{i+1}/{total}] Menganalisis {kode}...")
                r = scan(kode)
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
                        display:flex;justify-content:space-between;align-items:center;">
                <div><span style="color:#484f58;font-size:11px;font-family:monospace;">DIANALISIS</span><br>
                     <span style="color:#e6edf3;font-size:22px;font-weight:700;font-family:monospace;">{diproses}</span></div>
                <div style="text-align:center;">
                     <span style="color:#484f58;font-size:11px;font-family:monospace;">KANDIDAT</span><br>
                     <span style="color:#00d4aa;font-size:22px;font-weight:700;font-family:monospace;">{len(kandidat)}</span></div>
                <div style="text-align:right;">
                     <span style="color:#484f58;font-size:11px;font-family:monospace;">TOP</span><br>
                     <span style="color:#e6edf3;font-size:22px;font-weight:700;font-family:monospace;">{len(top3)}</span></div>
            </div>
            """, unsafe_allow_html=True)

            if not top3:
                st.warning("⚠️ Tidak ada kandidat hari ini. Coba scan lebih banyak saham atau coba besok.")
            else:
                st.markdown(f"### 🏆 Open=Low Besok — {datetime.now().strftime('%d %b %Y')}")
                st.caption("Urut: Confidence Score tertinggi")
                st.write("")
                for i, r in enumerate(top3, 1):
                    kartu(i, r)

            st.markdown("""
            <div style="text-align:center;padding:16px;color:#484f58;font-size:11px;font-family:monospace;">
                ⚠️ Bukan saran investasi · Selalu gunakan stop loss · DYOR
            </div>""", unsafe_allow_html=True)

    else:
        kode_input = st.text_input(
            "Kode saham",
            placeholder="Contoh: BBCA, ADRO, TLKM",
            label_visibility="collapsed"
        ).upper().strip().replace(".JK", "")

        if st.button("🔍 Analisis", type="primary", key="cek") and kode_input:
            with st.spinner(f"🧠 Menganalisis {kode_input}..."):
                r = scan(kode_input)
            if r is None:
                st.error(f"❌ {kode_input} tidak ditemukan atau data tidak cukup.")
            else:
                c1,c2,c3 = st.columns(3)
                c1.metric("Harga", f"Rp {r['harga']:,.0f}", f"{r['perubahan']:+.1f}%")
                c2.metric("Prob OL", f"{r['prob']}%")
                c3.metric("Confidence", f"{r['confidence']}/100")
                st.write("")
                kartu(1, r)

if __name__ == "__main__":
    main()
