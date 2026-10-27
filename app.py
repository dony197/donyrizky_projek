"""
Identifikasi Gambar - Ini Kacamata??
Aplikasi Streamlit: deteksi kacamata / tidak kacamata dari foto (unggah / kamera)
Mendukung model Keras (.h5) dan PyTorch (.pt, termasuk YOLO hasil deteksi objek).
"""
import base64
import io
import os

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

# ============================================================
# KONFIGURASI  -> sesuaikan di bagian ini saja
# ============================================================
MODEL_H5_PATH = "model.h5"          # path file model Keras
MODEL_PT_PATH = "best.pt"          # path file model PyTorch / YOLO
CLASS_NAMES = ["kacamata", "tidak kacamata"]   # urutan label = urutan output model
DEFAULT_IMG_SIZE = 224              # ukuran input (dipakai untuk .pt non-YOLO; .h5 dibaca otomatis)
HOME_URL = "https://halaman-beranda.vercel.app/"   # link halaman index.html Anda
CREDIT = "Projek ini dibuat oleh Dony Rizky Hidayat 2308108010002"
IMG_BACK = "karakter-belakang.png"  # opsional (sama seperti di index.html)
IMG_FRONT = "karakter-depan.png"    # opsional
IMG_MID = "karakter-tengah.png"    # opsional (gambar tengah, di antara belakang dan depan)

LABEL_INFO = {
    "kacamata": ("Kacamata", "👓", "Orang pada foto terdeteksi memakai kacamata."),
    "tidak kacamata": ("Tidak kacamata", "🙂", "Orang pada foto terdeteksi tidak memakai kacamata."),
}
NORM_OPTIONS = [
    "0 – 1 (dibagi 255)",
    "Tanpa normalisasi (0 – 255)",
    "-1 – 1 (MobileNet / Xception)",
    "ImageNet mean/std (umum di PyTorch)",
]

st.set_page_config(page_title="Ini Kacamata?", page_icon="👓", layout="wide",
                   initial_sidebar_state="collapsed")


# ============================================================
# TAMPILAN (meniru index.html: tema lukisan langit, coklat)
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Chau+Philomene+One&family=Kaushan+Script&display=swap');
:root{ --ink:#4a3a22; --dark:#2a2012; --card:rgba(255,255,255,.38); --script:'Kaushan Script',cursive;
  --start:1.6s; /* waktu loading selesai; semua animasi pembuka mengikuti angka ini */ }

html, body, [class*="css"], .stApp, .stMarkdown, p, label, span, button, input{
  font-family:'Chau Philomene One', system-ui, sans-serif !important; color:var(--ink);
}
[data-testid="stExpander"] summary [data-testid="stIconMaterial"]{
  font-family:'Material Symbols Rounded','Material Icons' !important;
}
.stApp{
  background:
    radial-gradient(ellipse 40% 22% at 78% 14%, rgba(222,150,80,.65), transparent 70%),
    radial-gradient(ellipse 30% 18% at 14% 20%, rgba(190,140,95,.45), transparent 70%),
    radial-gradient(ellipse 35% 20% at 30% 62%, rgba(214,160,100,.45), transparent 70%),
    radial-gradient(ellipse 30% 18% at 70% 75%, rgba(225,175,110,.45), transparent 70%),
    linear-gradient(180deg,#e2c28f 0%,#ecd0a0 45%,#e3bf88 100%);
  background-attachment:fixed;
}
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer{display:none !important}
.block-container{max-width:1250px; padding:0 4vw 110px !important}

/* ---------- loading ---------- */
#splash{position:fixed; inset:0; z-index:99999; background:#ecd0a0; display:flex; flex-direction:column;
  align-items:center; justify-content:center; gap:22px;
  animation:splashOut .9s cubic-bezier(.77,0,.18,1) var(--start) forwards}
@keyframes splashOut{to{transform:translateY(-100%); visibility:hidden}}
#splash svg{width:150px; overflow:visible}
#splash path,#splash circle{fill:none; stroke:#4a3a22; stroke-width:5; stroke-linecap:round; stroke-linejoin:round;
  stroke-dasharray:400; stroke-dashoffset:400; animation:draw 1s ease forwards}
#splash .r{animation-delay:.1s} #splash .b{animation-delay:.55s}
@keyframes draw{to{stroke-dashoffset:0}}
#splash .lt{font-size:18px; letter-spacing:.3px}
#splash .lt span{display:inline-block; animation:dot 1.2s infinite}
#splash .lt span:nth-child(2){animation-delay:.2s}
#splash .lt span:nth-child(3){animation-delay:.4s}
@keyframes dot{0%,100%{opacity:.2} 50%{opacity:1}}
@property --n{syntax:'<integer>'; initial-value:0; inherits:false}
@keyframes count{to{--n:100}}
#splash .pct{font-size:14px; color:#6b5636; counter-reset:n var(--n); animation:count calc(var(--start) - .2s) ease forwards}
#splash .pct::after{content:counter(n) "%"}
#splash .bar{width:220px; height:6px; background:rgba(74,58,34,.18); border-radius:99px; overflow:hidden}
#splash .bar i{display:block; height:100%; width:0; background:#4a3a22; border-radius:99px;
  animation:fill calc(var(--start) - .2s) ease forwards}
@keyframes fill{to{width:100%}}

/* ---------- nav ---------- */
.nav{display:flex; justify-content:flex-end; align-items:center; gap:56px; padding:44px 0 0;
  font-size:24px; letter-spacing:.5px}
.nav a{color:#111 !important; text-decoration:none !important; text-transform:uppercase; position:relative}
.nav a::after{content:""; position:absolute; left:0; bottom:-5px; height:2px; width:0; background:var(--ink); transition:width .25s}
.nav a:hover::after{width:100%}

/* ---------- hero (CSS grid, bukan st.columns) ---------- */
.hero-grid{position:relative; display:grid; grid-template-columns:1.35fr 1fr; align-items:start}
.hero-wrap{position:relative; min-height:62vh; padding-top:3vh; z-index:3}
.hero h1{font-weight:400; text-transform:uppercase; line-height:1; margin:0; padding:.08em 0;
  font-size:clamp(34px,5.2vw,86px); color:var(--ink)}
.hero .sub{font-family:var(--script) !important; font-size:clamp(44px,6.6vw,110px); line-height:1.1;
  transform:rotate(-3deg); transform-origin:left; margin:.1em 0 0 2vw; color:var(--ink)}
.hero .desc{font-size:clamp(17px,1.7vw,28px); line-height:1.4; margin-top:4vh; color:#3b2e1a; max-width:760px}
.hero .line{display:block; padding:.08em 0}
.hero .line > span{display:block; transform:translateY(110%);
  animation:rise .9s cubic-bezier(.2,.8,.2,1) var(--start) forwards}
.hero .sub.line > span{animation-delay:calc(var(--start) + .3s)}
.hero .sub span{font-family:var(--script) !important}   /* aturan global 'span' tadi menimpa font script */
@keyframes rise{to{transform:translateY(0)}}
.hero .desc{opacity:0; animation:fade .8s ease calc(var(--start) + .6s) forwards}
@keyframes fade{to{opacity:1}}

/* ---------- karakter ---------- */
[data-testid="stMain"], .stApp{overflow-x:hidden}
.art{position:relative; z-index:1; pointer-events:none;
  width:100%; height:max(340px, calc(100vh - 140px)); margin:0;
  opacity:0; animation:artIn 1s cubic-bezier(.2,.8,.2,1) calc(var(--start) + .4s) forwards}
@keyframes artIn{from{opacity:0; transform:translateX(60px)} to{opacity:1; transform:none}}
.art img{position:absolute; width:auto !important; max-width:none !important; max-height:none !important;
  object-fit:contain; object-position:bottom right; border:0 !important; border-radius:0 !important; box-shadow:none !important}
.art .back{right:-25%; top:-8%; height:80% !important; opacity:.55; z-index:1; will-change:transform;
  transform:translate(calc(var(--sy,0)*.28px), calc(var(--sy,0)*.18px)) rotate(calc(var(--sy,0)*.012deg))}
.art .mid{right:10%; top:-10%; height:110% !important; opacity:.5; z-index:2; will-change:transform;
  /* scaleX(-1) di paling kanan: gambar dibalik, arah geraknya tetap sama dengan index.html */
  transform:translate(calc(var(--sy,0)*-.22px), calc(var(--sy,0)*-.15px)) rotate(calc(var(--sy,0)*-.015deg))
            scale(calc(1 + var(--sy,0)*.0003)) scaleX(-1)}
.art .front{right:-15%; bottom:0; height:82% !important; z-index:3; will-change:transform;
  transform:translate(calc(var(--sy,0)*-.10px), calc(var(--sy,0)*-.12px)) scale(calc(1 + var(--sy,0)*.0004))}
.art svg{position:absolute; right:0; bottom:10%; width:100%; max-width:480px}

/* layar kecil (HP): teks di tengah, tiga karakter di BAWAH teks (sama seperti index.html) */
@media (max-width:820px){
  .hero-grid{grid-template-columns:1fr}
  .hero-wrap{min-height:0; text-align:center; padding-top:2vh}
  .hero h1{text-align:center; white-space:normal}
  .hero .desc{margin-left:auto; margin-right:auto}
  .art{width:90%; height:auto; aspect-ratio:4/5; margin:20px auto 0}
  .art .back{height:90% !important; right:-15%; top:-20%}
  .art .front{height:84% !important; right:15%; top:-5%; bottom:auto}
  .art .mid{height:800px !important; width:80% !important; right:45%; top:-120%}
}

/* ---------- section ---------- */
.sec-title{font-size:clamp(30px,4vw,48px); line-height:1.15; margin:70px 0 8px}
.lead{font-size:18px; max-width:640px; line-height:1.5}
.steps{display:grid; grid-template-columns:repeat(3,1fr); gap:22px; margin-top:28px}
.step{border-left:4px solid var(--ink); padding:4px 0 4px 18px}
.step b{display:block; font-size:24px; margin-bottom:2px}
.step h3{font-size:20px; margin:0 0 6px; font-weight:400}
.step p{line-height:1.5; font-size:16px; margin:0}
@media (max-width:820px){
  .steps{grid-template-columns:1fr}
  .nav{justify-content:center; gap:26px; font-size:18px; padding-top:24px}
  .hero .sub{margin-left:0; transform:none}
}

/* ---------- widget Streamlit ---------- */
.stTabs [data-baseweb="tab-list"]{gap:10px; border-bottom:none !important; box-shadow:none !important}
.stTabs [data-baseweb="tab-border"]{display:none !important}
.stTabs [data-baseweb="tab"] p{font-size:20px !important}
.stTabs [data-baseweb="tab-highlight"]{background:var(--ink) !important; height:3px}
[data-testid="stFileUploaderDropzone"]{background:var(--card); border:none !important; outline:none !important;
  border-radius:18px; padding:28px}
[data-testid="stFileUploader"] section{border:none !important; outline:none !important}
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span{color:var(--ink)}
.stButton>button, [data-testid="stFileUploaderDropzone"] button, [data-testid="stCameraInputButton"],
[data-testid="stBaseButton-secondary"]{
  background:var(--ink) !important; color:#fff !important; border:0 !important; border-radius:99px !important;
  padding:.55rem 1.6rem !important; box-shadow:0 6px 0 var(--dark); transition:transform .15s, box-shadow .15s}
.stButton>button:hover, [data-testid="stFileUploaderDropzone"] button:hover, [data-testid="stCameraInputButton"]:hover{
  transform:translateY(3px); box-shadow:0 3px 0 var(--dark)}
.stButton>button p, [data-testid="stFileUploaderDropzone"] button p, [data-testid="stCameraInputButton"] p{color:#fff !important}
[data-testid="stExpander"]{background:var(--card); border:none !important; border-radius:16px}
[data-testid="stExpander"] summary p{font-size:18px}
[data-testid="stImage"] img{border-radius:18px; border:3px solid var(--ink)}
div[data-baseweb="select"]>div, .stTextInput input, .stNumberInput input{background:rgba(255,255,255,.55) !important}
.stRadio label p{font-size:17px}

/* ---------- hasil ---------- */
.result{background:var(--card); border:3px solid var(--ink); border-radius:22px; padding:26px 28px}
.result.ok{background:#f3dfb7}
.r-cap{font-size:16px; opacity:.8}
.r-main{font-size:clamp(34px,4.4vw,60px); line-height:1.1; margin:6px 0 4px}
.r-main.script{font-family:var(--script) !important}
.r-desc{font-size:17px; margin-bottom:16px}
.pbar{margin:10px 0}
.pbar .row{display:flex; justify-content:space-between; font-size:16px; margin-bottom:4px}
.pbar .track{height:10px; background:rgba(74,58,34,.18); border-radius:99px; overflow:hidden}
.pbar .track i{display:block; height:100%; background:var(--ink); border-radius:99px}
.pbar.dim .track i{background:rgba(74,58,34,.45)}
.placeholder{border:none !important; border-radius:22px; padding:44px 20px; text-align:center; font-size:18px; background:var(--card)}


/* ---------- scroll halus (html { scroll-behavior:smooth } di index.html) ---------- */
html, [data-testid="stMain"]{scroll-behavior:smooth}
/* iframe pembawa script tidak perlu memakan ruang */
.element-container:has(iframe[height="0"]), [data-testid="stElementContainer"]:has(iframe[height="0"]){
  position:absolute; height:0; overflow:hidden}

/* ---------- ticker ---------- */
.ticker{position:fixed; left:0; right:0; bottom:0; z-index:999; overflow:hidden; white-space:nowrap;
  background:rgba(236,208,160,.94); padding:12px 0; border-top:2px solid var(--ink)}
.ticker .track{display:inline-block; animation:flow 22s linear infinite; font-size:18px}
.ticker .track span{padding-right:70px}
.ticker:hover .track{animation-play-state:paused}
@keyframes flow{from{transform:translateX(0)} to{transform:translateX(-50%)}}
@media (prefers-reduced-motion:reduce){*{animation-duration:.01s !important; animation-delay:0s !important}
  .ticker .track{animation:none}
  .art .back,.art .front{transform:none !important}
  .art .mid{transform:scaleX(-1) !important}}
</style>
"""

SPLASH = """
<div id="splash" aria-label="Memuat halaman">
  <svg viewBox="0 0 220 90" aria-hidden="true">
    <circle class="r" cx="55" cy="50" r="34"/><circle class="r" cx="165" cy="50" r="34"/>
    <path class="b" d="M89 46 Q110 28 131 46"/><path class="b" d="M21 46 L4 30"/><path class="b" d="M199 46 L216 30"/>
  </svg>
  <div class="lt">Memuat<span>.</span><span>.</span><span>.</span></div>
  <div class="bar"><i></i></div>
  <div class="pct"></div>
</div>
"""

FALLBACK_SVG = """
<svg viewBox="0 0 220 90" fill="none" stroke="#4a3a22" stroke-width="5" stroke-linecap="round" aria-hidden="true">
  <circle cx="55" cy="50" r="34"/><circle cx="165" cy="50" r="34"/>
  <path d="M89 46 Q110 28 131 46"/><path d="M21 46 L4 30"/><path d="M199 46 L216 30"/>
</svg>
"""

ANIM_JS = """
<script>
(function () {
  var P = window.parent, D = P.document;
  if (P.__kcParallax2) return;                   // cukup dipasang sekali per sesi browser
  P.__kcParallax2 = true;
  if (P.matchMedia && P.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  // sama seperti index.html: nilai skrol dihaluskan (lerp 0.12) lalu dikirim ke variabel CSS --sy
  var target = 0, cur = 0, running = false;
  function frame() {
    cur += (target - cur) * 0.12;
    if (Math.abs(target - cur) < 0.05) cur = target;
    D.querySelectorAll('.art').forEach(function (a) { a.style.setProperty('--sy', cur.toFixed(2)); });
    if (cur !== target) P.requestAnimationFrame(frame); else running = false;
  }
  function onScroll(v) {
    target = Math.min(v, 900);
    if (!running) { running = true; P.requestAnimationFrame(frame); }
  }
  // kalau Streamlit membuat ulang elemen .art (misalnya setelah upload), pasang lagi nilai --sy terakhir
  new P.MutationObserver(function () {
    var v = cur.toFixed(2);
    D.querySelectorAll('.art').forEach(function (a) {
      if (a.style.getPropertyValue('--sy') !== v) a.style.setProperty('--sy', v);
    });
  }).observe(D.body, { childList: true, subtree: true });

  // di Streamlit yang di-skrol adalah kontainer stMain, bukan window
  D.addEventListener('scroll', function (e) {
    var t = e.target;
    if (t === D || t === D.documentElement) { onScroll(P.scrollY || 0); return; }
    if (t.getAttribute && t.getAttribute('data-testid') === 'stMain') { onScroll(t.scrollTop); return; }
    if (t.classList && t.classList.contains('main')) { onScroll(t.scrollTop); }
  }, true);
})();
</script>
"""


@st.cache_data(show_spinner=False)
def _img_data(path, mtime):
    """Kecilkan gambar dan ubah ke WebP (tetap transparan) agar HTML yang dikirim ke browser ringan.
    mtime hanya dipakai sebagai kunci cache: kalau file diganti, cache otomatis diperbarui."""
    try:
        im = Image.open(path).convert("RGBA")
        im.thumbnail((1400, 1400))
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=85, method=4)
        return "image/webp", base64.b64encode(buf.getvalue()).decode()
    except Exception:                               # Pillow tanpa WebP -> pakai file asli
        with open(path, "rb") as f:
            return "image/png", base64.b64encode(f.read()).decode()


def img_tag(path, cls, alt=""):
    """Sisipkan gambar lokal sebagai base64 supaya tampil di HTML Streamlit."""
    if not os.path.exists(path):
        return ""
    mime, b64 = _img_data(path, os.path.getmtime(path))
    return f'<img class="{cls}" src="data:{mime};base64,{b64}" alt="{alt}">'


def show_image(img):
    """Tampilkan gambar selebar kolom. Streamlit baru memakai width="stretch",
    Streamlit lama masih memakai use_container_width=True."""
    try:
        st.image(img, width="stretch")
    except Exception:
        st.image(img, use_container_width=True)


def render_header():
    st.markdown(CSS, unsafe_allow_html=True)
    # Splash SELALU dirender (isinya tetap sama), supaya urutan elemen tidak bergeser saat rerun
    # dan animasi hero/karakter tidak ter-reset setelah upload foto. Animasinya hanya jalan saat halaman dibuka.
    st.markdown(SPLASH, unsafe_allow_html=True)

    st.markdown(
        f'<div class="nav"><a href="{HOME_URL}" target="_blank">Beranda</a>'
        f'<a href="#coba">Coba</a><a href="#cara">Cara kerja</a></div>',
        unsafe_allow_html=True,
    )

    back, mid = img_tag(IMG_BACK, "back"), img_tag(IMG_MID, "mid")
    front = img_tag(IMG_FRONT, "front", "Ilustrasi karakter berkacamata")
    fallback = "" if (back or mid or front) else FALLBACK_SVG
    art = f'<div class="art">{back}{mid}{front}{fallback}</div>'

    # Satu blok HTML tanpa baris kosong / indentasi, supaya tidak dibaca sebagai code block markdown
    hero = (
        '<div class="hero-grid">'
        '<div class="hero-wrap hero">'
        '<h1><span class="line"><span>identifikasi gambar</span></span></h1>'
        '<div class="sub line"><span>ini kacamata ??</span></div>'
        '<p class="desc">Unggah gambar atau ambil foto lewat kamera, lalu biarkan kecerdasan buatan '
        'mengidentifikasi apakah seseorang menggunakan kacamata atau tidak. Hasil muncul otomatis, '
        'cepat, dan praktis.</p>'
        '</div>'
        f'{art}'
        '</div>'
    )
    st.markdown(hero, unsafe_allow_html=True)


def render_steps():
    st.markdown(
        """<div id="cara"></div><h2 class="sec-title">Cara kerjanya</h2>
        <p class="lead">Tiga langkah singkat dari gambar sampai hasil.</p>
        <div class="steps">
          <div class="step"><b>1</b><h3>Unggah atau foto</h3><p>Pilih foto wajah dari galeri, atau ambil foto langsung dengan kamera.</p></div>
          <div class="step"><b>2</b><h3>Model menganalisis</h3><p>Model mencari ciri kacamata pada gambar.</p></div>
          <div class="step"><b>3</b><h3>Lihat hasilnya</h3><p>Aplikasi menampilkan apakah orang pada foto memakai kacamata, lengkap dengan tingkat keyakinan.</p></div>
        </div>""",
        unsafe_allow_html=True,
    )



def render_ticker():
    spans = "".join(f"<span>{CREDIT}</span>" for _ in range(4))
    st.markdown(f'<div class="ticker"><div class="track">{spans}</div></div>', unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================
@st.cache_resource(show_spinner="Memuat model...")
def load_keras(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource(show_spinner="Memuat model...")
def load_torch(path):
    """Mengembalikan (model, jenis). Jenis: 'yolo' atau 'torch'."""
    import torch
    # 1) YOLO (deteksi objek / klasifikasi) - hanya jika paket ultralytics terpasang
    try:
        from ultralytics import YOLO
        return YOLO(path), "yolo"
    except Exception:
        pass
    # 2) TorchScript (torch.jit.save)
    try:
        m = torch.jit.load(path, map_location="cpu")
        m.eval()
        return m, "torch"
    except Exception:
        pass
    # 3) Model utuh (torch.save(model, ...))
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        raise RuntimeError(
            "File .pt ini hanya berisi state_dict (bobot), bukan model utuh. "
            "Tambahkan definisi arsitektur di fungsi load_torch(), contoh:\n"
            "model = NamaArsitektur(); model.load_state_dict(obj); model.eval()"
        )
    obj.eval()
    return obj, "torch"


def preprocess(img: Image.Image, size, mode_idx):
    h, w = size
    arr = np.asarray(img.convert("RGB").resize((w, h)), dtype=np.float32)
    if mode_idx == 0:
        arr /= 255.0
    elif mode_idx == 2:
        arr = arr / 127.5 - 1.0
    elif mode_idx == 3:
        arr = (arr / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
              np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return arr  # H, W, 3


def to_probs(out):
    """Ubah output model (sigmoid 1 neuron ATAU softmax 2 neuron) menjadi [p_kelas0, p_kelas1]."""
    v = np.asarray(out, dtype=np.float64).reshape(-1)
    if v.size == 1:                         # sigmoid: nilai = peluang kelas indeks 1
        p = v[0] if 0.0 <= v[0] <= 1.0 else 1.0 / (1.0 + np.exp(-v[0]))
        return np.array([1.0 - p, p])
    if v.min() < 0 or v.max() > 1 or not np.isclose(v.sum(), 1.0, atol=1e-3):
        e = np.exp(v - v.max())
        v = e / e.sum()                     # logits -> softmax
    return v[:2]


def yolo_detect(model, img):
    """
    Cara baca sama seperti dashboard.py: model(np.array(img)) -> boxes + plot().
    Mengembalikan (probs, gambar_berkotak). probs = None bila tidak ada objek terdeteksi.
    Jika model YOLO adalah model klasifikasi (r.probs ada), dipakai langsung.
    """
    r = model(np.array(img.convert("RGB")), verbose=False)[0]
    annotated = r.plot()[:, :, ::-1]           # BGR -> RGB agar warna benar di Streamlit

    # model klasifikasi YOLO (*-cls.pt)
    if getattr(r, "probs", None) is not None:
        return to_probs(r.probs.data.cpu().numpy()), None

    # model deteksi objek
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return None, annotated                 # tidak ada objek terdeteksi

    name_to_idx = {n.lower(): i for i, n in enumerate(CLASS_NAMES)}
    p = np.zeros(len(CLASS_NAMES))
    for c, s in zip(boxes.cls.cpu().numpy().astype(int),
                    boxes.conf.cpu().numpy()):
        i = name_to_idx.get(str(r.names[c]).lower(), int(c))   # cocokkan lewat nama kelas
        if i < len(p):
            p[i] = max(p[i], float(s))         # confidence tertinggi per kelas
    return (p if p.sum() > 0 else None), annotated


def predict(img, framework, path, norm_idx, torch_size):
    """Mengembalikan (probs, gambar_berkotak_atau_None)."""
    if framework == "Keras (.h5)":
        model = load_keras(path)
        shp = model.input_shape
        shp = shp[0] if isinstance(shp, list) else shp
        h = shp[1] or DEFAULT_IMG_SIZE
        w = shp[2] or DEFAULT_IMG_SIZE
        x = preprocess(img, (h, w), norm_idx)[None, ...]
        return to_probs(model.predict(x, verbose=0)), None

    model, kind = load_torch(path)
    if kind == "yolo":
        return yolo_detect(model, img)

    import torch
    x = preprocess(img, (torch_size, torch_size), norm_idx).transpose(2, 0, 1)[None, ...]
    with torch.no_grad():
        out = model(torch.from_numpy(np.ascontiguousarray(x)))
    if isinstance(out, (list, tuple)):
        out = out[0]
    return to_probs(out.detach().cpu().numpy()), None


# ============================================================
# HASIL
# ============================================================
def pbar(name, p, dim=False):
    return (f'<div class="pbar {"dim" if dim else ""}"><div class="row"><span>{name}</span>'
            f'<span>{p*100:.1f}%</span></div><div class="track"><i style="width:{p*100:.1f}%"></i></div></div>')


def show_result(probs, names):
    top = int(np.argmax(probs))
    label, emoji, desc = LABEL_INFO.get(names[top], (names[top].title(), "", ""))
    bars = "".join(pbar(LABEL_INFO.get(n, (n,))[0], float(p), dim=(i != top))
                   for i, (n, p) in enumerate(zip(names, probs)))
    st.markdown(
        f"""<div class="result ok"><div class="r-cap">Hasil identifikasi</div>
        <div class="r-main script">{label} {emoji}</div>
        <div class="r-desc">{desc} Keyakinan model {probs[top]*100:.1f}%.</div>{bars}</div>""",
        unsafe_allow_html=True,
    )


def analyze(file, settings):
    """Jalankan prediksi lalu tampilkan gambar (berkotak bila deteksi) + hasil."""
    img = ImageOps.exif_transpose(Image.open(file))
    c1, c2 = st.columns([1, 1], gap="large")

    if not os.path.exists(settings["path"]):
        with c1:
            show_image(img)
        with c2:
            st.error(f"File model tidak ditemukan: `{settings['path']}`. "
                     "Letakkan file model di folder yang sama dengan app.py atau ubah path di bagian Pengaturan model.")
        return

    try:
        with st.spinner("Model sedang menganalisis..."):
            probs, annotated = predict(img, settings["framework"], settings["path"],
                                       settings["norm"], settings["size"])
    except ImportError as e:
        with c1:
            show_image(img)
        with c2:
            st.error(f"Library belum terpasang: {e}. Jalankan `pip install -r requirements.txt`.")
        return
    except Exception as e:
        with c1:
            show_image(img)
        with c2:
            st.error(f"Gagal menjalankan model: {e}")
        return

    with c1:
        show_image(annotated if annotated is not None else img)
    with c2:
        if probs is None:
            st.warning("⚠️ Tidak ada objek yang terdeteksi pada gambar ini.")
        else:
            names = CLASS_NAMES[::-1] if settings["flip"] else CLASS_NAMES
            show_result(probs, names)


# ============================================================
# HALAMAN UTAMA
# ============================================================
def main():
    render_header()

    st.markdown('<div id="coba"></div><h2 class="sec-title" style="margin-top:20px">Coba sekarang</h2>',
                unsafe_allow_html=True)

    with st.expander("Pengaturan model"):
        framework = st.radio("Model yang dipakai", ["Keras (.h5)", "PyTorch (.pt)"], horizontal=True)
        path = st.text_input("Lokasi file model", MODEL_H5_PATH if framework.startswith("Keras") else MODEL_PT_PATH,
                             key=f"path_{framework}")
        c1, c2, c3 = st.columns(3)
        norm = c1.selectbox("Normalisasi input", range(len(NORM_OPTIONS)),
                            format_func=lambda i: NORM_OPTIONS[i], index=0)
        size = c2.number_input("Ukuran input (khusus .pt non-YOLO)", 32, 1024, DEFAULT_IMG_SIZE, 32,
                               disabled=framework.startswith("Keras"))
        flip = c3.checkbox("Balik urutan label", help="Centang jika hasil terbalik "
                           "(kacamata terbaca tidak kacamata, dan sebaliknya). "
                           "Jangan dicentang untuk model YOLO: kelas sudah dicocokkan lewat nama.")
    settings = dict(framework=framework, path=path, norm=norm, size=int(size), flip=flip)

    tab_up, tab_cam = st.tabs(["📁  Unggah gambar", "📷  Kamera"])

    with tab_up:
        f = st.file_uploader("Pilih foto wajah (JPG / PNG)", type=["jpg", "jpeg", "png", "webp"],
                             label_visibility="collapsed")
        if f:
            analyze(f, settings)
        else:
            st.markdown('<div class="placeholder">Belum ada gambar. Seret foto ke kotak di atas atau klik Browse files.</div>',
                        unsafe_allow_html=True)

    with tab_cam:
        st.caption("Izinkan akses kamera di browser. Kamera hanya bisa dipakai lewat localhost atau HTTPS.")
        shot = st.camera_input("Ambil foto wajah", label_visibility="collapsed")
        if shot:
            analyze(shot, settings)

    render_steps()
    render_ticker()
    components.html(ANIM_JS, height=0)   # animasi saat skrol


main()