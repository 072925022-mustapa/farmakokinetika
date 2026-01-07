# %%
# ---------- Library ----------
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.set_page_config(page_title="Regresi Non-Linier Kesehatan (Farmakokinetika)", layout="wide")

# %%
# ---------- Model ----------
def exp_decay(t, C0, k):
    return C0 * np.exp(-k * t)

# %%
# ---------- UI ----------
st.title("Aplikasi Regresi Non-Linier di Kesehatan: Estimasi Farmakokinetika (Eksponensial)")
st.write(
    "Contoh ini memodelkan konsentrasi obat dalam darah terhadap waktu menggunakan regresi non-linier "
    "(Nonlinear Least Squares). Anda bisa unggah data CSV atau gunakan data simulasi."
)

with st.sidebar:
    st.header("Opsi Data")
    mode = st.radio("Sumber data", ["Simulasi", "Upload CSV"], index=0)

    st.divider()
    st.header("Pengaturan Fitting")
    init_C0 = st.number_input("Tebakan awal C0", min_value=0.0, value=10.0, step=0.5)
    init_k = st.number_input("Tebakan awal k", min_value=0.0001, value=0.2, step=0.01, format="%.4f")

    st.caption("Catatan: regresi non-linier sensitif terhadap tebakan awal (initial values).")

# %%
# ---------- Data ----------
df = None

if mode == "Upload CSV":
    st.info("Format CSV: minimal punya kolom 'time' dan 'conc' (nama kolom harus sama).")
    uploaded_file  = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file  is not None:
        df = pd.read_csv(uploaded_file )
        if not {"time", "conc"}.issubset(df.columns):
            st.error("Kolom wajib tidak ditemukan. Pastikan ada kolom: time, conc")
            st.stop()
        df = df[["time", "conc"]].dropna()
    else:
        st.warning("Silakan upload file CSV terlebih dahulu.")
        st.stop()
else:
    with st.sidebar:
        st.header("Parameter Simulasi")
        true_C0 = st.number_input("C0 (simulasi)", min_value=0.1, value=12.0, step=0.5)
        true_k = st.number_input("k (simulasi)", min_value=0.001, value=0.25, step=0.01, format="%.4f")
        n_points = st.slider("Jumlah titik waktu", min_value=6, max_value=40, value=12)
        t_max = st.number_input("Waktu maksimum (jam)", min_value=1.0, value=12.0, step=1.0)
        noise_sd = st.number_input("SD noise (konsentrasi)", min_value=0.0, value=0.6, step=0.1)

    rng = np.random.default_rng(123)
    t = np.linspace(0, t_max, n_points)
    y_true = exp_decay(t, true_C0, true_k)
    y_obs = y_true + rng.normal(0, noise_sd, size=n_points)
    y_obs = np.clip(y_obs, a_min=0, a_max=None)  # konsentrasi tidak negatif

    df = pd.DataFrame({"time": t, "conc": y_obs})

# %%
# ---------- Fit ----------
t_data = df["time"].to_numpy(dtype=float)
y_data = df["conc"].to_numpy(dtype=float)

# Untuk stabilitas: kalau ada y=0 persis, tetap boleh, curve_fit tetap jalan.
# Tetapi dataset yang terlalu "flat" atau noise besar bisa menyulitkan konvergensi.
p0 = [init_C0, init_k]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Data")
    st.dataframe(df, use_container_width=True)

with col2:
    st.subheader("Hasil Estimasi Parameter (Nonlinear Least Squares)")

    try:
        popt, pcov = curve_fit(
            exp_decay,
            t_data,
            y_data,
            p0=p0,
            bounds=([0.0, 0.0], [np.inf, np.inf]),  # C0>=0, k>=0
            maxfev=5000
        )
        C0_hat, k_hat = popt
        se = np.sqrt(np.diag(pcov))
        C0_se, k_se = se

        # CI 95% (approx, normal) - cocok untuk contoh, bukan standar emas
        z = 1.96
        C0_ci = (C0_hat - z * C0_se, C0_hat + z * C0_se)
        k_ci = (k_hat - z * k_se, k_hat + z * k_se)

        half_life = np.log(2) / k_hat if k_hat > 0 else np.nan

        metrics = pd.DataFrame(
            {
                "Parameter": ["C0", "k", "t1/2"],
                "Estimasi": [C0_hat, k_hat, half_life],
                "SE (aproks.)": [C0_se, k_se, np.nan],
                "CI 95% bawah": [C0_ci[0], k_ci[0], np.nan],
                "CI 95% atas": [C0_ci[1], k_ci[1], np.nan],
            }
        )
        st.dataframe(metrics, use_container_width=True)

        st.caption(
            "t1/2 dihitung dari ln(2)/k. CI 95% di sini adalah pendekatan berbasis kovarians asimtotik."
        )

    except Exception as e:
        st.error("Fitting gagal. Coba ubah tebakan awal atau cek kualitas data.")
        st.exception(e)
        st.stop()

# %%
# ---------- Predictions & Diagnostics ----------
t_grid = np.linspace(float(np.min(t_data)), float(np.max(t_data)), 200)
y_hat = exp_decay(t_grid, C0_hat, k_hat)
y_fit = exp_decay(t_data, C0_hat, k_hat)
resid = y_data - y_fit

# Goodness-of-fit sederhana
ss_res = float(np.sum((y_data - y_fit) ** 2))
ss_tot = float(np.sum((y_data - np.mean(y_data)) ** 2))
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
rmse = np.sqrt(np.mean(resid ** 2))

st.divider()
st.subheader("Visualisasi")

c3, c4 = st.columns([2, 1])

with c3:
    fig1 = plt.figure()
    plt.scatter(t_data, y_data, label="Observasi")
    plt.plot(t_grid, y_hat, label="Kurva fit (non-linier)")
    plt.xlabel("Waktu (jam)")
    plt.ylabel("Konsentrasi")
    plt.title("Konsentrasi Obat vs Waktu (Eksponensial)")
    plt.legend()
    st.pyplot(fig1)

with c4:
    st.write("Ringkasan Kecocokan (indikatif)")
    st.metric("RMSE", f"{rmse:.4f}")
    st.metric("R² (indikatif)", f"{r2:.4f}" if np.isfinite(r2) else "NA")

    fig2 = plt.figure()
    plt.axhline(0, linewidth=1)
    plt.scatter(t_data, resid)
    plt.xlabel("Waktu (jam)")
    plt.ylabel("Residual")
    plt.title("Plot Residual")
    st.pyplot(fig2)

st.divider()
st.subheader("Interpretasi Klinis Singkat (Contoh)")
st.write(
    f"Estimasi k = {k_hat:.4f} menunjukkan laju eliminasi. "
    f"Dengan k tersebut, waktu paruh obat t½ ≈ {half_life:.2f} jam. "
    "Dalam praktik, model dapat diperluas (mis. multi-kompartemen, absorpsi oral, kovariat pasien)."
)


