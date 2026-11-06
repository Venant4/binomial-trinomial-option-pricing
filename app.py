import numpy as np
import pandas as pd
from scipy.stats import norm
import streamlit as st
import plotly.graph_objects as go

# =====================================================================
# 1. CONFIGURATION DE LA PAGE
# =====================================================================
st.set_page_config(
    page_title="Valorisation d'Options : CRR & Boyle",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background-color: #1a1f2c;
        border-radius: 8px;
        padding: 15px;
        border-top: 3px solid #00d4b1;
        text-align: center;
    }
    .metric-title { font-size: 13px; color: #a0a0a0; font-weight: bold; }
    .metric-val { font-size: 22px; font-weight: bold; color: #ffffff; margin: 5px 0; }
    .metric-sub { font-size: 12px; color: #e63946; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. FONCTIONS DE CALCUL (CRR.py et Boyle.py)
#    @st.cache_data : un calcul déjà fait avec les mêmes paramètres
#    n'est pas refait quand on touche un autre widget.
# =====================================================================

@st.cache_data(show_spinner=False)
def Sholes(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    c = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    p = K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
    return c, p

def Prix(S0, sigma, T, N):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    Stock = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(i+1):
            Stock[i, j] = S0 * u**(i-j) * d**j
    return Stock

@st.cache_data(show_spinner=False)
def Payoff(S0, r, sigma, T, N, K):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d) / (u - d)
    Stock = Prix(S0, sigma, T, N)
    ST = Stock[N, :]
    CT = np.maximum(ST - K, 0)
    PT = np.maximum(K - ST, 0)
    call = np.zeros((N+1, N+1))
    put = np.zeros((N+1, N+1))
    call[N, :] = CT
    put[N, :] = PT
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            call[i, j] = np.exp(-r*dt) * (p*call[i+1, j] + (1-p)*call[i+1, j+1])
            put[i, j] = np.exp(-r*dt) * (p*put[i+1, j] + (1-p)*put[i+1, j+1])
    return call[0, 0], put[0, 0]

@st.cache_data(show_spinner=False)
def Payoff_A(S0, r, sigma, T, N, K):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d) / (u - d)
    Stock = Prix(S0, sigma, T, N)
    ST = Stock[N, :]
    CT = np.maximum(ST - K, 0)
    PT = np.maximum(K - ST, 0)
    call = np.zeros((N+1, N+1))
    put = np.zeros((N+1, N+1))
    put[N, :] = PT
    call[N, :] = CT
    pay_antcall = np.maximum(Stock - K, 0)
    pay_antput = np.maximum(K - Stock, 0)
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            call[i, j] = max(pay_antcall[i, j], np.exp(-r*dt)*(p*call[i+1, j] + (1-p)*call[i+1, j+1]))
            put[i, j] = max(pay_antput[i, j], np.exp(-r*dt)*(p*put[i+1, j] + (1-p)*put[i+1, j+1]))
    return call[0, 0], put[0, 0]

@st.cache_data(show_spinner=False)
def Richardson(S0, r, sigma, T, N, K):
    call_N, put_N = Payoff(S0, r, sigma, T, N, K)
    call_2N, put_2N = Payoff(S0, r, sigma, T, 2*N, K)
    call_R = 2*call_2N - call_N
    put_R = 2*put_2N - put_N
    return call_R, put_R

def prix_boyle(S0, sigma, lamda, T, N):
    dt = T/N
    u = np.exp(sigma*lamda*np.sqrt(dt))
    d = 1/u
    Stock = np.zeros((N+1, 2*N+1))
    Stock[0, N] = S0
    for i in range(1, N+1):
        for j in range(N-i, N+i+1):
            if j > N:
                Stock[i, j] = d * Stock[i-1, j-1]
            elif j < N:
                Stock[i, j] = u * Stock[i-1, j+1]
            else:
                Stock[i, j] = S0
    return Stock

def _probas_boyle(sigma, lamda, T, N, r):
    """Probabilités du modèle trinomial de Boyle (renommées p_up / p_down
    pour ne pas écraser l'alias pandas `pd`)."""
    dt = T/N
    u = np.exp(sigma*lamda*np.sqrt(dt))
    M = np.exp(r*dt)
    V = M**2 * (np.exp(sigma**2 * dt) - 1)
    p_up = (u*(V + M*(M-1)) - (M-1)) / ((u-1)**2 * (u+1))
    p_down = (u**2*(V + M*(M-1)) - u**3*(M-1)) / ((u-1)**2 * (u+1))
    p_mid = 1 - p_up - p_down
    return u, p_up, p_mid, p_down

@st.cache_data(show_spinner=False)
def payoff_boyle(S0, sigma, lamda, T, N, r, K):
    Stock = prix_boyle(S0, sigma, lamda, T, N)
    ST = Stock[N, :]
    dt = T/N
    u, p_up, p_mid, p_down = _probas_boyle(sigma, lamda, T, N, r)
    CT = np.maximum(ST - K, 0)
    PT = np.maximum(K - ST, 0)
    call = np.zeros((N+1, 2*N+1))
    put = np.zeros((N+1, 2*N+1))
    call[N, :] = CT
    put[N, :] = PT
    for i in range(N-1, -1, -1):
        for j in range(N-i, N+i+1):
            call[i, j] = np.exp(-r*dt)*(p_up*call[i+1, j-1] + p_mid*call[i+1, j] + p_down*call[i+1, j+1])
            put[i, j] = np.exp(-r*dt)*(p_up*put[i+1, j-1] + p_mid*put[i+1, j] + p_down*put[i+1, j+1])
    return call[0, N], put[0, N]

@st.cache_data(show_spinner=False)
def payoff_A_boyle(S0, sigma, lamda, T, N, r, K):
    Stock = prix_boyle(S0, sigma, lamda, T, N)
    ST = Stock[N, :]
    dt = T/N
    u, p_up, p_mid, p_down = _probas_boyle(sigma, lamda, T, N, r)
    CT = np.maximum(ST - K, 0)
    PT = np.maximum(K - ST, 0)
    call = np.zeros((N+1, 2*N+1))
    put = np.zeros((N+1, 2*N+1))
    call[N, :] = CT
    put[N, :] = PT
    pay_ant = np.maximum(Stock - K, 0)
    pay_ant2 = np.maximum(K - Stock, 0)
    for i in range(N-1, -1, -1):
        for j in range(N-i, N+i+1):
            call[i, j] = np.maximum(pay_ant[i, j], np.exp(-r*dt)*(p_up*call[i+1, j-1] + p_mid*call[i+1, j] + p_down*call[i+1, j+1]))
            put[i, j] = np.maximum(pay_ant2[i, j], np.exp(-r*dt)*(p_up*put[i+1, j-1] + p_mid*put[i+1, j] + p_down*put[i+1, j+1]))
    return call[0, N], put[0, N]

# =====================================================================
# 2 bis. ARBRES COMPLETS POUR LA VISUALISATION
#    (vectorisés ; renvoient prix du sous-jacent, valeur de l'option
#     et masque des noeuds où l'exercice anticipé est optimal)
# =====================================================================

def _payoff_final(S, K, is_call):
    return np.maximum(S - K, 0) if is_call else np.maximum(K - S, 0)

def build_crr_tree(S0, r, sigma, T, N, K, is_call, american):
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    disc = np.exp(-r*dt)
    p = (np.exp(r*dt) - d) / (u - d)

    S = np.full((N+1, N+1), np.nan)
    V = np.full((N+1, N+1), np.nan)
    early = np.zeros((N+1, N+1), dtype=bool)

    for i in range(N+1):
        j = np.arange(i+1)
        S[i, :i+1] = S0 * u**(i - 2*j)          # j = nombre de baisses

    V[N, :] = _payoff_final(S[N, :], K, is_call)
    for i in range(N-1, -1, -1):
        cont = disc*(p*V[i+1, :i+1] + (1-p)*V[i+1, 1:i+2])
        if american:
            ex = _payoff_final(S[i, :i+1], K, is_call)
            early[i, :i+1] = ex > cont + 1e-12
            V[i, :i+1] = np.maximum(ex, cont)
        else:
            V[i, :i+1] = cont

    info = {"u": u, "d": d, "probas": {"p (hausse)": p, "1-p (baisse)": 1-p}, "p_valid": 0 <= p <= 1}
    return S, V, early, info

def build_boyle_tree(S0, r, sigma, lamda, T, N, K, is_call, american):
    dt = T/N
    disc = np.exp(-r*dt)
    u, p_up, p_mid, p_down = _probas_boyle(sigma, lamda, T, N, r)

    S = np.full((N+1, 2*N+1), np.nan)
    V = np.full((N+1, 2*N+1), np.nan)
    early = np.zeros((N+1, 2*N+1), dtype=bool)

    for i in range(N+1):
        j = np.arange(N-i, N+i+1)
        S[i, j] = S0 * u**(N - j)               # niveau = N - j

    V[N, :] = _payoff_final(S[N, :], K, is_call)
    for i in range(N-1, -1, -1):
        a, b = N-i, N+i+1
        cont = disc*(p_up*V[i+1, a-1:b-1] + p_mid*V[i+1, a:b] + p_down*V[i+1, a+1:b+1])
        if american:
            ex = _payoff_final(S[i, a:b], K, is_call)
            early[i, a:b] = ex > cont + 1e-12
            V[i, a:b] = np.maximum(ex, cont)
        else:
            V[i, a:b] = cont

    probs = {"p_u (hausse)": p_up, "p_m (stable)": p_mid, "p_d (baisse)": p_down}
    info = {"u": u, "d": 1/u, "probas": probs, "p_valid": all(0 <= v <= 1 for v in probs.values())}
    return S, V, early, info

def tree_figure(model, S, V, early, N, K, y_mode, show_labels, option_label):
    """Construit la figure Plotly de l'arbre (binomial ou trinomial)."""
    is_crr = model.startswith("CRR")
    price_axis = (y_mode == "Prix S")

    edge_style = {
        "up":   ("rgba(81,207,102,0.55)",  "Hausse"),
        "mid":  ("rgba(160,160,160,0.45)", "Stable"),
        "down": ("rgba(230,57,70,0.55)",   "Baisse"),
    }
    edges = {k: ([], []) for k in edge_style}

    def level_of(i, j):
        return (i - 2*j) if is_crr else (N - j)

    def y_of(i, j):
        return S[i, j] if price_axis else level_of(i, j)

    nx, ny, nv, ntxt, nhover = [], [], [], [], []
    ex_, ey_ = [], []

    for i in range(N+1):
        j_range = range(i+1) if is_crr else range(N-i, N+i+1)
        for j in j_range:
            x, y = i, y_of(i, j)
            s, v = S[i, j], V[i, j]
            nx.append(x); ny.append(y); nv.append(v)
            ntxt.append(f"{s:.2f}<br><b>{v:.2f}</b>")
            tag = "<br>⚠️ Exercice anticipé optimal" if early[i, j] else ""
            nhover.append(f"Pas {i}<br>Niveau {level_of(i, j):+d}<br>"
                          f"S = {s:.4f}<br>Option = {v:.4f}{tag}")
            if early[i, j]:
                ex_.append(x); ey_.append(y)

            if i < N:
                children = ([(j, "up"), (j+1, "down")] if is_crr
                            else [(j-1, "up"), (j, "mid"), (j+1, "down")])
                for jc, kind in children:
                    edges[kind][0].extend([x, i+1, None])
                    edges[kind][1].extend([y, y_of(i+1, jc), None])

    fig = go.Figure()
    for kind, (color, name) in edge_style.items():
        if edges[kind][0]:
            fig.add_trace(go.Scatter(
                x=edges[kind][0], y=edges[kind][1], mode="lines",
                line=dict(color=color, width=1.3), hoverinfo="skip", name=name))

    size = 22 if N <= 6 else (16 if N <= 10 else max(6, int(200/N)))
    fig.add_trace(go.Scatter(
        x=nx, y=ny,
        mode="markers+text" if (show_labels and N <= 8) else "markers",
        text=ntxt if (show_labels and N <= 8) else None,
        textposition="top center", textfont=dict(size=10, color="#dddddd"),
        marker=dict(size=size, color=nv, colorscale="Viridis", showscale=True,
                    colorbar=dict(title=f"Valeur<br>{option_label}"),
                    line=dict(width=1, color="#0e1117")),
        hovertext=nhover, hoverinfo="text", name="Noeuds", showlegend=False))

    if ex_:
        fig.add_trace(go.Scatter(
            x=ex_, y=ey_, mode="markers",
            marker=dict(symbol="diamond-open", size=size+8, color="#ff6b6b",
                        line=dict(width=2)),
            hoverinfo="skip", name="Exercice anticipé"))

    if price_axis:
        fig.add_hline(y=K, line_dash="dash", line_color="white",
                      annotation_text=f"K = {K:g}", annotation_position="top left")

    fig.update_layout(
        template="plotly_dark",
        height=520 if N <= 10 else 640,
        xaxis=dict(title="Pas de temps i", dtick=1 if N <= 15 else None, zeroline=False),
        yaxis=dict(title="Prix du sous-jacent S" if price_axis else "Niveau (hausses − baisses)",
                   zeroline=False, dtick=None if price_axis else (1 if not is_crr else 2)),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

def matrix_frames(model, S, V, early, N):
    """DataFrames lisibles (colonnes = pas, lignes = niveaux) + mise en évidence."""
    is_crr = model.startswith("CRR")
    if is_crr:
        idx = [f"{j} baisse(s)" for j in range(N+1)]
    else:
        idx = [f"niveau {N-j:+d}" for j in range(2*N+1)]
    cols = [f"t={i}" for i in range(N+1)]

    df_S = pd.DataFrame(S.T, index=idx, columns=cols)
    df_V = pd.DataFrame(V.T, index=idx, columns=cols)
    mask = pd.DataFrame(np.where(early.T, "background-color:#5c1a1f;color:#ffb3b8", ""),
                        index=idx, columns=cols)
    return df_S, df_V, mask

# --- Calculs de convergence (mis en cache) ---------------------------

@st.cache_data(show_spinner="Calcul de la convergence…")
def convergence_europeenne(S0, r, sigma, T, K, lamda, n_range):
    out = {m: {"call": [], "put": []} for m in ("crr", "rich", "boyle")}
    for n in n_range:
        c, p = Payoff(S0, r, sigma, T, n, K)
        out["crr"]["call"].append(c); out["crr"]["put"].append(p)
        c, p = Richardson(S0, r, sigma, T, n, K)
        out["rich"]["call"].append(c); out["rich"]["put"].append(p)
        c, p = payoff_boyle(S0, sigma, lamda, T, n, r, K)
        out["boyle"]["call"].append(c); out["boyle"]["put"].append(p)
    return out

@st.cache_data(show_spinner="Calcul de la convergence…")
def convergence_americaine(S0, r, sigma, T, K, lamda, n_range):
    out = {m: {"call": [], "put": []} for m in ("crr", "boyle")}
    for n in n_range:
        c, p = Payoff_A(S0, r, sigma, T, n, K)
        out["crr"]["call"].append(c); out["crr"]["put"].append(p)
        c, p = payoff_A_boyle(S0, sigma, lamda, T, n, r, K)
        out["boyle"]["call"].append(c); out["boyle"]["put"].append(p)
    return out

# =====================================================================
# 3. PANNEAU DE CONFIGURATION
# =====================================================================

st.title("Valorisation d'Options : Modèles CRR et Boyle")
st.write("---")

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.subheader("1. Marché & Sous-jacent")
    S0 = st.number_input("Prix initial (S0)", value=100.0, step=1.0)
    r = st.number_input("Taux sans risque (r)", value=0.05, step=0.01)
    sigma = st.number_input("Volatilité (sigma)", value=0.20, step=0.01)

with col_p2:
    st.subheader("2. Contrat d'Option")
    K = st.number_input("Prix d'exercice (K)", value=100.0, step=1.0)
    T = st.number_input("Maturité (T en années)", value=1.0, step=0.1)

    type_option = st.radio("Type d'Option", ["Call", "Put"], horizontal=True)
    style_option = st.radio("Style d'exercice", ["Européenne", "Américaine"], horizontal=True)

    is_call = (type_option == "Call")
    is_american = (style_option == "Américaine")

with col_p3:
    st.subheader("3. Discrétisation")
    N = st.slider("Nombre de pas (N)", min_value=10, max_value=2000, value=50, step=10)
    lamda = st.number_input("Paramètre lambda (Boyle)", value=float(np.sqrt(3/2)), step=0.05)

st.write("---")

# =====================================================================
# 4. CALCULS ET SELECTION DES RÉSULTATS
# =====================================================================

if is_american:
    c_crr, p_crr = Payoff_A(S0, r, sigma, T, N, K)
    c_boyle, p_boyle = payoff_A_boyle(S0, sigma, lamda, T, N, r, K)

    price_crr = c_crr if is_call else p_crr
    price_boyle = c_boyle if is_call else p_boyle

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #e63946;">
            <div class="metric-title">CRR AMÉRICAIN ({type_option.upper()})</div>
            <div class="metric-val">{price_crr:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #339af0;">
            <div class="metric-title">BOYLE AMÉRICAIN ({type_option.upper()})</div>
            <div class="metric-val">{price_boyle:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    c_bs, p_bs = Sholes(S0, K, r, sigma, T)
    c_crr, p_crr = Payoff(S0, r, sigma, T, N, K)
    c_boyle, p_boyle = payoff_boyle(S0, sigma, lamda, T, N, r, K)
    c_r, p_r = Richardson(S0, r, sigma, T, N, K)

    price_bs = c_bs if is_call else p_bs
    price_crr = c_crr if is_call else p_crr
    price_boyle = c_boyle if is_call else p_boyle
    price_r = c_r if is_call else p_r

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    with res_col1:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #888888;">
            <div class="metric-title">BLACK-SCHOLES</div>
            <div class="metric-val">{price_bs:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        err_c = abs(price_crr - price_bs)
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #e63946;">
            <div class="metric-title">CRR ({type_option.upper()})</div>
            <div class="metric-val">{price_crr:.4f}</div>
            <div class="metric-sub">Erreur: {err_c:.2e}</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col3:
        err_r = abs(price_r - price_bs)
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #51cf66;">
            <div class="metric-title">RICHARDSON ({type_option.upper()})</div>
            <div class="metric-val">{price_r:.4f}</div>
            <div class="metric-sub" style="color:#51cf66;">Erreur: {err_r:.2e}</div>
        </div>
        """, unsafe_allow_html=True)

    with res_col4:
        err_b = abs(price_boyle - price_bs)
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: #339af0;">
            <div class="metric-title">BOYLE ({type_option.upper()})</div>
            <div class="metric-val">{price_boyle:.4f}</div>
            <div class="metric-sub" style="color:#339af0;">Erreur: {err_b:.2e}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# =====================================================================
# 5. GRAPHIQUES, ARBRE ET MATRICES
# =====================================================================

tab_graph, tab_tree, tab_matrix = st.tabs(
    [" Convergence des Modèles", " Arbre interactif", " Matrices de l'arbre"]
)
side = "call" if is_call else "put"

# ---------------------------- Convergence ----------------------------
with tab_graph:
    n_range = tuple(range(10, 160, 10))

    if is_american:
        st.subheader(f"Convergence du {type_option} Américain : CRR vs Boyle")
        conv = convergence_americaine(S0, r, sigma, T, K, lamda, n_range)

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=n_range, y=conv["crr"][side], mode='lines+markers',
                                   name=f'CRR {type_option} Américain', line=dict(color='#e63946')))
        fig_p.add_trace(go.Scatter(x=n_range, y=conv["boyle"][side], mode='lines+markers',
                                   name=f'Boyle {type_option} Américain', line=dict(color='#339af0')))
        fig_p.update_layout(
            title=f"Prix du {type_option} Américain en fonction de N",
            xaxis_title="Nombre de pas (N)",
            yaxis_title=f"Prix du {type_option}",
            template="plotly_dark", height=400)
        st.plotly_chart(fig_p, use_container_width=True)

    else:
        st.subheader(f"Analyse de la convergence du {type_option} Européen vers Black-Scholes")
        conv = convergence_europeenne(S0, r, sigma, T, K, lamda, n_range)
        target = price_bs

        err = {m: [abs(v - target) for v in conv[m][side]] for m in conv}

        g_col1, g_col2 = st.columns(2)

        with g_col1:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=n_range, y=conv["crr"][side], mode='lines+markers',
                                       name='CRR', line=dict(color='#e63946')))
            fig_p.add_trace(go.Scatter(x=n_range, y=conv["rich"][side], mode='lines+markers',
                                       name='Richardson', line=dict(color='#51cf66')))
            fig_p.add_trace(go.Scatter(x=n_range, y=conv["boyle"][side], mode='lines+markers',
                                       name='Boyle', line=dict(color='#339af0')))
            fig_p.add_hline(y=price_bs, line_dash="dash", line_color="white",
                            annotation_text="Black-Scholes")
            fig_p.update_layout(
                title=f"Prix du {type_option} vs N",
                xaxis_title="Nombre de pas (N)",
                yaxis_title=f"Prix du {type_option}",
                template="plotly_dark", height=380)
            st.plotly_chart(fig_p, use_container_width=True)

        with g_col2:
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=n_range, y=err["crr"], mode='lines+markers',
                                       name='Erreur CRR O(1/N)', line=dict(color='#e63946')))
            fig_e.add_trace(go.Scatter(x=n_range, y=err["rich"], mode='lines+markers',
                                       name='Erreur Richardson O(1/N²)', line=dict(color='#51cf66')))
            fig_e.add_trace(go.Scatter(x=n_range, y=err["boyle"], mode='lines+markers',
                                       name='Erreur Boyle', line=dict(color='#339af0')))
            fig_e.update_xaxes(type="log", title_text="Pas N (Échelle Log)")
            fig_e.update_yaxes(type="log", title_text="Erreur Absolue (Échelle Log)")
            fig_e.update_layout(
                title=f"Graphique Log-Log de l'Erreur ({type_option})",
                template="plotly_dark", height=380)
            st.plotly_chart(fig_e, use_container_width=True)

# ------------------------- Arbre interactif --------------------------
with tab_tree:
    st.subheader(f"Arbre de valorisation — {type_option} {style_option}")

    t1, t2, t3, t4 = st.columns(4)
    model = t1.radio("Modèle", ["CRR (binomial)", "Boyle (trinomial)"], key="tree_model")
    N_tree = t2.slider("Pas affichés", min_value=2, max_value=25, value=6, key="tree_n",
                       help="Arbre pédagogique : indépendant du N utilisé pour le calcul du prix.")
    y_mode = t3.radio("Axe vertical", ["Niveau", "Prix S"], key="tree_y",
                      help="Niveau : l'arbre est régulier. Prix S : position réelle des prix, avec la ligne K.")
    show_labels = t4.checkbox("Afficher S et V (N ≤ 8)", value=True, key="tree_labels")

    if model.startswith("CRR"):
        S_t, V_t, early_t, info = build_crr_tree(S0, r, sigma, T, N_tree, K, is_call, is_american)
    else:
        S_t, V_t, early_t, info = build_boyle_tree(S0, r, sigma, lamda, T, N_tree, K, is_call, is_american)

    # Indicateurs du modèle
    cols_info = st.columns(3 + len(info["probas"]))
    cols_info[0].metric("u", f"{info['u']:.4f}")
    cols_info[1].metric("d", f"{info['d']:.4f}")
    for k, (name, val) in enumerate(info["probas"].items()):
        cols_info[2 + k].metric(name, f"{val:.4f}")
    cols_info[-1].metric(f"Prix racine (N={N_tree})", f"{V_t[0, 0] if model.startswith('CRR') else V_t[0, N_tree]:.4f}")

    if not info["p_valid"]:
        st.warning("⚠️ Au moins une probabilité sort de [0, 1] : la discrétisation (N, λ, σ, r) "
                   "n'est pas viable pour ce modèle.")

    fig_tree = tree_figure(model, S_t, V_t, early_t, N_tree, K, y_mode, show_labels, type_option)
    st.plotly_chart(fig_tree, use_container_width=True)

    legend_txt = "🟢 branche de hausse · ⚪ stable (Boyle) · 🔴 branche de baisse · couleur du noeud = valeur de l'option"
    if is_american:
        n_early = int(early_t.sum())
        legend_txt += f" · ◇ rouge = exercice anticipé optimal ({n_early} noeud(s))"
    st.caption(legend_txt)

# ------------------------------ Matrices -----------------------------
with tab_matrix:
    st.subheader("Matrices de l'arbre (mêmes paramètres que l'onglet précédent)")
    df_S, df_V, mask = matrix_frames(model, S_t, V_t, early_t, N_tree)

    m1, m2 = st.tabs(["Prix du sous-jacent S", f"Valeur du {type_option}"])
    with m1:
        st.dataframe(df_S.style.format("{:.2f}", na_rep=""), use_container_width=True)
    with m2:
        st.dataframe(df_V.style.format("{:.4f}", na_rep="").apply(lambda _: mask, axis=None),
                     use_container_width=True)
        if is_american:
            st.caption("Cellules rouges : l'exercice immédiat vaut plus que la continuation.")