# app.py  —  Solana-style ML dashboard (self-contained)
import numpy as np, pandas as pd, plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

st.set_page_config(page_title="Solana ML Portfolio (Simulated)", layout="wide")
st.title("Solana Token ML — Signals, Backtests, and Trends (Simulated)")

# ---------- simulate hourly data ----------
np.random.seed(42)
N_TOKENS, HOURS = 30, 24*20
tokens = [f"TOK{str(i).zfill(2)}" for i in range(N_TOKENS)]
mu = np.random.uniform(0.00, 0.002, size=N_TOKENS)
sigma = np.random.uniform(0.01, 0.05, size=N_TOKENS)
base_liq = np.random.uniform(2e5, 2e6, size=N_TOKENS)
base_vol = np.random.uniform(5e4, 1e6, size=N_TOKENS)
base_tx  = np.random.uniform(50, 2000, size=N_TOKENS)
base_wallet_growth = np.random.uniform(1, 25, size=N_TOKENS)

rows=[]
for tid, tok in enumerate(tokens):
    p0 = np.random.uniform(0.5, 10.0)
    prices=[p0]; vol=[]; tx=[]; uw=[]; whale=[]; liq=[]
    uw_series = 10000 + np.cumsum(np.random.poisson(base_wallet_growth[tid]/24, HOURS))
    for t in range(HOURS):
        spike = np.random.rand() < 0.03
        whale.append(np.random.gamma(2.0,5000)*(3.0 if spike else 1.0)+np.random.rand()*1000)
        tx.append(np.random.poisson(base_tx[tid]/24)+(30 if spike else 0)+np.random.randint(0,10))
        v = max(0, np.random.normal(base_vol[tid]/24, base_vol[tid]/100))*(2.0 if spike else 1.0)
        vol.append(v)
        li = base_liq[tid]*(1.0+0.01*np.sin(2*np.pi*t/24/7)) + np.random.normal(0, base_liq[tid]/300)
        liq.append(max(li, 1e4))
        uw.append(uw_series[t])
        eps = np.random.normal(0,1)
        nextp = prices[-1]*np.exp(mu[tid]+sigma[tid]*eps)*(1.0+ (0.005 if spike else 0.0))
        prices.append(max(0.01, nextp))
    prices=prices[:-1]
    rows.append(pd.DataFrame({
        "token": tok,
        "t": pd.date_range("2025-06-01", periods=HOURS, freq="H"),
        "price": prices, "dex_volume": vol, "tx_count": tx,
        "unique_wallets": uw, "whale_inflow": whale, "liquidity_usd": liq
    }))
df = pd.concat(rows).sort_values(["token","t"]).reset_index(drop=True)

# ---------- features ----------
# ---------- features (fixed) ----------
feat = df.copy()
gf = feat.groupby("token")

feat["ret_1h"]   = gf["price"].pct_change(1)
feat["ret_6h"]   = gf["price"].pct_change(6)
feat["ret_24h"]  = gf["price"].pct_change(24)

feat["vol_24h"]  = gf["ret_1h"].rolling(24).std().reset_index(level=0, drop=True)

feat["ema_6"]    = gf["price"].transform(lambda s: s.ewm(span=6,  adjust=False).mean())
feat["ema_24"]   = gf["price"].transform(lambda s: s.ewm(span=24, adjust=False).mean())
feat["ema_ratio"]= feat["ema_6"]/(feat["ema_24"]+1e-9)

feat["dv_z"]     = gf["dex_volume"].transform(lambda s: (s - s.rolling(24).mean()) / (s.rolling(24).std()+1e-9))
feat["tx_z"]     = gf["tx_count"].transform(lambda s: (s - s.rolling(24).mean()) / (s.rolling(24).std()+1e-9))
feat["whale_z"]  = gf["whale_inflow"].transform(lambda s: (s - s.rolling(24).mean()) / (s.rolling(24).std()+1e-9))

feat["liq_chg_24h"] = gf["liquidity_usd"].pct_change(24)
feat["fwd_ret_1h"]  = gf["price"].pct_change(-1) * -1
feat["label_up1h"]  = (feat["fwd_ret_1h"] > 0.004).astype(int)

feat = feat.dropna().sort_values(["t","token"]).reset_index(drop=True)


# ---------- train/valid/test split ----------
times = feat["t"].sort_values().unique()
t_train_end = times[int(0.7*len(times))]
t_valid_end = times[int(0.85*len(times))]
train = feat[feat["t"]<=t_train_end]
valid = feat[(feat["t"]>t_train_end)&(feat["t"]<=t_valid_end)]
test  = feat[feat["t"]>t_valid_end]
cols  = ["ret_1h","ret_6h","ret_24h","vol_24h","ema_ratio","dv_z","tx_z","whale_z","liq_chg_24h"]

clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=42)
clf.fit(train[cols], train["label_up1h"])
p_valid = clf.predict_proba(valid[cols])[:,1]
p_test  = clf.predict_proba(test[cols])[:,1]

prec, rec, thr = precision_recall_curve(valid["label_up1h"], p_valid)
f1 = 2*(prec*rec)/(prec+rec+1e-9)
thr_star = float(np.clip(thr[max(0,min(np.nanargmax(f1)-1, len(thr)-1))], 0.05, 0.95))
test = test.copy(); test["p"] = p_test

# ---------- simple backtest ----------
TP, SL, H, FEE = 0.02, 0.01, 6, 0.001
test["signal"] = ((test["p"]>=thr_star)&(test["dv_z"]>0)&(test["whale_z"]>0)).astype(int)

trades=[]
for tok, g2 in test.groupby("token"):
    g2 = g2.sort_values("t").reset_index(drop=True)
    price = g2["price"].values; sig = g2["signal"].values; tt=g2["t"].values
    i=0
    while i < len(g2)-1:
        if sig[i]==1:
            entry_i = i+1
            if entry_i>=len(g2): break
            entry_p = price[entry_i]
            exit_i  = min(entry_i+H, len(g2)-1)
            exit_p  = price[exit_i]; reason="timed"
            for j in range(entry_i+1, exit_i+1):
                r=(price[j]-entry_p)/entry_p
                if r>=TP: exit_i=j; exit_p=price[j]; reason="tp"; break
                if r<=-SL: exit_i=j; exit_p=price[j]; reason="sl"; break
            net=((exit_p-entry_p)/entry_p)-FEE
            trades.append({"token":tok,"entry":tt[entry_i],"exit":tt[exit_i],"entry_p":entry_p,"exit_p":exit_p,"net":net,"reason":reason})
            i=exit_i+1
        else:
            i+=1
trades=pd.DataFrame(trades)
if not trades.empty:
    trades=trades.sort_values("entry").reset_index(drop=True)
    trades["equity"]=(1.0+trades["net"]).cumprod()
    wr=float((trades["net"]>0).mean())
    total=float(trades["equity"].iloc[-1]-1.0)
else:
    wr=float("nan"); total=float("nan")

# ---------- trend clustering (14d) ----------
daily = df.copy(); daily["date"]=daily["t"].dt.floor("D")
daily = daily.groupby(["token","date"], as_index=False).agg(
    tx_count=("tx_count","sum"),
    unique_wallets=("unique_wallets","last"),
    liquidity_usd=("liquidity_usd","mean")
)
trend=[]
for tok, g3 in daily.groupby("token"):
    g3=g3.sort_values("date"); recent=g3.tail(14)
    if len(recent)<2: continue
    tx_tr=(recent["tx_count"].iloc[-1]-recent["tx_count"].iloc[0])/(recent["tx_count"].iloc[0]+1e-9)
    w_tr =(recent["unique_wallets"].iloc[-1]-recent["unique_wallets"].iloc[0])/(recent["unique_wallets"].iloc[0]+1e-9)
    li_tr=(recent["liquidity_usd"].iloc[-1]-recent["liquidity_usd"].iloc[0])/(recent["liquidity_usd"].iloc[0]+1e-9)
    trend.append([tok,tx_tr,w_tr,li_tr])
clusters=pd.DataFrame(trend, columns=["token","tx_trend","wallets_trend","liq_trend"])
if not clusters.empty:
    X=clusters[["tx_trend","wallets_trend","liq_trend"]].values
    Xs=StandardScaler().fit_transform(X)
    km=KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xs)
    clusters["cluster"]=km.labels_
    # label by activity score
    labels={}
    for c in sorted(clusters["cluster"].unique()):
        sl = clusters[clusters["cluster"]==c][["tx_trend","wallets_trend"]].mean().sum()
        labels[c] = "Emerging" if sl>0.05 else ("Declining" if sl<-0.05 else "Stable")
    clusters["cluster_label"]=clusters["cluster"].map(labels)

# ---------- UI ----------
tab1, tab2 = st.tabs(["🔮 Signal & Backtest","📈 Trend Detection"])

with tab1:
    c1,c2 = st.columns([1,3])
    with c1:
        st.metric("Best threshold", f"{thr_star:.2f}")
        st.metric("ROC AUC (valid)", f"{roc_auc_score(valid['label_up1h'], p_valid):.3f}")
        st.metric("PR AUC (valid)",  f"{average_precision_score(valid['label_up1h'], p_valid):.3f}")
        st.metric("Win rate", f"{wr:.2%}" if wr==wr else "—")
        st.metric("Total return", f"{total:.2%}" if total==total else "—")
    with c2:
        token = st.selectbox("Token", sorted(df["token"].unique()))
        thr   = st.slider("Signal Threshold", 0.05, 0.95, float(thr_star), 0.01)
        dftok = feat[feat["token"]==token].copy()
        fig = px.line(dftok, x="t", y="price", title=f"{token} Price")
        # overlay signals
        maybe = dftok.dropna(subset=["p"]) if "p" in dftok.columns else test[test["token"]==token]
        if not maybe.empty:
            ss = maybe[maybe["p"]>=thr]
            if not ss.empty:
                fig.add_scatter(x=ss["t"], y=ss["price"], mode="markers", name="signal", marker=dict(size=7, symbol="triangle-up"))
        st.plotly_chart(fig, use_container_width=True)
    if not trades.empty:
        st.subheader("Recent Trades")
        st.dataframe(trades.tail(50), use_container_width=True, height=280)
        st.subheader("Equity Curve (All Tokens)")
        st.plotly_chart(px.line(trades, x="exit", y="equity", title="Equity"), use_container_width=True)

with tab2:
    if not clusters.empty:
        st.subheader("14-day Activity Trends")
        st.plotly_chart(px.scatter(clusters, x="tx_trend", y="wallets_trend",
                                   color="cluster_label", hover_data=["token","liq_trend"],
                                   title="Clusters: Emerging / Stable / Declining"),
                        use_container_width=True)
        tok2 = st.selectbox("Inspect token", sorted(df["token"].unique()), index=0, key="tok2")
        d2 = df[df["token"]==tok2]
        c1,c2 = st.columns(2)
        c1.plotly_chart(px.line(d2, x="t", y="tx_count", title=f"{tok2} — TX Count"), use_container_width=True)
        c1.plotly_chart(px.line(d2, x="t", y="dex_volume", title=f"{tok2} — DEX Volume"), use_container_width=True)
        c2.plotly_chart(px.line(d2, x="t", y="unique_wallets", title=f"{tok2} — Unique Wallets"), use_container_width=True)
        c2.plotly_chart(px.line(d2, x="t", y="liquidity_usd", title=f"{tok2} — Liquidity"), use_container_width=True)
    else:
        st.info("Not enough daily data to build trend clusters yet.")
st.caption("Synthetic data. Built with Streamlit, Plotly, scikit-learn.")

