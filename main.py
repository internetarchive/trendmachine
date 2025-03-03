import requests

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from copy import deepcopy
from dataclasses import dataclass, field, asdict
from math import exp
from urllib.parse import quote_plus

from archive import DailyRecord
from samples import PeriodicSamples


TITLE = "Webpage Resilience"
ICON = "https://archive.org/favicon.ico"
WBM = "https://web.archive.org/web"
CDXAPI = "https://web.archive.org/cdx/search/cdx"
MAXCDXPAGES = 2000
CRLF = "\n"
IFRAME_HEIGHT = 600

st.set_page_config(page_title=TITLE, page_icon=ICON, layout="wide")
st.title(TITLE)


def ymd(d):
    y, d = divmod(d, 365)
    m, d = divmod(d, 30)
    if y or m > 6:
        if d > 15:
            m += 1
        d = 0
    if m == 12:
        y += 1
        m = 0
    t = {"y": y, "m": m, "d": d}
    return "".join([s for k, v in t.items() if v for s in (str(v), k)])


@st.cache_data(max_entries=65536, show_spinner=False)
def _sigmoid_inverse(x, shift, slope):
    return 1 + exp(shift - x / slope)


def sigmoid(x, shift=5, slope=1, spread=1):
    return spread / _sigmoid_inverse(x, shift, slope)


def fill_identical(f, lk, lv, rk, rv, gap):
    if lv != rv:
        return
    for day in pd.date_range(lk, rk, inclusive="neither"):
        t = day.strftime("%Y-%m-%d")
        f[t] = DailyRecord(t, specimen=lv)


def fill_closest(f, lk, lv, rk, rv, gap):
    mid = gap / 2
    for i, day in enumerate(pd.date_range(lk, rk, inclusive="neither")):
        t = day.strftime("%Y-%m-%d")
        f[t] = DailyRecord(t, specimen=lv) if i < mid else DailyRecord(t, specimen=rv)


def fill_forward(f, lk, lv, rk, rv, gap):
    for day in pd.date_range(lk, rk, inclusive="neither"):
        t = day.strftime("%Y-%m-%d")
        f[t] = DailyRecord(t, specimen=lv)


def fill_backward(f, lk, lv, rk, rv, gap):
    for day in pd.date_range(lk, rk, inclusive="neither"):
        t = day.strftime("%Y-%m-%d")
        f[t] = DailyRecord(t, specimen=rv)


fillpolicies = {
    "identical": fill_identical,
    "closest": fill_closest,
    "forward": fill_forward,
    "backward": fill_backward
}


def filler(drs, fill, policy):
    f = {}
    kv = iter(drs.items())
    pk, pv = next(kv)
    pv = pv.specimen
    pk = pd.to_datetime(pk)
    for k, v in kv:
        v = v.specimen
        k = pd.to_datetime(k)
        gap = (k - pk).days - 1
        if gap and (fill == -1 or gap <= fill):
            fillpolicies[policy](f, pk, pv, k, v, gap)
        pk, pv = k, v
    return f


@st.cache_data(ttl=3600)
def get_resp_headers(url):
    res = requests.head(url, allow_redirects=True)
    rh = res.history + [res]
    return [f"HTTP/1.1 {r.status_code} {r.reason}{CRLF}{CRLF.join(': '.join(i) for i in r.headers.items())}{CRLF}" for r in rh]


def load_cdx_pages(url):
    ses = requests.Session()
    prog = st.progress(0)
    page = 0
    while page < MAXCDXPAGES:
        pageurl = f"{url}&page={page}"
        r = ses.get(pageurl, stream=True)
        if not r.ok:
            prog.empty()
            raise ValueError(f"CDX API returned `{r.status_code}` status code for `{url}`")
        r.raw.decode_content = True
        for line in r.raw:
            yield line
        page += 1
        maxp = int(r.headers.get("x-cdx-num-pages", 1))
        prog.progress(min(page/maxp, 1.0))
        if page >= maxp:
            prog.empty()
            break


@st.cache_data(ttl=3600, persist=True, show_spinner=False)
def load_cdx(url):
    digest_status = {}
    date_record = {}
    psc = PeriodicSamples()
    STPR = {"2xx": 4, "4xx": 3, "5xx": 2, "3xx": 1}
    SWS = 1000
    sw = ["~"] * SWS
    cp = -1
    dr = None
    pt = ""
    pc = "~"
    ps = "~"
    rs = us = uw = 0
    for l in load_cdx_pages(f"{CDXAPI}?fl=timestamp,statuscode,digest&url={quote_plus(url)}"):
        ts, s, d = l.decode().split()
        psc(ts)
        t = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        s = f"{s[:1]}xx" if "200" <= s <= "599" else s
        if s == "-":
            s = digest_status.get(d, "~")
        else:
            digest_status[d] = s
        d = d[:8]
        if t != pt:
            if pt:
                pc = dr.digest
                dr.chaos = us / rs
                dr.chaosn = uw / min(SWS, rs)
                date_record[pt] = dr
            dr = DailyRecord(t)
            cp = -1
            pt = t
        dr.incr(s)
        pr = STPR.get(s, 0)
        if pr > cp:
            dr.specimen = s
            dr.datetime = ts
            dr.digest = d
            dr.content = "Unchanged" if d == pc else "Changed"
            cp = pr
        wp = rs % SWS
        rs += 1
        if s != ps:
            ps = s
            us += 1
            uw += 1
        if sw[wp] != sw[wp-SWS+1]:
            uw -= 1
        sw[wp] = s
    if pt:
        dr.chaos = us / rs
        dr.chaosn = uw / min(SWS, rs)
        date_record[pt] = dr
    return (date_record, psc.sample)


@st.cache_data(ttl=3600)
def load_data(url, fill, policy, sigparams):
    date_record, psc = deepcopy(load_cdx(url))
    if not date_record:
        raise ValueError(f"Empty or malformed CDX API response for `{url}`")
    if fill != 0:
        date_record.update(filler(date_record, fill, policy))
    res = []
    ps = "~"
    pc = "Unknown"
    pch = pchn = 0.0
    base = basec = scale = scalec = h = hc = 0.5
    x = xc = 0
    for day in pd.date_range(next(iter(date_record)), pd.to_datetime("today")):
        t = day.strftime("%Y-%m-%d")
        dr = date_record.get(t, DailyRecord(t))
        if dr.chaos:
            pch = dr.chaos
            pchn = dr.chaosn
        else:
            dr.chaos = pch
            dr.chaosn = pchn
        s = dr.specimen
        p = sigparams.get(s)
        if s != ps:
            base = h
            scale = base if p[2] < 0 else 1 - base
            ps = s
            x = 0
        x += 1
        h = base + scale * sigmoid(x, *p)
        dr.resilience = h
        c = dr.content
        cp = sigparams.get(c)
        if c != pc:
            basec = hc
            scalec = basec if cp[2] < 0 else 1 - basec
            pc = c
            xc = 0
        xc += 1
        hc = basec + scalec * sigmoid(xc, *cp)
        dr.fixity = hc
        res.append(dr)
    resdf = pd.DataFrame(res)
    resdf.columns = [c[1:] if c[0] == "_" else c.title() for c in resdf.columns]
    resdf["URIM"] = resdf["Datetime"].apply(lambda x: f"{WBM}/{x}/{url}" if x != "~" else "#")
    trs = {
        "2xx": {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 0},
        "3xx": {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 0},
        "4xx": {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 0},
        "5xx": {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 0}
    }
    rs = iter(res)
    pr = next(rs)
    for r in rs:
        try:
            trs[r.specimen][pr.specimen] += 1
            pr = r
        except KeyError as e:
            continue
    trsdf = pd.DataFrame(trs).reset_index().rename(columns={"index": "Source"}).melt(id_vars=["Source"], value_vars=["2xx", "3xx", "4xx", "5xx"], var_name="Target", value_name="Count")
    pscdf = pd.DataFrame.from_dict(psc, orient="index", columns=["Samples"]).reset_index().rename(columns={"index": "Period"})
    return (resdf, trsdf, pscdf)


@st.cache_data(max_entries=10, show_spinner=False)
def sigmoid_shape(k, v):
    t = range(101)
    initial = 1 if v[2] < 0 else 0
    h = [initial + sigmoid(x, *v) for x in t]
    cd = pd.DataFrame(list(zip(t, h)), columns=["Time", "Resilience"])
    return alt.Chart(cd).mark_line().encode(x="Time:Q", y=alt.Y("Resilience:Q", scale=alt.Scale(domain=[0, 1]))).properties(height=79).configure_axis(grid=False, title=None)


prev = st.session_state.get("prev", "")
qp = st.query_params

if "url" not in st.session_state and qp.get("url"):
    st.session_state["url"] = qp.get("url")[0]
if "fill" not in st.session_state and qp.get("fill"):
    st.session_state["fill"] = int(float(qp.get("fill")[0]))
if "policy" not in st.session_state and qp.get("policy"):
    st.session_state["policy"] = qp.get("policy")[0]

cols = st.columns([8, 2, 2])
url = cols[0].text_input("URL", key="url", placeholder="https://example.com/")
fill = cols[1].number_input("Fill Missing Days", -1, None, 0, key="fill", help="`0` for no filling  \n`-1` for any gap size")
policy = cols[2].selectbox("Filling Policy", fillpolicies, format_func=lambda x: x.capitalize(), key="policy", help="`Identical` fills the gap only if both the ends have the same status")

sigattrs =  ["shift2", "slope2", "spread2", "shift3", "slope3", "spread3", "shift4", "slope4", "spread4", "shift5", "slope5", "spread5", "shiftu", "slopeu", "spreadu", "shiftcu", "slopecu", "spreadcu", "shiftcc", "slopecc", "spreadcc", "shiftuk", "slopeuk", "spreaduk"]
for p in sigattrs:
    if p not in st.session_state and qp.get(p):
        st.session_state[p] = float(qp.get(p)[0])

st.session_state["prev"] = "".join([str(st.session_state.get(a, "")) for a in sigattrs])

shapes = {}

with st.expander("Sigmoid Parameters"):
    st.latex(r'{Resilience}_t = \frac{{Spread}}{1 + e^{{Shift} - \frac{t}{{Slope}}}}')

    st.subheader("Status Code")

    cols = st.columns(4, gap="medium")
    shift2 = cols[0].slider("2xx Shift", -10, 50, 4, key="shift2")
    slope2 = cols[1].slider("2xx Slope", 0.1, 50.0, 1.0, 0.1, key="slope2")
    spread2 = cols[2].slider("2xx Spread", -1.0, 1.0, 1.0, 0.1, key="spread2")
    shapes["2xx"] = cols[3].empty()

    cols = st.columns(4, gap="medium")
    shift3 = cols[0].slider("3xx Shift", -10, 50, 5, key="shift3")
    slope3 = cols[1].slider("3xx Slope", 0.1, 50.0, 10.0, 0.1, key="slope3")
    spread3 = cols[2].slider("3xx Spread", -1.0, 1.0, -0.5, 0.1, key="spread3")
    shapes["3xx"] = cols[3].empty()

    cols = st.columns(4, gap="medium")
    shift4 = cols[0].slider("4xx Shift", -10, 50, 5, key="shift4")
    slope4 = cols[1].slider("4xx Slope", 0.1, 50.0, 1.0, 0.1, key="slope4")
    spread4 = cols[2].slider("4xx Spread", -1.0, 1.0, -1.0, 0.1, key="spread4")
    shapes["4xx"] = cols[3].empty()

    cols = st.columns(4, gap="medium")
    shift5 = cols[0].slider("5xx Shift", -10, 50, 5, key="shift5")
    slope5 = cols[1].slider("5xx Slope", 0.1, 50.0, 1.0, 0.1, key="slope5")
    spread5 = cols[2].slider("5xx Spread", -1.0, 1.0, -1.0, 0.1, key="spread5")
    shapes["5xx"] = cols[3].empty()

    cols = st.columns(4, gap="medium")
    shiftu = cols[0].slider("Unobserved Shift", -10, 50, 10, key="shiftu")
    slopeu = cols[1].slider("Unobserved Slope", 0.1, 50.0, 20.0, 0.1, key="slopeu")
    spreadu = cols[2].slider("Unobserved Spread", -1.0, 1.0, -0.5, 0.1, key="spreadu")
    shapes["~"] = cols[3].empty()

    st.subheader("Content Digest")

    cols = st.columns(4, gap="medium")
    shiftcc = cols[0].slider("Changed Shift", -10, 50, 6, key="shiftcc")
    slopecc = cols[1].slider("Changed Slope", 0.1, 50.0, 1.0, 0.1, key="slopecc")
    spreadcc = cols[2].slider("Changed Spread", -1.0, 1.0, -1.0, 0.1, key="spreadcc")
    shapes["Changed"] = cols[3].empty()

    cols = st.columns(4, gap="medium")
    shiftcu = cols[0].slider("Unchanged Shift", -10, 50, 4, key="shiftcu")
    slopecu = cols[1].slider("Unchanged Slope", 0.1, 50.0, 1.0, 0.1, key="slopecu")
    spreadcu = cols[2].slider("Unchanged Spread", -1.0, 1.0, 1.0, 0.1, key="spreadcu")
    shapes["Unchanged"] = cols[3].empty()

    cols = st.columns(4, gap="medium")
    shiftuk = cols[0].slider("Unknown Shift", -10, 50, 10, key="shiftuk")
    slopeuk = cols[1].slider("Unknown Slope", 0.1, 50.0, 30.0, 0.1, key="slopeuk")
    spreaduk = cols[2].slider("Unknown Spread", -1.0, 1.0, -0.5, 0.1, key="spreaduk")
    shapes["Unknown"] = cols[3].empty()

sigparams = {
    "2xx": (shift2, slope2, spread2),
    "3xx": (shift3, slope3, spread3),
    "4xx": (shift4, slope4, spread4),
    "5xx": (shift5, slope5, spread5),
    "~": (shiftu, slopeu, spreadu),
    "Changed": (shiftcc, slopecc, spreadcc),
    "Unchanged": (shiftcu, slopecu, spreadcu),
    "Unknown": (shiftuk, slopeuk, spreaduk)
}

for k, v in sigparams.items():
    c = sigmoid_shape(k, v)
    shapes[k].altair_chart(c, use_container_width=True)

if not url:
    st.stop()

if prev and prev != st.session_state["prev"]:
    st.info("Sigmoid parameters changed!")
    st.button("Recalculate Resilience")
    st.stop()

qarg = dict(st.session_state)
qarg.pop("prev", None)
for k, v in qarg:
    st.query_params[k] = v

try:
    # TODO: canonicalize url for better caching
    d, t, p = load_data(url, fill, policy, sigparams)
except ValueError as e:
    st.warning(e)
    st.stop()

TDM = d["Day"].iloc[[0, -1]].values[:]
OBS = ["Active", "Filled", "Missing"]
DOM = ["2xx", "3xx", "4xx", "5xx", "~"]
RNG = ["teal", "orange", "red", "darkred", "gray"]
DOMC = ["Changed", "Unchanged", "Unknown"]
RNGC = ["red", "teal", "gray"]

orig = d[d["All"]!=0][["Day", "Datetime", "2xx", "3xx", "4xx", "5xx", "All"]]
monthly = orig.groupby(orig["Day"].str[:8] + "15").sum().reset_index().rename(columns={"Day": "Month"})
summary = orig.replace(0, np.nan).describe().transpose().reset_index().rename(columns={"index": "Status", "25%": "q1", "50%": "median", "75%": "q3"}).replace(np.nan, 0).astype({"count": int, "min": int, "max": int})
totals = orig.sum(numeric_only=True).reset_index().rename(columns={"index": "Status", 0: "Total"})
inactive = (pd.to_datetime("today") - pd.to_datetime(orig["Day"].iloc[-1])).days
filled = len(d[d["Filled"]==True])
specimens = pd.DataFrame({"Specimen": OBS, "Days": [len(orig), filled, len(d)-len(orig)-filled]})
fixities = d.Content.value_counts().sort_index().reset_index().rename(columns={"index": "Content", "Content": "Count"})
okct, mmct = totals[totals["Status"].isin(["2xx", "All"])]["Total"].values

cols = st.columns(6)
cols[0].metric("Captures", f"{mmct:,}", f"{(okct/mmct*100):.3f}% (OK)", help="Total number of captures of different variations of the URL")
cols[1].metric("Span", ymd(len(d)), f"{ymd(inactive) if inactive else 'Today'} (Last)", delta_color="inverse" if inactive > 1 else "normal", help="Temporal span of archival activities since the first capture")
cols[2].metric("Gaps", f"{len(d)-len(orig):,}", f"{filled:,} (Filled)", delta_color="normal" if filled else "off", help="Number of days with no captures after the first capture")
prevres = d["Resilience"].iloc[-2] if len(d) > 1 else 0.5
lastres = d["Resilience"].iloc[-1]
cols[3].metric("Resilience", f"{lastres:.5f}", f"{lastres-prevres:.5f}", delta_color="off" if d["Specimen"].iloc[-1] == "~" else "normal", help="Current resilience score and the trend from the day before")
lastfix = d["Fixity"].iloc[-1]
chng = len(d[d["Content"]=="Changed"])
cols[4].metric("Fixity", f"{lastfix:.5f}", f"{(chng/len(orig)*100):.3f}% (Changed)", delta_color="inverse", help="Current fixity of the page content")
lastchs = d["Chaos"].iloc[-1]
lastchsn = d["Chaosn"].iloc[-1]
cols[5].metric("Chaos", f"{d['Chaos'].iloc[-1]:.5f}", f"{d['Chaosn'].iloc[-1]:.5f}", delta_color="normal" if lastchsn <= lastchs else "inverse", help="Current chaos score of all and last 1,000 status codes")

stips = [alt.Tooltip("2xx:Q", format=","), alt.Tooltip("3xx:Q", format=","), alt.Tooltip("4xx:Q", format=","), alt.Tooltip("5xx:Q", format=","), alt.Tooltip("All:Q", format=",")]
zoom = alt.selection_interval(bind="scales", encodings=["x"])

tbs = st.tabs(["📈 Resilience", "☰ Data"])
with tbs[0]:
    c = alt.LayerChart(d, height=250).encode(
        x=alt.X("Day:T", scale=alt.Scale(domain=TDM)),
        y=alt.Y("Resilience:Q", impute={"value": None}, scale=alt.Scale(domain=[-0.05, 1.05])),
        tooltip=["Day:T", alt.Tooltip("Resilience:Q", format="0.5f"), "Specimen", "Filled"] + stips
    ).add_layers(
        alt.Chart(d).mark_line(size=3).add_selection(zoom),
        alt.Chart(d).mark_line(size=3).encode(color=alt.Color("Specimen:N", scale=alt.Scale(domain=DOM, range=RNG))),
        alt.Chart(d).mark_circle().encode(
            color=alt.Color("Specimen:N", scale=alt.Scale(domain=DOM, range=RNG)),
            href="URIM:N",
            opacity=alt.condition(alt.datum.Specimen!="~", alt.value(0.7), alt.value(0.0))
        )
    ).configure_axisX(grid=False)
    st.altair_chart(c, use_container_width=True)
with tbs[1]:
    st.write(d.loc[:, "Day":"Resilience"])

tbs = st.tabs(["📈 Fixity", "☰ Data"])
with tbs[0]:
    c = alt.LayerChart(d, height=250).encode(
        x=alt.X("Day:T", scale=alt.Scale(domain=TDM)),
        y=alt.Y("Fixity:Q", impute={"value": None}, scale=alt.Scale(domain=[-0.05, 1.05])),
        tooltip=["Day:T", alt.Tooltip("Fixity:Q", format="0.5f"), "Digest", "Content"]
    ).add_layers(
        alt.Chart(d).mark_line(size=3).add_selection(zoom),
        alt.Chart(d).mark_line(size=3).encode(color=alt.Color("Content:N", scale=alt.Scale(domain=DOMC, range=RNGC))),
        alt.Chart(d).mark_circle().encode(
            color=alt.Color("Content:N", scale=alt.Scale(domain=DOMC, range=RNGC)),
            href="URIM:N",
            opacity=alt.condition(alt.datum.Content!="Unknown", alt.value(0.7), alt.value(0.0))
        )
    ).configure_axisX(grid=False)
    st.altair_chart(c, use_container_width=True)
with tbs[1]:
    st.write(d.loc[:, ["Day", "Digest", "Content", "Fixity"]])

tbs = st.tabs(["📈 Chaos", "☰ Data"])
chd = d.loc[:, ["Day", "Chaosn", "Chaos"]].rename(columns={"Chaos": "All", "Chaosn": "Last1000"}).melt("Day", var_name="Window", value_name="Chaos")
with tbs[0]:
    c = alt.Chart(chd, height=250).mark_line(size=3).encode(
        x=alt.X("Day:T", scale=alt.Scale(domain=TDM)),
        y=alt.Y("Chaos:Q", scale=alt.Scale(domain=[-0.05, 1.05])),
        color="Window:N",
        tooltip=["Day:T", "Window:N", alt.Tooltip("Chaos:Q", format="0.5f")]
    ).add_selection(zoom).configure_axisX(grid=False)
    st.altair_chart(c, use_container_width=True)
with tbs[1]:
    st.write(d.loc[:, ["Day", "Chaos", "Chaosn"]].rename(columns={"Chaosn": "Chaos1000"}))

tbs = st.tabs(["📈 Monthly Status", "📈 Log Scale", "☰ Data"])
mtr = alt.Chart(monthly, height=250).transform_fold(
        ["2xx", "3xx", "4xx", "5xx"],
        as_=["Status", "Count"]
    ).mark_bar()
with tbs[0]:
    c = mtr.encode(
        x=alt.X("yearmonth(Month):T", scale=alt.Scale(domain=TDM), title="Month"),
        y=alt.Y("Count:Q", axis=alt.Axis(format="~s")),
        color=alt.Color("Status:N", scale=alt.Scale(domain=DOM[:-1], range=RNG[:-1])),
        order=alt.Order("Status:N"),
        tooltip=[alt.Tooltip("yearmonth(Month):T", title="Month")] + stips
    ).add_selection(zoom).configure_axisX(grid=False)
    st.altair_chart(c, use_container_width=True)
with tbs[1]:
    c = mtr.encode(
        x=alt.X("yearmonth(Month):T", scale=alt.Scale(domain=TDM), title="Month"),
        y=alt.Y("Count:Q", axis=alt.Axis(format="~s"), scale=alt.Scale(type="symlog"), title="Count (Log Scale)"),
        color=alt.Color("Status:N", scale=alt.Scale(domain=DOM[:-1], range=RNG[:-1])),
        order=alt.Order("Status:N"),
        tooltip=[alt.Tooltip("yearmonth(Month):T", title="Month")] + stips
    ).add_selection(zoom).configure_axisX(grid=False)
    st.altair_chart(c, use_container_width=True)
with tbs[2]:
    st.write(monthly)

cols = st.columns(3)

tbs = cols[0].tabs(["📈 Total Status", "☰ Data"])
with tbs[0]:
    c = alt.Chart(totals[totals["Status"]!="All"]).mark_arc().encode(
        theta="Total:Q",
        color=alt.Color("Status:N", scale=alt.Scale(domain=DOM[:-1], range=RNG[:-1]), legend=alt.Legend(orient="top")),
        tooltip=["Status", alt.Tooltip("Total:Q", format=",")]
    )
    st.altair_chart(c, use_container_width=True)
with tbs[1]:
    st.write(totals)

tbs = cols[1].tabs(["📈 Daily Specimens", "☰ Data"])
with tbs[0]:
    c = alt.Chart(specimens).mark_arc().encode(
        theta="Days:Q",
        color=alt.Color("Specimen:N", scale=alt.Scale(domain=OBS, range=["teal", "orange", "gray"]), legend=alt.Legend(orient="top")),
        tooltip=["Specimen", alt.Tooltip("Days:Q", format=",")]
    )
    st.altair_chart(c, use_container_width=True)
with tbs[1]:
    st.write(specimens)

tbs = cols[2].tabs(["📈 Content Fixity", "☰ Data"])
with tbs[0]:
    c = alt.Chart(fixities).mark_arc().encode(
        theta="Count:Q",
        color=alt.Color("Content:N", scale=alt.Scale(domain=DOMC, range=RNGC), legend=alt.Legend(orient="top")),
        tooltip=["Content", alt.Tooltip("Count:Q", format=",")]
    )
    st.altair_chart(c, use_container_width=True)
with tbs[1]:
    st.write(fixities)

cols = st.columns(3)

tbs = cols[0].tabs(["📈 Daily Status", "☰ Data"])
with tbs[0]:
    c = alt.LayerChart(summary).encode(
        x="Status:N",
        color=alt.Color("Status:N", scale=alt.Scale(domain=DOM[:-1]+["All"], range=RNG[:-1]+["#4c78a8"]), legend=alt.Legend(orient="top")),
        tooltip=["Status:N", alt.Tooltip("count:Q", format=","), alt.Tooltip("mean:Q", format=",.2f"), alt.Tooltip("min:Q", format=","), alt.Tooltip("q1:Q", format=",.2f"), alt.Tooltip("median:Q", format=",.2f"), alt.Tooltip("q3:Q", format=",.2f"), alt.Tooltip("max:Q", format=",")]
    ).add_layers(
        alt.Chart().mark_rule(size=2).encode(y="min:Q", y2="max:Q"),
        alt.Chart().mark_bar(width=30, opacity=0.5).encode(y="q1:Q", y2="q3:Q"),
        alt.Chart().mark_tick(width=40, thickness=2).encode(y="median:Q"),
        alt.Chart().mark_circle(size=50).encode(y=alt.Y("mean:Q", axis=alt.Axis(format="~s"), scale=alt.Scale(type="symlog"), title="Count (Log Scale)"))
    )
    st.altair_chart(c, use_container_width=True)
with tbs[1]:
    st.write(summary)

tbs = cols[1].tabs(["📈 Periodic Samples", "☰ Data"])
with tbs[0]:
    c = alt.Chart(p).mark_bar().encode(
        x=alt.X("Period:N", sort=None),
        y="Samples:Q",
        tooltip=["Period:N", alt.Tooltip("Samples:Q", format=",")]
    )
    st.altair_chart(c, use_container_width=True)
with tbs[1]:
    st.write(p)

tbs = cols[2].tabs(["📈 Status Transition", "☰ Data"])
with tbs[0]:
    base = alt.Chart(t).transform_calculate(
        CountPlus="datum.Count+1"
    ).encode(
        x=alt.X("Target:N", axis=alt.Axis(orient="top")),
        y="Source:N",
        tooltip=["Source:N", "Target:N", alt.Tooltip("Count:Q", format=",")]
    )
    cr = base.mark_rect().encode(
        color=alt.Color("CountPlus:Q", scale=alt.Scale(type="log", scheme="greenblue"), legend=alt.Legend(orient="top", title=""))
    )
    ct = base.mark_text(baseline="middle").encode(
        text="Count:Q"
    )
    c = (cr + ct).configure_view(step=60)
    st.altair_chart(c, use_container_width=True)
with tbs[1]:
    st.write(t)

with st.expander("Compare Archives"):
    fld = orig["Day"].iloc[[0, -1]].values[:]
    flt = orig["Datetime"].iloc[[0, -1]].values[:]
    ft, lt = (t for t in flt)
    cols = st.columns(2)
    with cols[0]:
        st.caption(f"First Capture: {fld[0]}")
        first_urim = f"{WBM}/{ft}if_/{url}"
        components.iframe(first_urim, height=IFRAME_HEIGHT, scrolling=True)
    with cols[1]:
        st.caption(f"Last Capture: {fld[-1]}")
        last_urim = f"{WBM}/{lt}if_/{url}"
        components.iframe(last_urim, height=IFRAME_HEIGHT, scrolling=True)

with st.expander("Live Status"):
    urir = url if (url.startswith("http://") or url.startswith("https://")) else f"https://{url}"
    cols = st.columns(2)
    with cols[0]:
        st.caption("Live Page")
        components.iframe(urir, height=IFRAME_HEIGHT, scrolling=True)
    with cols[1]:
        st.caption("HTTP Headers")
        try:
            resp = get_resp_headers(urir)
            for res in resp:
                st.code(res, language="http")
        except Exception as e:
            st.error(e)
