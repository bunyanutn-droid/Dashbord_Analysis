"""ETL: BORA (กรมการปกครอง) province x single-year-age population files -> compact JSON for the dashboard.

Input : data/raw/{yy}_{cc}.txt  (first line of each BORA province file: name|male0|female0|male1|female1|...)
Output: data/dashboard_data.json  (+ prints validation of the projection model)
"""
import json, math, re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
YEARS = list(range(43, 69))  # พ.ศ. 2543-2568 (BORA Dec snapshot)
NAGE = 101  # ages 0..99 + 100+  (BORA gives 0..100 and >100; we fold >100 into 100+)

geo = json.loads((ROOT / "data" / "th.geojson").read_text(encoding="utf-8"))
gprops = {f["properties"]["pro_code"]: f["properties"] for f in geo["features"]}

# GPP per capita 2562 (2019), NESDC via Wikipedia "List of Thai provinces by GPP" (English names)
GPP = {
 "Bangkok":573907,"Samut Prakan":343215,"Pathum Thani":254627,"Samut Sakhon":411326,"Nakhon Pathom":308167,
 "Nonthaburi":204404,"Saraburi":330750,"Sing Buri":129095,"Chainat":101282,"Ang Thong":107129,"Lopburi":144041,
 "Phra Nakhon Si Ayutthaya":465972,"Chonburi":581475,"Chachoengsao":427409,"Rayong":1095667,"Trat":171189,
 "Chanthaburi":254582,"Nakhon Nayok":96589,"Prachin Buri":486601,"Sa Kaeo":72555,"Khon Kaen":117560,
 "Udon Thani":88673,"Loei":97903,"Nong Khai":89913,"Mukdahan":74729,"Nakhon Phanom":76000,"Sakon Nakhon":68887,
 "Kalasin":61084,"Nakhon Ratchasima":110301,"Chaiyaphum":63010,"Yasothon":54183,"Ubon Ratchathani":70551,
 "Roi Et":68751,"Buriram":67621,"Surin":65810,"Maha Sarakham":67784,"Sisaket":67362,"Nong Bua Lamphu":53416,
 "Amnat Charoen":63860,"Bueng Kan":78022,"Chiang Mai":135991,"Lampang":92749,"Uttaradit":87982,
 "Mae Hong Son":65448,"Chiang Rai":91308,"Phrae":67057,"Lamphun":191568,"Nan":71121,"Phayao":87858,
 "Nakhon Sawan":109977,"Phitsanulok":104175,"Kamphaeng Phet":142660,"Uthai Thani":97948,"Sukhothai":73251,
 "Tak":94902,"Phichit":83504,"Phetchabun":84058,"Phuket":388559,"Surat Thani":200471,"Ranong":104517,
 "Phang Nga":265768,"Krabi":239309,"Chumphon":161626,"Nakhon Si Thammarat":99899,"Songkhla":156245,
 "Satun":129565,"Yala":96867,"Trang":116394,"Narathiwat":61765,"Phatthalung":71298,"Pattani":88442,
 "Ratchaburi":214742,"Kanchanaburi":121570,"Prachuap Khiri Khan":198434,"Phetchaburi":143460,
 "Suphan Buri":100595,"Samut Songkhram":114990,
}

# Official national births / deaths (กรมการปกครอง via Bangkok Biz News 2569 / MOPH statistics) -- see README notes
BIRTHS = {58:736352,59:704058,60:703003,61:666366,62:618205,63:587368,64:544570,65:502107,66:517934,67:462240,68:416574}
DEATHS = {64:563650,65:595965,66:565992,67:571646,68:559684}
# MOPH Public Health Statistics (different compilation -> kept as a separate series)
MOPH_BIRTHS = {54:782198,56:748081,62:596736,63:569338}
MOPH_DEATHS = {54:414670,56:426065,62:494339,63:489717}


def region_of(p):
    r = p["reg_royin"]
    return {"North": "เหนือ", "Northeast": "อีสาน", "South": "ใต้"}.get(r, "กลาง")


EEC = {"20", "21", "24"}  # Chonburi, Rayong, Chachoengsao

codes = [c.strip() for c in (ROOT / "data" / "codes.txt").read_text().split()]
provs, pop = [], {}
for c in codes:
    g = gprops[c]
    name_en = g["pro_en"].strip()
    gpp = GPP.get(name_en) or GPP.get(name_en.replace("Buri Ram", "Buriram"))
    provs.append({"c": c, "n": g["pro_th"].strip(), "en": name_en, "reg": region_of(g), "eec": c in EEC,
                  "gpp": gpp, "area": round(g["area_sqkm"])})

missing_gpp = [p["en"] for p in provs if not p["gpp"]]
print("missing GPP:", missing_gpp)


def load(y, c):
    t = (RAW / f"{y}_{c}.txt").read_text(encoding="utf-8").strip().split("|")
    v = [int(x) for x in t[1:] if x.strip().isdigit()]
    a = np.array(v[:204], dtype=float).reshape(102, 2)  # ages 0..100, >100
    a[100] += a[101]
    return a[:101, 0], a[:101, 1]


single = {}  # (year, code) -> (male[101], female[101])
for y in YEARS:
    for c in codes:
        single[(y, c)] = load(y, c)


def bands(a):  # 21 five-year bands, last = 100+ (we keep 95+ merged from 95..100)
    out = [int(a[i:i + 5].sum()) for i in range(0, 100, 5)]
    out.append(int(a[100]))
    return out


for y in YEARS:
    pop[2500 + y] = {}
    for c in codes:
        m, f = single[(y, c)]
        pop[2500 + y][c] = bands(m) + bands(f) + [int(m[0]), int(f[0])]

# latest single-year pyramid per province, for the projection simulator
latest = {c: [int(x) for x in single[(68, c)][0]] + [int(x) for x in single[(68, c)][1]] for c in codes}

# ---------------- model: calibration + back-test (Python is the reference implementation) ----------------
ages = np.arange(101)
# ASFR shape: Thai pattern (peak 25-29), normalised so that sum over 15..49 = 1
shape = np.zeros(101)
for x in range(15, 50):
    shape[x] = math.exp(-((x - 27.0) ** 2) / (2 * 6.0 ** 2)) * (1.0 if x >= 27 else 0.85)
shape /= shape.sum()
SRB = 0.512  # male share of births
C_GOMP = 0.085
MALE_F, FEM_F = 1.30, 0.80


def qx(b, a=0.0006, scale_infant=0.0045):
    mu = a + b * np.exp(C_GOMP * ages)
    q = 1 - np.exp(-mu)
    q[0] += scale_infant
    return np.clip(q, 0, 1)


def nat(y):
    m = sum(single[(y, c)][0] for c in codes)
    f = sum(single[(y, c)][1] for c in codes)
    return m, f


def deaths_for(b, y):
    m, f = nat(y)
    return float((m * np.minimum(1, qx(b) * MALE_F)).sum() + (f * np.minimum(1, qx(b) * FEM_F)).sum())


# calibrate b on 2568 deaths
lo, hi = 1e-6, 1e-3
for _ in range(60):
    mid = (lo + hi) / 2
    if deaths_for(mid, 68) < DEATHS[68]:
        lo = mid
    else:
        hi = mid
B0 = (lo + hi) / 2
print("calibrated Gompertz b =", B0)
for y in (64, 65, 66, 67, 68):
    print(f"  deaths model 25{y}: {deaths_for(B0, y):,.0f}  official {DEATHS[y]:,}")

# implied TFR from age-0 population (registered births proxy) per year
tfr_series = {}
for y in YEARS:
    m, f = nat(y)
    births = m[0] + f[0]
    w = (f * shape).sum()
    tfr_series[2500 + y] = float(births / w)
print("implied TFR (age-0 proxy):", {k: round(v, 2) for k, v in tfr_series.items() if k in (2543, 2553, 2558, 2563, 2568)})
w68 = (nat(68)[1] * shape).sum()
tfr_official_births_68 = BIRTHS[68] / w68
print("implied TFR 2568 using official births:", round(tfr_official_births_68, 3))


def project(m, f, years, tfr, b=B0, improve=0.01, mig_work=0.0):
    """annual cohort-component; returns list of (m,f) for each year"""
    m, f = m.copy(), f.copy()
    out = []
    for i in range(years):
        bb = b * (1 - improve) ** (i + 1)
        qm, qf = np.minimum(1, qx(bb) * MALE_F), np.minimum(1, qx(bb) * FEM_F)
        t = tfr[i] if hasattr(tfr, "__len__") else tfr
        births = t * (f * shape).sum()
        nm, nf = np.zeros(101), np.zeros(101)
        nm[1:] = (m * (1 - qm))[:-1]; nm[100] += (m * (1 - qm))[100]
        nf[1:] = (f * (1 - qf))[:-1]; nf[100] += (f * (1 - qf))[100]
        nm[0], nf[0] = births * SRB * (1 - qm[0]), births * (1 - SRB) * (1 - qf[0])
        if mig_work:
            wk = (nm[15:60].sum() + nf[15:60].sum()) * mig_work
            nm[20:45] += wk * 0.55 / 25; nf[20:45] += wk * 0.45 / 25
        m, f = nm, nf
        out.append((m.copy(), f.copy()))
    return out


# back-test: start from 2558 pyramid, use implied TFR path, compare to actual 2568
m58, f58 = nat(58)
path = [tfr_series[2558 + i + 1] for i in range(10)]
bt = project(m58, f58, 10, path)
mb, fb = bt[-1]
ma, fa = nat(68)
def shares(m, f):
    t = m.sum() + f.sum(); c = m[:15].sum() + f[:15].sum(); w = m[15:60].sum() + f[15:60].sum(); e = m[60:].sum() + f[60:].sum()
    return t, c / t, w / t, e / t, e / w * 100
print("back-test 2558->2568  model :", [round(x, 4) for x in shares(mb, fb)])
print("back-test 2558->2568  actual:", [round(x, 4) for x in shares(ma, fa)])
print("note: model has no migration; registered-population differences reflect household registration moves.")

out = {
    "provinces": provs,
    "years": [2500 + y for y in YEARS],
    "pop": pop,  # year -> code -> [21 male bands, 21 female bands, male<1, female<1]
    "latest_single": latest,
    "tfr_proxy": tfr_series,
    "births_official": {2500 + k: v for k, v in BIRTHS.items()},
    "deaths_official": {2500 + k: v for k, v in DEATHS.items()},
    "births_moph": {2500 + k: v for k, v in MOPH_BIRTHS.items()},
    "deaths_moph": {2500 + k: v for k, v in MOPH_DEATHS.items()},
    "model": {"b": B0, "a": 0.0006, "c": C_GOMP, "infant": 0.0045, "male_f": MALE_F, "fem_f": FEM_F,
              "srb": SRB, "shape": [round(float(s), 6) for s in shape]},
}
(ROOT / "data" / "dashboard_data.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

# simplified geojson (3-decimals)
def rnd(x):
    return [rnd(i) for i in x] if isinstance(x, list) else round(x, 2)

feats = []
for f in geo["features"]:
    feats.append({"type": "Feature", "properties": {"c": f["properties"]["pro_code"]},
                  "geometry": {"type": f["geometry"]["type"], "coordinates": rnd(f["geometry"]["coordinates"])}})
(ROOT / "data" / "geo_small.json").write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")), encoding="utf-8")
print("national pop 2568:", int(sum(nat(68)[0]) + sum(nat(68)[1])))


# ---------------- assemble single-file dashboard ----------------
tpl = (ROOT / "template.html").read_text(encoding="utf-8")
html = tpl.replace("/*DATA*/", (ROOT / "data" / "dashboard_data.json").read_text(encoding="utf-8")) \
          .replace("/*GEO*/", (ROOT / "data" / "geo_small.json").read_text(encoding="utf-8"))
(ROOT / "index.html").write_text(html, encoding="utf-8")
print("wrote index.html", len(html) // 1024, "KB")
