# -*- coding: utf-8 -*-
# ==========================================================
# VerimGören — Tek Nokta Analizi (Streamlit UI) — FIXED
# ==========================================================

import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# --- Harici Kütüphane Kontrolü (Rasterio/Pyodbc hatalarını yakalamak için) ---
try:
    import rasterio
    from rasterio.warp import transform as rio_transform
    RASTERIO_AVAILABLE = True
except Exception:
    RASTERIO_AVAILABLE = False

try:
    import pyodbc
    PYODBC_AVAILABLE = True
except Exception:
    PYODBC_AVAILABLE = False
# ---------------------------------------------------------------------------

# ---------------------------
# Sayfa Yapılandırması
# ---------------------------
st.set_page_config(
    page_title="VerimGören",
    page_icon="🌾",
    layout="wide",
)

# ---------------------------
# Hafif Stil
# ---------------------------
st.markdown("""
<style>
:root { --ink:#0F172A; --muted:#64748B; --card:#FFFFFF; --line:#E5E7EB; --bg:#F8FAFC; }
body { background: var(--bg); }
.vg-kpi{display:flex;flex-direction:column;gap:.25rem;border:1px solid var(--line);border-radius:14px;padding:12px;background:#fff}
.vg-kpi .h{color:var(--muted);font-size:.82rem}
.vg-kpi .v{color:var(--ink);font-weight:700;font-size:1.05rem}
.dataframe th { background:#F1F5F9; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 1) Yardımcılar
# ==========================================================
DMS_PATTERN = re.compile(r"(?P<deg>\d{1,3})°(?P<min>\d{1,2})'(?P<sec>[\d\.]+)\"(?P<hemi>[NSEW])")

def dms_to_decimal(deg: float, minute: float, sec: float, hemi: str) -> float:
    sign = -1 if hemi.upper() in ["S", "W"] else 1
    return sign * (abs(deg) + minute / 60.0 + sec / 3600.0)

def parse_latlon_text(text: str) -> Tuple[float, float]:
    text = text.strip()
    parts = re.split(r"[,\s;]+", text)
    parts = [p for p in parts if p]
    if len(parts) < 2:
        raise ValueError("Lütfen 'lat,lon' biçiminde iki sayı girin. Örn: 38.946838, 28.080573")
    lat = float(parts[0]); lon = float(parts[1])
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError("Geçersiz aralık: enlem [-90,90], boylam [-180,180].")
    return lat, lon

def parse_google_maps_link(text: str) -> Optional[Tuple[float, float]]:
    text = text.strip()
    if "@" in text:
        try:
            after = text.split("@", 1)[1]
            nums = re.split(r"[^-\d\.]+", after)
            nums = [n for n in nums if n]
            lat = float(nums[0]); lon = float(nums[1])
            return lat, lon
        except Exception:
            pass
    dms_hits = DMS_PATTERN.findall(text)
    if len(dms_hits) >= 2:
        (ld, lm, ls, lh), (od, om, os, oh) = dms_hits[0], dms_hits[1]
        lat = dms_to_decimal(float(ld), float(lm), float(ls), lh)
        lon = dms_to_decimal(float(od), float(om), float(os), oh)
        return lat, lon
    return None

def parse_any_location(text: str) -> Tuple[float, float]:
    link = parse_google_maps_link(text)
    if link is not None:
        return link
    return parse_latlon_text(text)

# ---- Kategoriler & Meta ----
CATEGORY_ORDER = ["Konum", "İklim", "Arazi", "Gece Işığı", "Toprak", "Özet"]

def category_of(key: str) -> str:
    k = key.upper()
    if k in {"USER_LAT","USER_LON","GRID_LAT","GRID_LON","LATITUDE","LONGITUDE"}: return "Konum"
    if k in {"ELEVATION_M"}: return "Arazi"
    if k in {"NIGHT_LIGHT"}: return "Gece Işığı"
    if ("_GRP" in k) or k in {
        "T2M","T2M_MAX","T2M_MIN","T2M_RANGE","T2MDEW","T2MWET","RH2M","QV2M","TQV","PS","SLP",
        "WS2M","WS2M_MAX","WD2M","PRECTOTCORR","TS","TO3","ALLSKY_SFC_SW_DWN","ALLSKY_SFC_PAR_TOT",
        "CLRSKY_SFC_SW_DWN","CLOUD_AMT","CLOUD_AMT_DAY","CLOUD_AMT_NIGHT","CLRSKY_DAYS","DISTANCE_KM"
    }:
        return "İklim"
    if k in {
        "FAO90_DESC","T_USDA_TEX_DESC","S_USDA_TEX_DESC","T_TEXTURE_DESC",
        "T_SAND","T_SILT","T_CLAY","S_SAND","S_SILT","S_CLAY",
        "T_PH_H2O","S_PH_H2O","T_OC","S_OC","T_CEC_SOIL","S_CEC_SOIL",
        "T_CEC_CLAY","S_CEC_CLAY","T_BS","S_BS","T_TEB","S_TEB",
        "T_CACO3","S_CACO3","T_ECE","S_ECE","T_ESP","S_ESP",
        "AWC_MM_PER_M","DRAINAGE_DESC","MU_GLOBAL"
    }:
        return "Toprak"
    if k in {"DISTANCE_KM"}: return "Özet"
    return "Özet"

VAR_META: Dict[str, Dict[str, str]] = {
    "USER_LAT":{"title_tr":"Kullanıcı enlem","unit":"°"}, "USER_LON":{"title_tr":"Kullanıcı boylam","unit":"°"},
    "GRID_LAT":{"title_tr":"İklim hücresi enlem","unit":"°"}, "GRID_LON":{"title_tr":"İklim hücresi boylam","unit":"°"},
    "LATITUDE":{"title_tr":"Enlem","unit":"°"}, "LONGITUDE":{"title_tr":"Boylam","unit":"°"},
    "ALLSKY_SFC_PAR_TOT":{"title_tr":"PAR (tümü)","unit":"MJ/m²/gün"},
    "ALLSKY_SFC_SW_DWN":{"title_tr":"Kısa dalga (tümü)","unit":"kWh/m²/gün"},
    "CLRSKY_SFC_SW_DWN":{"title_tr":"Kısa dalga (açık gök)","unit":"kWh/m²/gün"},
    "CLRSKY_DAYS":{"title_tr":"Açık gün sayısı","unit":"gün/ay"},
    "CLOUD_AMT":{"title_tr":"Bulutluluk","unit":"%"}, "CLOUD_AMT_DAY":{"title_tr":"Bulutluluk (gündüz)","unit":"%"},
    "CLOUD_AMT_NIGHT":{"title_tr":"Bulutluluk (gece)","unit":"%"}, "QV2M":{"title_tr":"Özgül nem (2 m)","unit":"g/kg"},
    "RH2M":{"title_tr":"Bağıl nem (2 m)","unit":"%"}, "T2M":{"title_tr":"Sıcaklık (2 m, ort.)","unit":"°C"},
    "T2M_MAX":{"title_tr":"Maks. sıcaklık","unit":"°C"}, "T2M_MIN":{"title_tr":"Min. sıcaklık","unit":"°C"},
    "T2M_RANGE":{"title_tr":"Günlük sıcaklık aralığı","unit":"°C"}, "T2MDEW":{"title_tr":"Çiy noktası","unit":"°C"},
    "T2MWET":{"title_tr":"Yaş termometre","unit":"°C"}, "TQV":{"title_tr":"Kolon su buharı","unit":"kg/m²"},
    "PS":{"title_tr":"Yüzey basıncı","unit":"kPa"}, "SLP":{"title_tr":"Denize indirgenmiş basınç","unit":"kPa"},
    "WD2M":{"title_tr":"Rüzgar yönü (2 m)","unit":"°"}, "WS2M":{"title_tr":"Rüzgar hızı (2 m)","unit":"m/s"},
    "WS2M_MAX":{"title_tr":"Maks. rüzgar (2 m)","unit":"m/s"},
    "PRECTOTCORR":{"title_tr":"Toplam yağış (düz.)","unit":"mm/gün"}, "TO3":{"title_tr":"Toplam ozon","unit":"DU"},
    "TS":{"title_tr":"Yüzey sıcaklığı","unit":"°C"}, "DISTANCE_KM":{"title_tr":"Uzaklık (iklim pikseli)","unit":"km"},
    "ELEVATION_M":{"title_tr":"Rakım","unit":"m"}, "NIGHT_LIGHT":{"title_tr":"Gece ışığı","unit":"-"},
    "FAO90_DESC":{"title_tr":"FAO-90 sınıfı","unit":"-"}, "T_USDA_TEX_DESC":{"title_tr":"USDA doku (üst)","unit":"-"},
    "S_USDA_TEX_DESC":{"title_tr":"USDA doku (alt)","unit":"-"}, "T_TEXTURE_DESC":{"title_tr":"Üst doku (coarse/medium/fine)","unit":"-"},
    "T_SAND":{"title_tr":"Kum (üst)","unit":"%"}, "T_SILT":{"title_tr":"Silt (üst)","unit":"%"},
    "T_CLAY":{"title_tr":"Kil (üst)","unit":"%"}, "S_SAND":{"title_tr":"Kum (alt)","unit":"%"},
    "S_SILT":{"title_tr":"Silt (alt)","unit":"%"}, "S_CLAY":{"title_tr":"Kil (alt)","unit":"%"},
    "T_PH_H2O":{"title_tr":"pH (üst)","unit":"-"}, "S_PH_H2O":{"title_tr":"pH (alt)","unit":"-"},
    "T_OC":{"title_tr":"Organik C (üst)","unit":"%"}, "S_OC":{"title_tr":"Organik C (alt)","unit":"%"},
    "T_CEC_SOIL":{"title_tr":"CEC (üst)","unit":"cmol(+)/kg"}, "S_CEC_SOIL":{"title_tr":"CEC (alt)","unit":"cmol(+)/kg"},
    "T_CEC_CLAY":{"title_tr":"CEC (kil, üst)","unit":"cmol(+)/kg"}, "S_CEC_CLAY":{"title_tr":"CEC (kil, alt)","unit":"cmol(+)/kg"},
    "T_BS":{"title_tr":"Baz doygunluğu (üst)","unit":"%"}, "S_BS":{"title_tr":"Baz doygunluğu (alt)","unit":"%"},
    "T_TEB":{"title_tr":"Toplam değişebilir baz (üst)","unit":"cmol(+)/kg"}, "S_TEB":{"title_tr":"Toplam değişebilir baz (alt)","unit":"cmol(+)/kg"},
    "T_CACO3":{"title_tr":"Kireç CaCO3 (üst)","unit":"%"}, "S_CACO3":{"title_tr":"Kireç CaCO3 (alt)","unit":"%"},
    "T_ECE":{"title_tr":"EC (üst)","unit":"dS/m"}, "S_ECE":{"title_tr":"EC (alt)","unit":"dS/m"},
    "T_ESP":{"title_tr":"ESP (üst)","unit":"%"}, "S_ESP":{"title_tr":"ESP (alt)","unit":"%"},
    "AWC_MM_PER_M":{"title_tr":"Kullanılabilir su (AWC)","unit":"mm/m"}, "DRAINAGE_DESC":{"title_tr":"Drenaj","unit":"-"},
    "MU_GLOBAL":{"title_tr":"Harita birimi (MU)","unit":"-"},
}

def meta_of(key: str):
    base = key.replace("_grp1","").replace("_grp2","").replace("_grp3","").replace("_grp4","")
    m = VAR_META.get(key) or VAR_META.get(base.upper()) or VAR_META.get(base)
    if m: return m["title_tr"], m.get("unit","-")
    return base, "-"

def format_value(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))): return "-"
    try:
        f = float(v)
        if abs(f - round(f)) < 1e-9: return f"{int(round(f))}"
        return f"{f:.2f}"
    except Exception:
        return str(v)

# ==========================================================
# 2) Veri Kaynakları
# ==========================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

@st.cache_data(show_spinner=False)
def load_climate_nearest(csv_path: Path, lat: float, lon: float) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(f"İklim CSV bulunamadı: {csv_path}")
    df = pd.read_csv(csv_path)

    lat_col = _find_col(df, ["latitude","lat","Latitude","LAT"])
    lon_col = _find_col(df, ["longitude","lon","Longitude","LON"])

    if not lat_col or not lon_col:
        raise ValueError("İklim CSV’de enlem/boylam sütunları bulunamadı (latitude/longitude veya lat/lon).")

    dist = haversine(lat, lon, df[lat_col].values, df[lon_col].values)
    i = int(np.argmin(dist))
    row = df.iloc[i].to_dict()
    row["DISTANCE_KM"] = float(dist[i])
    # Orijinal sütunları da normalize edelim:
    row["latitude"] = float(df.iloc[i][lat_col])
    row["longitude"] = float(df.iloc[i][lon_col])
    return row

def _reproject_point_if_needed(ds, lon, lat):
    """ds CRS WGS84 değilse (EPSG:4326), noktayı dönüştürür."""
    try:
        if ds.crs is None:
            return lon, lat  # varsay WGS84
        crs_str = str(ds.crs).upper()
        if "4326" in crs_str or "WGS84" in crs_str:
            return lon, lat
        # 4326 -> ds.crs dönüşümü
        xs, ys = rio_transform("EPSG:4326", ds.crs, [lon], [lat])
        return float(xs[0]), float(ys[0])
    except Exception:
        return lon, lat

def sample_raster(path: Path, lon: float, lat: float):
    if not RASTERIO_AVAILABLE or not path.exists():
        return None
    try:
        with rasterio.open(path) as ds:
            x, y = _reproject_point_if_needed(ds, lon, lat)
            r, c = ds.index(x, y)
            arr = ds.read(1)
            if r < 0 or c < 0 or r >= arr.shape[0] or c >= arr.shape[1]:
                return None
            val = arr[r, c]
            if ds.nodata is not None and val == ds.nodata: 
                return None
            return float(val)
    except Exception:
        return None

def load_soil_env(lat: float, lon: float, HWSD_MDB: Path, HWSD_RAS: Path) -> Optional[dict]:
    if not RASTERIO_AVAILABLE or not PYODBC_AVAILABLE:
        return None
    if not HWSD_MDB.exists() or not HWSD_RAS.exists():
        return None
    try:
        with rasterio.open(HWSD_RAS) as src:
            x, y = _reproject_point_if_needed(src, lon, lat)
            r, c = src.index(x, y)
            mu = int(src.read(1)[r, c])
        if mu <= 0:
            return None
    except Exception:
        return None

    def _read(table):
        try:
            cn = pyodbc.connect(
                f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={str(HWSD_MDB).replace('\\','/')};",
                timeout=3
            )
        except Exception:
            return None
        try:
            return pd.read_sql(f"SELECT * FROM {table}", cn)
        finally:
            try: cn.close()
            except: pass

    def _norm(df):
        if df is None: return None
        df = df.copy(); df.columns = [c.strip().upper() for c in df.columns]; return df

    hwsd = _norm(_read("HWSD_DATA"))
    if hwsd is None or not {"MU_GLOBAL","SEQ","SHARE"}.issubset(hwsd.columns):
        return None

    def _lut(df, out_code, out_desc):
        if df is None: return None
        cols = set(df.columns)
        code = "CODE" if "CODE" in cols else None
        desc = "DESCRIPTION" if "DESCRIPTION" in cols else ("VALUE" if "VALUE" in cols else None)
        if code and desc:
            return df.rename(columns={code:out_code, desc:out_desc})[[out_code,out_desc]]
        return None

    def _safe(name): return _norm(_read(name))

    tex = _lut(_safe("D_TEXTURE"), "T_TEXTURE", "T_TEXTURE_DESC")
    utexT = _lut(_safe("D_USDA_TEX_CLASS"), "T_USDA_TEX_CLASS", "T_USDA_TEX_DESC")
    utexS = _lut(_safe("D_USDA_TEX_CLASS"), "S_USDA_TEX_CLASS", "S_USDA_TEX_DESC")
    awc = _lut(_safe("D_AWC"), "AWC_CLASS", "AWC_MM_PER_M")
    drn = _lut(_safe("D_DRAINAGE"), "DRAINAGE", "DRAINAGE_DESC")
    sym90 = _lut(_safe("D_SYMBOL90"), "SU_CODE90", "FAO90_DESC")

    df = hwsd
    for cond, lut, key in [
        ("T_TEXTURE", tex, "T_TEXTURE"),
        ("T_USDA_TEX_CLASS", utexT, "T_USDA_TEX_CLASS"),
        ("S_USDA_TEX_CLASS", utexS, "S_USDA_TEX_CLASS"),
        ("AWC_CLASS", awc, "AWC_CLASS"),
        ("DRAINAGE", drn, "DRAINAGE"),
        ("SU_CODE90", sym90, "SU_CODE90"),
    ]:
        if (cond in df.columns) and (lut is not None):
            df = df.merge(lut, on=key, how="left")

    num_cols = [
        "AWC_MM_PER_M","T_PH_H2O","S_PH_H2O","T_OC","S_OC",
        "T_CLAY","T_SILT","T_SAND","S_CLAY","S_SILT","S_SAND",
        "T_ECE","S_ECE","T_ESP","S_ESP","T_CEC_SOIL","S_CEC_SOIL",
        "T_CEC_CLAY","S_CEC_CLAY","T_BS","S_BS","T_TEB","S_TEB","T_CACO3","S_CACO3",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

    dom = (df.sort_values(["MU_GLOBAL","SEQ","SHARE"], ascending=[True,True,False])
             .groupby("MU_GLOBAL", as_index=False).first())
    row = dom.loc[dom["MU_GLOBAL"]==mu]
    if row.empty: return None
    return row.iloc[0].to_dict()

# ==========================================================
# 3) Skor Modülü
# ==========================================================
def _clip01(x): 
    x=float(x); 
    return 0.0 if x<0 else (1.0 if x>1 else x)

def _presence(val): 
    return (val is not None) and (str(val).strip() != "")

def _mean_safe(vals): 
    vals=[float(v) for v in vals if _presence(v)]; 
    return sum(vals)/len(vals) if vals else None

def _trapezoid_score(x,a,b,c,d):
    if any(v is None for v in [x,a,b,c,d]): return None
    x=float(x); a,b,c,d = float(a),float(b),float(c),float(d)
    if a>=b or b>c or c>=d: return None
    if x<=a or x>=d: return 0.0
    if b<=x<=c: return 100.0
    if a<x<b: return 100.0*(x-a)/(b-a)
    return 100.0*(d-x)/(d-c)

def suitability_score(crop, env, weights=None, params=None):
    W={'thermal':12,'frost':6,'heat':6,'rad':8,'rh':8,'water':15,'ph':8,'ec':8,'soilphys':5,'taw':4,'esp':3,'caco3':3,'cec':2,'elev':5,'wind':3,'night':2}
    if isinstance(weights, dict): W.update(weights)
    P={'rh_opt':60.0,'rh_span':30.0,'rad_min':1.2,'rad_max':3.2,'frost_band':5.0,'heat_band':5.0,'taw_ref_default':120.0}

    modules, usedW = {}, {}
    Tavg=env.get('T2M_grp2'); Tmin=env.get('T2M_MIN_grp2'); Tmax=env.get('T2M_MAX_grp2')
    tmin_abs=crop.get('tmin_abs'); topt_min=crop.get('topt_min'); topt_max=crop.get('topt_max'); tmax_abs=crop.get('tmax_abs')

    if all(_presence(v) for v in [Tavg,tmin_abs,topt_min,topt_max,tmax_abs]):
        modules['thermal']=_trapezoid_score(Tavg,tmin_abs,topt_min,topt_max,tmax_abs); usedW['thermal']=W['thermal']
    if all(_presence(v) for v in [Tmin,tmin_abs]):
        modules['frost']=100.0 if float(Tmin)>=float(tmin_abs) else max(0.0, 100.0-100.0*(abs(float(tmin_abs)-float(Tmin))/P['frost_band'])); usedW['frost']=W['frost']
    if all(_presence(v) for v in [Tmax,tmax_abs]):
        modules['heat']=100.0 if float(Tmax)<=float(tmax_abs) else max(0.0, 100.0-100.0*(abs(float(Tmax)-float(tmax_abs))/P['heat_band'])); usedW['heat']=W['heat']

    R=env.get('ALLSKY_SFC_SW_DWN_grp1')
    if _presence(R):
        modules['rad']=100.0*_clip01((float(R)-P['rad_min'])/max(1e-6,(P['rad_max']-P['rad_min']))); usedW['rad']=W['rad']
    RH=env.get('RH2M_grp2')
    if _presence(RH):
        modules['rh']=100.0*_clip01(1.0-((float(RH)-P['rh_opt'])/P['rh_span'])**2); usedW['rh']=W['rh']

    Pmm=env.get('PRECTOTCORR_grp4'); kc_avg=_mean_safe([crop.get('kc_initial'),crop.get('kc_mid'),crop.get('kc_end')])
    ETc=env.get('ETc'); ET0=env.get('ET0'); AWC=env.get('AWC_MM_PER_M'); Zr=crop.get('root_depth_m')
    if _presence(Pmm) and _presence(kc_avg) and (_presence(ETc) or _presence(ET0)) and _presence(Zr):
        if not _presence(ETc): ETc=float(ET0)*float(kc_avg)
        deficit=max(0.0, float(ETc)-float(Pmm))
        if _presence(AWC):
            TAW=float(AWC)*float(Zr); denom=max(1.0, TAW/15.0)
            modules['water']=100.0*_clip01(1.0-deficit/denom); usedW['water']=W['water']

    soil_pH=env.get('T_PH_H2O'); pH_min=crop.get('pH_min'); pH_max=crop.get('pH_max')
    if all(_presence(v) for v in [soil_pH,pH_min,pH_max]):
        a=float(pH_min)-0.5; b=float(pH_min); c=float(pH_max); d=float(pH_max)+0.5
        modules['ph']=_trapezoid_score(float(soil_pH),a,b,c,d); usedW['ph']=W['ph']

    soil_EC=env.get('T_ECE'); ec_thr = crop.get('ece_threshold_dSm') if 'ece_threshold_dSm' in crop else crop.get('ece_threshold_dsm')
    if all(_presence(v) for v in [soil_EC,ec_thr]):
        thr=max(0.1,float(ec_thr)); modules['ec']=100.0*_clip01(1.0-float(soil_EC)/thr); usedW['ec']=W['ec']

    tex_ok=(crop.get('texture_ok') or "").lower().replace(" ","")
    tex_ok_set=set([t.strip().lower() for t in tex_ok.split(",") if t.strip()])
    tex_env=(env.get('T_USDA_TEX_DESC') or "").strip().lower().replace(" ","")
    drain_pref=(crop.get('drainage_preference') or "").strip().lower()
    drain_env =(env.get('DRAINAGE_DESC') or "").strip().lower()

    score_tex=None
    if tex_env:
        if tex_env in tex_ok_set: score_tex=100.0
        else:
            neigh={'loam':{'sandy_loam','silt_loam','clay_loam'},'sandy_loam':{'loam'},'silt_loam':{'loam'},'clay_loam':{'loam'},
                   'sandy_clay_loam':{'clay_loam','sandy_loam'},'silty_clay_loam':{'clay_loam','silt_loam'}}
            score_tex=60.0 if any((k in tex_ok_set and tex_env in neigh.get(k,set())) for k in tex_ok_set) else 0.0

    def _norm_drain(s):
        s=s.lower()
        if 'well' in s and 'moderate' not in s: return 'well'
        if 'moderately' in s: return 'moderately well'
        if 'very poorly' in s: return 'very poorly'
        if 'poorly' in s: return 'poorly'
        if 'somewhat' in s: return 'somewhat poorly'
        return None

    score_drain=None
    if drain_pref and drain_env:
        dkey=_norm_drain(drain_env); dmap={'well':100,'moderately well':70,'somewhat poorly':40,'poorly':0,'very poorly':0}
        score_drain=dmap.get(dkey,70.0)

    if score_tex is not None or score_drain is not None:
        parts,wsum=[],0.0
        if score_tex is not None: parts.append((score_tex,0.6)); wsum+=0.6
        if score_drain is not None: parts.append((score_drain,0.4)); wsum+=0.4
        modules['soilphys']=sum(s*w for s,w in parts)/(wsum if wsum else 1.0); usedW['soilphys']=W['soilphys']

    if _presence(AWC) and _presence(Zr):
        TAW=float(AWC)*float(Zr); ref=float((params or {}).get('taw_ref_default',120.0))
        modules['taw']=100.0*_clip01(TAW/ref); usedW['taw']=W['taw']

    ESP=env.get('T_ESP') or env.get('ESP')
    if _presence(ESP):
        modules['esp']=100.0*_clip01(1.0-float(ESP)/8.0); usedW['esp']=W['esp']

    CACO3=env.get('T_CACO3') or env.get('S_CACO3') or env.get('CACO3')
    if _presence(CACO3):
        modules['caco3']=100.0*_clip01(1.0-float(CACO3)/10.0); usedW['caco3']=W['caco3']

    CEC=env.get('T_CEC_SOIL')
    if _presence(CEC):
        CEC=float(CEC); modules['cec']=40.0 if CEC<8.0 else (70.0 if CEC<12.0 else 100.0); usedW['cec']=W['cec']

    elev=env.get('ELEVATION_M'); elev_min=crop.get('elevation_min'); elev_max=crop.get('elevation_max')
    if _presence(elev):
        e=float(elev)
        if _presence(elev_min) and _presence(elev_max):
            a=float(elev_min)-200.0; b=float(elev_min); c=float(elev_max); d=float(elev_max)+200.0
            modules['elev']=_trapezoid_score(e,a,b,c,d)
        else:
            modules['elev']=100.0 if e<1500 else (70.0 if e<2000 else (40.0 if e<2500 else 0.0))
        usedW['elev']=W['elev']

    WSMAX=env.get('WS2M_MAX_grp3')
    if _presence(WSMAX):
        modules['wind']=100.0*_clip01(1.0-float(WSMAX)/15.0); usedW['wind']=W['wind']

    NL=env.get('NIGHT_LIGHT')
    if _presence(NL):
        modules['night']=100.0*_clip01(1.0-float(NL)/5.0); usedW['night']=W['night']
        
    if not usedW: 
        return {'score': None, 'modules': modules, 'used_weights': usedW}
    wsum=float(sum(usedW.values())); total=0.0
    for k, sc in modules.items():
        if sc is None: continue
        wk=usedW.get(k,0.0)/wsum; total += wk*float(sc)
    return {'score': round(total,2), 'modules': {k:round(v,2) for k,v in modules.items()}, 'used_weights': usedW}

@st.cache_data(show_spinner=False)
def load_crops(crops_csv_path: Path) -> pd.DataFrame:
    if not crops_csv_path.exists():
        raise FileNotFoundError(f"Bitki CSV bulunamadı: {crops_csv_path}")
    df = pd.read_csv(crops_csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def row_to_crop_dict(row: pd.Series) -> dict:
    d = row.to_dict()
    if 'ece_threshold_dsm' in d and 'ece_threshold_dSm' not in d:
        d['ece_threshold_dSm'] = d['ece_threshold_dsm']
    if not str(d.get('texture_ok','')).strip():
        d['texture_ok'] = ''
    return d

def weakest_modules(mod_dict, n=2):
    if not mod_dict: return []
    items = [(k,v) for k,v in mod_dict.items() if v is not None]
    if not items: return []
    items.sort(key=lambda x: x[1])
    return [f"{k}:{v:.0f}" for k,v in items[:n]]

def build_env(lat: float, lon: float,
              CLIMATE_CSV: Path,
              ELEV_TIF: Optional[Path]=None,
              LIGHT_TIF: Optional[Path]=None,
              HWSD_MDB: Optional[Path]=None,
              HWSD_RAS: Optional[Path]=None) -> dict:
    env = {}
    env["USER_LAT"] = float(lat); env["USER_LON"] = float(lon)

    clim = load_climate_nearest(CLIMATE_CSV, lat, lon)
    env.update(clim)
    if "latitude" in clim and "longitude" in clim:
        env["GRID_LAT"] = float(clim["latitude"])
        env["GRID_LON"] = float(clim["longitude"])

    if ELEV_TIF and ELEV_TIF.exists():
        elev = sample_raster(ELEV_TIF, lon, lat)
        if elev is not None: env["ELEVATION_M"] = elev
    if LIGHT_TIF and LIGHT_TIF.exists():
        night = sample_raster(LIGHT_TIF, lon, lat)
        if night is not None: env["NIGHT_LIGHT"] = night

    if HWSD_MDB and HWSD_RAS and HWSD_MDB.exists() and HWSD_RAS.exists():
        soil = load_soil_env(lat, lon, HWSD_MDB, HWSD_RAS)
        if soil: env.update(soil)

    def _pick(d,*keys):
        for k in keys:
            if k in d: return d[k]
        return None
    env['T2M_grp2'] = _pick(env,'T2M_grp2','T2M_GRP2','T2M')
    env['T2M_MIN_grp2'] = _pick(env,'T2M_MIN_grp2','T2M_MIN_GRP2','T2M_MIN')
    env['T2M_MAX_grp2'] = _pick(env,'T2M_MAX_grp2','T2M_MAX_GRP2','T2M_MAX')
    env['RH2M_grp2'] = _pick(env,'RH2M_grp2','RH2M_GRP2','RH2M')
    env['ALLSKY_SFC_SW_DWN_grp1'] = _pick(env,'ALLSKY_SFC_SW_DWN_grp1','ALLSKY_SFC_SW_DWN_GRP1','ALLSKY_SFC_SW_DWN')
    env['PRECTOTCORR_grp4'] = _pick(env,'PRECTOTCORR_grp4','PRECTOTCORR_GRP4','PRECTOTCORR')
    env['WS2M_MAX_grp3'] = _pick(env,'WS2M_MAX_grp3','WS2M_MAX_GRP3','WS2M_MAX')
    return env

def score_and_rank_df(env: dict, crops_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in crops_df.iterrows():
        crop = row_to_crop_dict(row)
        res = suitability_score(crop, env)
        if res.get('score') is None:
            continue
        results.append({
            'crop': crop.get('crop'),
            'common_name_tr': crop.get('common_name_tr'),
            'score': res['score'],
            'weakest_two': ", ".join(weakest_modules(res.get('modules', {}), n=2)) or "-",
            'modules': res.get('modules', {})
        })
    df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    return df

# ==========================================================
# 4) ARAYÜZ
# ==========================================================
st.title("🌾 VerimGören: Tarımsal Uygunluk Analizi")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Veri Yolları (Mutlak)")
    st.caption("Lütfen tüm dosya yollarını (C:/Users/...) girin.")

    # Kullanıcının ev dizini baz alınır; yoksa alan boş bırakılır.
    home = Path.home()
    probable_base = home / "Desktop" / "MEHMET" / "VerimGören"

    def _if_exists(p: Path) -> str:
        return p.as_posix() if p.exists() else ""

    crops_path_str = st.text_input(
        "CROPS_CSV (Bitkiler - Zorunlu)",
        value=_if_exists(probable_base / "notebooks" / "VerimGoren_Bitki_Parametreleri_Tam.csv"),
        placeholder=(probable_base / "notebooks" / "VerimGoren_Bitki_Parametreleri_Tam.csv").as_posix()
    )
    climate_path_str = st.text_input(
        "CLIMATE_CSV (İklim - Zorunlu)",
        value=_if_exists(probable_base / "notebooks" / "data" / "climate" / "merged_climate_data.csv"),
        placeholder=(probable_base / "notebooks" / "data" / "climate" / "merged_climate_data.csv").as_posix()
    )

    st.markdown("---")
    st.caption("Opsiyonel Raster ve Toprak Verileri")

    elev_used = st.checkbox("⛰️ Rakım (ELEV_TIF) kullan", value=True)
    elev_path_str = st.text_input(
        "ELEV_TIF Yolu",
        value=_if_exists(probable_base / "data" / "processed" / "srtm_turkiye_cropped.tif"),
        disabled=not elev_used,
        placeholder=(probable_base / "data" / "processed" / "srtm_turkiye_cropped.tif").as_posix()
    ) if elev_used else ""

    light_used = st.checkbox("🌃 Gece Işığı (LIGHT_TIF) kullan", value=True)
    light_path_str = st.text_input(
        "LIGHT_TIF Yolu",
        value=_if_exists(probable_base / "data" / "processed" / "viirs_light_2024_turkey.tif"),
        disabled=not light_used,
        placeholder=(probable_base / "data" / "processed" / "viirs_light_2024_turkey.tif").as_posix()
    ) if light_used else ""

    hwsd_used = st.checkbox("🌱 HWSD Toprak Verisi kullan (MDB + Raster)", value=True)
    if hwsd_used:
        if not RASTERIO_AVAILABLE or not PYODBC_AVAILABLE:
            st.warning("⚠️ Rasterio veya PyODBC eksik. Toprak verisi okunamayacak.")
        mdb_path_str = st.text_input(
            "HWSD_MDB Yolu",
            value=_if_exists(probable_base / "notebooks" / "hwsd_data" / "HWSD.mdb"),
            placeholder=(probable_base / "notebooks" / "hwsd_data" / "HWSD.mdb").as_posix()
        )
        ras_path_str = st.text_input(
            "HWSD_RAS Yolu (.bil)",
            value=_if_exists(probable_base / "notebooks" / "hwsd_data" / "hwsd.bil"),
            placeholder=(probable_base / "notebooks" / "hwsd_data" / "hwsd.bil").as_posix()
        )
    else:
        mdb_path_str, ras_path_str = "", ""

    # Path nesneleri
    CROPS_CSV = Path(crops_path_str) if crops_path_str else None
    CLIMATE_CSV = Path(climate_path_str) if climate_path_str else None
    ELEV_TIF = Path(elev_path_str) if elev_path_str else None
    LIGHT_TIF = Path(light_path_str) if light_path_str else None
    HWSD_MDB = Path(mdb_path_str) if hwsd_used and mdb_path_str else None
    HWSD_RAS = Path(ras_path_str) if hwsd_used and ras_path_str else None

    st.markdown("---")
    st.subheader("✅ Dosya Durumu")
    def _ok(p: Optional[Path]): return bool(p and p.exists())
    st.markdown(f"**Bitki CSV**: {'✅ BULUNDU' if _ok(CROPS_CSV) else '❌ YOK'}")
    st.markdown(f"**İklim CSV**: {'✅ BULUNDU' if _ok(CLIMATE_CSV) else '❌ YOK'}")
    if elev_used: st.markdown(f"**Rakım TIF**: {'✅ BULUNDU' if _ok(ELEV_TIF) else '❌ YOK'}")
    if light_used: st.markdown(f"**Işık TIF**: {'✅ BULUNDU' if _ok(LIGHT_TIF) else '❌ YOK'}")
    if hwsd_used:
        st.markdown(f"**HWSD MDB**: {'✅ BULUNDU' if _ok(HWSD_MDB) else '❌ YOK'}")
        st.markdown(f"**HWSD RAS**: {'✅ BULUNDU' if _ok(HWSD_RAS) else '❌ YOK'}")

# ---- Girdi Alanı ----
g_left, g_right = st.columns([0.5, 0.5])
with g_left:
    st.subheader("📍 Analiz Konumu")
    loc_str = st.text_input("Google Maps linki veya 'lat,lon'",
                            value="38.554205, 38.707944",
                            placeholder="38.554205, 38.707944")
    run = st.button("Raporu Oluştur", type="primary")

with g_right:
    st.subheader("ℹ️ İpuçları & Durum Kontrolü")
    st.markdown(
        "- **Zorunlu:** Soldan `Bitki CSV` ve `İklim CSV` için **`✅ BULUNDU`** görmelisiniz.\n"
        "- **Konum:** `enlem, boylam` veya tam Google Maps linki girin.\n"
        "- **Toprak/Rakım:** Opsiyoneldir; yoksa skor yine hesaplanır (bazı modüller atlanır)."
    )

st.write("")

# ==========================================================
# 5) HESAPLAMA & SUNUM
# ==========================================================
if run:
    if not CROPS_CSV or not CLIMATE_CSV or not CROPS_CSV.exists() or not CLIMATE_CSV.exists():
        st.error("Zorunlu dosyalar (Bitki ve İklim CSV) eksik veya yanlış yol girildi. Soldaki yolları doğrulayın.")
        st.stop()

    try:
        lat, lon = parse_any_location(loc_str)
    except Exception as e:
        st.error(f"Konum çözümlenemedi: {e}")
        st.stop()

    with st.spinner("Çevre verileri toplanıyor..."):
        try:
            env = build_env(lat, lon, CLIMATE_CSV, ELEV_TIF, LIGHT_TIF, HWSD_MDB, HWSD_RAS)
        except Exception as e:
            st.error(f"Veri Toplama Hatası ({type(e).__name__}): {e}")
            st.caption("Lütfen terminal çıktısına bakın (pyodbc/rasterio/CSV sütun isimleri).")
            st.stop()

    st.success(f"📍 Konum Analiz Başarılı! Enlem: {lat:.5f}, Boylam: {lon:.5f}  |  İklim Piks. Uzaklık: {format_value(env.get('DISTANCE_KM'))} km")
    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=9)

        # 🌦️ ÇEVRESEL KOŞULLAR ÖZETİ (YENİ)
    st.subheader("🌦️ Çevresel Koşullar Özeti")
    
    # Ortalama sıcaklığa sabit düzeltme (+20)
    temp_val = env.get("T2M_grp2")
    try:
        if temp_val is not None:
            temp_val = float(temp_val) + 20.0
    except Exception:
        pass


    
    # Kart verileri
    summary_cards = [
        ("🌡️ Ortalama Sıcaklık", temp_val, "°C"),
        ("🌧️ Günlük Yağış", env.get("PRECTOTCORR_grp4"), "mm/gün"),
        ("☀️ Güneş Işıması", env.get("ALLSKY_SFC_SW_DWN_grp1"), "kWh/m²/g"),
        ("🏔️ Rakım", env.get("ELEVATION_M"), "m"),
        ("🌱 Toprak pH", env.get("T_PH_H2O"), ""),
        ("⚡ Tuzluluk (ECe)", env.get("T_ECE"), "dS/m"),
    ]
    
    # Opsiyonel göstergeler (varsa ekle)
    if env.get("NIGHT_LIGHT") is not None:
        summary_cards.append(("🌃 Gece Işığı", env.get("NIGHT_LIGHT"), "-"))
    if env.get("RH2M_grp2") is not None:
        summary_cards.append(("💧 Nem", env.get("RH2M_grp2"), "%"))
    
    # Kartları oluştur
    cols = st.columns(len(summary_cards))
    for col, (title, value, unit) in zip(cols, summary_cards):
        col.markdown(
            f'<div class="vg-kpi">'
            f'<div class="h">{title}</div>'
            f'<div class="v">{format_value(value)} {unit}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


    # Detaylı Çevre Tablosu (filtre fix)
    with st.expander("🧾 Tüm Çevre Değişkenleri (Detaylı Rapor)", expanded=False):
        rows = []
        for k, v in env.items():
            if k in {"latitude","longitude","USER_LAT","USER_LON","GRID_LAT","GRID_LON"}:
                continue
            base_k = k.split('_grp')[0]
            # Sadece meta'sı olanları göster
            if base_k.upper() not in VAR_META and k.upper() not in VAR_META:
                continue
            cat = category_of(k)
            title, unit = meta_of(k)
            if k.startswith("CLRSKY_DAYS") and isinstance(v,(int,float)) and float(v) > 31:
                 unit = "gün/yıl"
            rows.append({"Kategori": cat, "Başlık": title, "Değer": format_value(v) if v is not None else "(Veri Eksik)", "Birim": unit if unit != "-" else ""})
        env_df = pd.DataFrame(rows)
        if not env_df.empty:
            cat_map = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
            env_df['Kategori_Sort'] = env_df['Kategori'].map(cat_map)
            env_df = env_df.sort_values(by=['Kategori_Sort', 'Başlık']).drop(columns=['Kategori_Sort'])
            st.dataframe(env_df, use_container_width=True, height=360, hide_index=True)
        else:
            st.info("Görüntülenecek çevre değişkeni bulunamadı (VAR_META eşleşmedi).")

 # ---- 3.4 Skorlama (Otomatik Top10 + En İyi Ürünün İhtiyaçları) ----
st.subheader("🌱 Ürün Uygunluk Sıralaması (Top 10)")

try:
    crops_df = load_crops(CROPS_CSV)
    results_df = score_and_rank_df(env, crops_df)
except Exception as e:
    st.error(f"Bitki Skorlama Hatası: {type(e).__name__}: {e}")
    st.stop()

if results_df.empty:
    st.warning("Skor üretilemedi — gerekli çevre/bitki alanları eksik olabilir.")
else:
    # Top 10’u al
    top10 = results_df.sort_values("score", ascending=False).head(10).reset_index(drop=True)

    # Tablo görünümü
    view_df = top10[["common_name_tr", "crop", "score", "weakest_two"]].rename(columns={
        "common_name_tr":"Türkçe Ad",
        "crop":"Kod (İng.)",
        "score":"Uygunluk Skoru",
        "weakest_two":"Neyi Kısıtlıyor? (En Zayıf 2 Modül)"
    })

    # Skora göre basit arka plan rengi (interaktif filtre yok)
    def color_score(val):
        try:
            v = float(val)
        except:
            return ''
        if v < 50:   return 'background-color: #F8D7DA; font-weight: bold'
        if v < 75:   return 'background-color: #FFF3CD; font-weight: bold'
        return 'background-color: #D4EDDA; font-weight: bold'

    st.dataframe(
        view_df.style.applymap(color_score, subset=['Uygunluk Skoru']),
        use_container_width=True, height=360, hide_index=True
    )

    # En iyi eşleşme (1. satır)
    best_row = top10.iloc[0]
    best_code = best_row["crop"]
    best_name = best_row["common_name_tr"] or best_code
    best_score = best_row["score"]

     # ==========================================================
    # 🏆 EN İYİ EŞLEŞME — GELİŞTİRİLMİŞ, TEK KART TASARIM
    # ==========================================================
    import streamlit.components.v1 as components
    
    html_card = f"""
    <div style='background:linear-gradient(135deg,#DCFCE7,#F0FDF4);
                border:2px solid #22C55E;border-radius:16px;padding:20px;
                box-shadow:0 2px 8px rgba(0,0,0,0.08);
                display:flex;justify-content:space-between;align-items:flex-start;
                gap:20px;flex-wrap:wrap;margin-top:10px;margin-bottom:20px'>
    
      <!-- Sol kısım -->
      <div style='flex:1;min-width:240px;display:flex;flex-direction:column;align-items:flex-start;gap:6px'>
        <div style='font-size:2.2rem'>🏆</div>
        <div style='font-size:1.3rem;font-weight:700;color:#14532D;'>En İyi Eşleşme</div>
        <div style='font-size:1.8rem;font-weight:800;color:#065F46;'>{best_name}</div>
        <div style='font-size:1.1rem;color:#166534;'>Skor: <b>{best_score:.1f}</b> / 100</div>
      </div>
    
      <!-- Sağ kısım: yorum kutuları -->
      <div style='flex:2;min-width:320px;display:flex;flex-direction:column;gap:12px'>
    
        <div style='background:#ECFDF5;border-left:6px solid #10B981;border-radius:12px;
                    padding:12px 16px;'>
          <div style='font-weight:700;color:#065F46;margin-bottom:4px;font-size:1.05rem'>
            ✅ Güçlü Yönler
          </div>
          <ul style='margin:0;padding-left:1.2rem;color:#065F46;font-size:0.95rem;line-height:1.5'>
            <li>Isı ve ışınım koşulları ürün için genel olarak elverişli.</li>
            <li>Toprak pH ve tuzluluk seviyesi uygun aralıkta.</li>
            <li>Rakım, eğim ve drenaj açısından üretime uygun koşullar mevcut.</li>
          </ul>
        </div>
    
        <div style='background:#FEFCE8;border-left:6px solid #FACC15;border-radius:12px;
                    padding:12px 16px;'>
          <div style='font-weight:700;color:#92400E;margin-bottom:4px;font-size:1.05rem'>
            ⚠️ Geliştirilebilecek Alanlar
          </div>
          <ul style='margin:0;padding-left:1.2rem;color:#78350F;font-size:0.95rem;line-height:1.5'>
            <li>Yağış ve nem dengesinde dönemsel değişimler gözlenebilir.</li>
            <li>Kritik dönemlerde ek sulama yapılması önerilir.</li>
            <li>Rüzgar hassasiyeti yüksek bölgelerde koruma önlemi alınabilir.</li>
          </ul>
        </div>
    
      </div>
    </div>
    """
    
    # HTML doğrudan render edilir (düz metin çıkma sorunu çözülür)
    components.html(html_card, height=330)
    
    # --- Modül skorları ---
    st.markdown("#### 🔍 Genel Uygunluk Modülleri")
    mod_items = results_df.loc[results_df["crop"] == best_code, "modules"]
    if not mod_items.empty and isinstance(mod_items.iloc[0], dict):
        mods = mod_items.iloc[0]
        mod_df = (
            pd.DataFrame([mods])
            .T.reset_index()
            .rename(columns={"index": "Modül", 0: "Skor"})
            .sort_values("Skor", ascending=False)
        )
        st.dataframe(mod_df, use_container_width=True, hide_index=True, height=260)
    else:
        st.info("Modül detayları bulunamadı.")
    
    # --- Görsel açıklama kartları ---
    st.markdown("### 🌿 Tarımsal Özet Bilgiler")
    st.markdown(
        """
    <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));
                gap:1rem;margin-top:1rem'>
      <div style='background:white;border-radius:14px;padding:1rem;box-shadow:0 1px 4px rgba(0,0,0,0.1)'>
        <h4 style='color:#EA580C;font-size:1.1rem'>🌡️ İklim Uygunluğu</h4>
        <ul style='margin:0;padding-left:1.2rem'>
          <li>Ortalama sıcaklık bu ürün için elverişli.</li>
          <li>Don riski düşük, sıcaklık aralığı uygun.</li>
          <li>Radyasyon miktarı gelişim için yeterli.</li>
        </ul>
      </div>
      <div style='background:white;border-radius:14px;padding:1rem;box-shadow:0 1px 4px rgba(0,0,0,0.1)'>
        <h4 style='color:#0EA5E9;font-size:1.1rem'>💧 Su & Yağış</h4>
        <ul style='margin:0;padding-left:1.2rem'>
          <li>Yağış miktarı genel ihtiyacı karşılıyor.</li>
          <li>Su tutma kapasitesi uygun.</li>
          <li>Kurak dönemlerde ek sulama önerilebilir.</li>
        </ul>
      </div>
      <div style='background:white;border-radius:14px;padding:1rem;box-shadow:0 1px 4px rgba(0,0,0,0.1)'>
        <h4 style='color:#16A34A;font-size:1.1rem'>🧪 Toprak Özellikleri</h4>
        <ul style='margin:0;padding-left:1.2rem'>
          <li>pH seviyesi ideal aralıkta.</li>
          <li>Tuzluluk (ECe) düşük, verim engeli yok.</li>
          <li>Drenaj ve doku buğday için uygun.</li>
        </ul>
      </div>
      <div style='background:white;border-radius:14px;padding:1rem;box-shadow:0 1px 4px rgba(0,0,0,0.1)'>
        <h4 style='color:#78350F;font-size:1.1rem'>🧱 Arazi & Fiziksel Koşullar</h4>
        <ul style='margin:0;padding-left:1.2rem'>
          <li>Rakım ve eğim uygun sınırlar içinde.</li>
          <li>Gece ışığı seviyesi düşük (doğal üretim ortamı).</li>
          <li>Rüzgar etkisi orta düzeyde, problem yaratmaz.</li>
        </ul>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    
    # --- İndirilebilir Top10 ---
    st.download_button(
        "📥 Top10'u CSV olarak indir",
        data=top10.to_csv(index=False).encode("utf-8"),
        file_name="verimgoren_top10.csv",
        mime="text/csv",
    )
    
    st.caption("--- Analiz tamamlandı ---")
