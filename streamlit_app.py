import streamlit as st
import pandas as pd
from io import BytesIO
from streamlit_option_menu import option_menu
import numpy as np
from functools import partial
from datetime import datetime, timedelta, date

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_icon="🎯", page_title="PricePilot", layout="wide")

# ---------------------------
# Data laden
# ---------------------------
@st.cache_data(show_spinner=False)
def load_articles_df() -> pd.DataFrame:
    from articles import articles  # laadt ALLE artikelen uit jouw articles.py
    df = pd.DataFrame(articles)
    # Normaliseer kolomnamen indien nodig
    if "Material" not in df.columns or "Description" not in df.columns:
        raise ValueError("articles.py mist verplichte velden: Material, Description")
    # Zorg voor types
    df["Material"] = pd.to_numeric(df["Material"], errors="coerce").astype("Int64")
    if "mm" in df.columns:
        df["mm"] = pd.to_numeric(df["mm"], errors="coerce").fillna(0.0)
    else:
        df["mm"] = 0.0
    if "SML" not in df.columns:
        df["SML"] = "L"
    return df

@st.cache_data(show_spinner=False)
def load_sap_prices() -> dict:
    try:
        from SAPprijs import sap_prices
        return sap_prices
    except Exception:
        # Fallback als SAPprijs.py (nog) niet aanwezig is
        return {
            "100007": {},  # lege mapping; kan later gevuld worden
        }

if "accounts_df" not in locals():
    # Simulatie of fallback als het nog niet geladen is
    accounts_df = pd.DataFrame({
        "Klantnummer": [100001, 100002, 100003],
        "Omzet klant (€)": [1_200_000, 600_000, 150_000],
    })
        
# 1) Matrixen toevoegen (eenmalig bovenin je script)
BASE_PRICE_ALFA_BY_CLASS = {'A': 32, 'B': 34, 'C': 36, 'D': 37}
PER_MM_UPLIFT_BY_CLASS   = {'A': 2.0, 'B': 2.5, 'C': 2.5, 'D': 2.75}

per_pg_uplift = {
    "IsoPerform SS Zero (HR++)": 12,
    "SolarControl SKN 154 (ZHR++)": 25,
    "SolarControl SKN165 (ZHR++)": 25,
    "IsoPerform Eclaz Zen (HR++)": 6,
    "IP Energy 72/38 (ZHR++)": 6,
    "IsoPerform ALFA (HR++)": 0,
}

DEFAULT_CUSTOMER_ID = "100007"

def get_sap_price(material_id, customer_id, prices=None):
    """
    Haalt prijs op voor (customer_id, material_id) met fallback naar DEFAULT_CUSTOMER_ID.
    Geeft None terug als nergens gevonden.
    """
    prices = prices or sap_prices_all
    m = str(material_id)
    c = str(customer_id)

    # 1) klant-specifiek
    if c in prices and m in prices[c]:
        return prices[c][m]

    # 2) fallback naar default klant
    default = prices.get(DEFAULT_CUSTOMER_ID, {})
    return default.get(m)

# ---------------------------
# Helper functies
# ---------------------------
def sml_filter(df: pd.DataFrame, pick: str) -> pd.DataFrame:
    if pick == "S":
        return df[df["SML"] == "S"]
    elif pick == "M":
        return df[df["SML"].isin(["S", "M"])]
    return df  # L = alles




def prijs_kwaliteit(final_price, min_p, max_p) -> str:
    try:
        if pd.isna(final_price):
            return ""
        if pd.notna(min_p) and final_price < float(min_p):
            return "Onder min"
        if pd.notna(max_p) and final_price > float(max_p):
            return "Boven max"
        return "Binnen band"
    except Exception:
        return ""

def bepaal_omzet(klantnummer: str) -> float:
    if klantnummer.startswith("10"):
        return 76000
    elif klantnummer.startswith("11"):
        return 200000
    elif klantnummer.startswith("12"):
        return 300000
    elif klantnummer.startswith(("13", "14", "15", "16")):
        return 700000
    else:
        return 50000  # default bij onbekende prefix

def bepaal_klantgrootte(omzet: float) -> str:
    if omzet > 500000:
        return "A"
    elif omzet > 250000:
        return "B"
    elif omzet > 100000:
        return "C"
    else:
        return "D"
        
accounts_df["Klantgrootte"] = accounts_df["Omzet klant (€)"].apply(bepaal_klantgrootte)

# Defaults
# Defaults voor tonen en coating-opslag per productgroep
ALWAYS_ON = {
    "IsoPerform ALFA (HR++)",
    #"IsoPerform Eclaz Zen (HR++)",
    #"IsoPerform SS Zero (HR++)",
    #"TriplePerform 74/54 (HR++)",
}
DEFAULT_UPLIFTS = {
    "IsoPerform SS Zero (HR++)": 12,
    "SolarControl SKN 154 (ZHR++)": 25,
    "SolarControl SKN165 (ZHR++)": 25,
    "IsoPerform Eclaz Zen (HR++)": 6,
    "IP Energy 72/38 (ZHR++)": 6,
    "IsoPerform ALFA (HR++)": 0,
    # overige niet genoemde productgroepen krijgen 30
}

def df_to_simple_pdf(df: pd.DataFrame, title: str = "Prijslijst") -> bytes:
    """Zeer eenvoudige PDF-export (optioneel). Vereist reportlab."""
    try:
        from io import BytesIO
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=landscape(A4))
        width, height = landscape(A4)

        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, height - 1.5*cm, title)

        cols = ["Material","Description","Productgroep","mm","Huidige m2 prijs","RSP","Handmatige prijs","Final prijs","Omzet conditie","Omzet totaal","Effect aanpassing"]
        show_df = df.reindex(columns=[c for c in cols if c in df.columns]).copy()

        # Willekeurige omzet totaal tussen €0 en €2000
        show_df["Omzet totaal"] = np.random.uniform(0, 2000, size=len(show_df)).round(2)
        
        # Willekeurige omzet conditie tussen 0% en 100% van omzet totaal
        show_df["Omzet conditie"] = (show_df["Omzet totaal"] * np.random.uniform(0, 1, size=len(show_df))).round(2)
        
        # Effect aanpassing = Omzet conditie * (Handmatige prijs / Huidige m2 prijs)
        show_df["Effect aanpassing"] = (
            show_df["Omzet conditie"]
            * (
                pd.to_numeric(show_df["Handmatige prijs"], errors="coerce")
                / pd.to_numeric(show_df["Huidige m2 prijs"], errors="coerce")
            )
        ).round(2)

        x0, y0 = 1.2*cm, height - 3*cm
        line_h = 0.6*cm
        col_widths = [3*cm, 6.2*cm, 3.2*cm, 1.6*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm]

        def draw_row(y, values, bold=False):
            c.setFont("Helvetica-Bold" if bold else "Helvetica", 8)
            x = x0
            for v, w in zip(values, col_widths):
                c.drawString(x, y, str(v))
                x += w

        # Header
        draw_row(y0, cols[:len(col_widths)], bold=True)
        y = y0 - line_h

        for _, row in show_df.iterrows():
            vals = [row.get(col, "") for col in cols]
            draw_row(y, vals[:len(col_widths)], bold=False)
            y -= line_h
            if y < 2*cm:
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.drawString(2*cm, height - 1.5*cm, title)
                draw_row(y0, cols[:len(col_widths)], bold=True)
                y = y0 - line_h

        c.save()
        buffer.seek(0)
        return buffer.read()
    except Exception:
        return b""

# ---------------------------
# Session init
# ---------------------------
if "articles_df" not in st.session_state:
    st.session_state.articles_df = load_articles_df()
if "sap_prices" not in st.session_state:
    st.session_state.sap_prices = load_sap_prices()

df_all = st.session_state.articles_df.copy()
sap_prices_all = st.session_state.sap_prices

# ---------------------------
# Sidebar navigatie (option_menu)
# ---------------------------
with st.sidebar:
    st.markdown("### 🎯 PricePilot")
    selected = option_menu(
        menu_title=None,
        options=["Prijslijst", "Beheer"],
        icons=["list-ul", "tools"],
        default_index=0,
        orientation="vertical",
    )

    st.markdown("---")
    st.caption("Selecties")
    
    # Klantselectie via Salesforce (fallback = sap_prices_all)
    from simple_salesforce import Salesforce, SalesforceLogin
    import os
    import pandas as pd
    
    # SF login (zelfde patroon als je andere app)
    SF_USERNAME = os.getenv("SALESFORCE_USERNAME")
    SF_PASSWORD = os.getenv("SALESFORCE_PASSWORD")
    SF_SECURITY_TOKEN = os.getenv("SF_SECURITY_TOKEN")
    SF_DOMAIN = "test"  # 'test' = Sandbox

    # Altijd een DataFrame hebben (voorkomt NameError)
    accounts_df = pd.DataFrame(columns=["Klantnaam", "Klantnummer", "Klantinfo", "Omzet klant (€)", "Klantgrootte"])
    
    def fetch_salesforce_accounts_direct(sf_connection):
        try:
            res = sf_connection.query("""
                SELECT Id, Name, ERP_Number__c
                FROM Account
                WHERE ERP_Number__c != NULL AND Is_Active__c = TRUE
                ORDER BY ERP_Number__c ASC
                LIMIT 7000
            """)
            return res["records"]
        except Exception as e:
            st.warning(f"Fout bij ophalen van Salesforce-accounts: {e}")
            return []
    
    # Verbind met Salesforce + opbouwen accounts_df
    try:
        sf = None
        if SF_USERNAME and SF_PASSWORD and SF_SECURITY_TOKEN:
            session_id, instance = SalesforceLogin(
                username=SF_USERNAME,
                password=f"{SF_PASSWORD}{SF_SECURITY_TOKEN}",
                domain=SF_DOMAIN
            )
            sf = Salesforce(instance=instance, session_id=session_id)
            records = fetch_salesforce_accounts_direct(sf)
            if records:
                tmp_df = pd.DataFrame(records).drop(columns="attributes", errors="ignore")
                # Controleer verplichte kolommen
                if {"Name", "ERP_Number__c"}.issubset(tmp_df.columns):
                    accounts_df = tmp_df.rename(columns={"Name": "Klantnaam", "ERP_Number__c": "Klantnummer"})
                    accounts_df["Klantnummer"] = accounts_df["Klantnummer"].astype(str)

                    # Omzet en klantgrootte
                    def bepaal_omzet(klantnummer: str) -> float:
                        if klantnummer.startswith("10"):
                            return 76000
                        elif klantnummer.startswith("11"):
                            return 200000
                        elif klantnummer.startswith("12"):
                            return 300000
                        elif klantnummer.startswith(("13", "14", "15", "16")):
                            return 700000
                        else:
                            return 50000

                    def bepaal_klantgrootte(omzet: float) -> str:
                        if omzet > 500000:
                            return "A"
                        elif omzet > 250000:
                            return "B"
                        elif omzet > 100000:
                            return "C"
                        else:
                            return "D"

                    accounts_df["Omzet klant (€)"] = accounts_df["Klantnummer"].apply(bepaal_omzet)
                    accounts_df["Klantgrootte"] = accounts_df["Omzet klant (€)"].apply(bepaal_klantgrootte)
                    accounts_df["Klantinfo"] = accounts_df["Klantnummer"] + " - " + accounts_df["Klantnaam"]
                else:
                    st.warning(f"Onverwachte Salesforce-kolommen: {list(tmp_df.columns)}")
    except Exception as e:
        st.warning(f"Fout bij het verbinden met Salesforce: {e}")
    
    # UI: kies klant (Salesforce → klantnummer), anders fallback
    if not accounts_df.empty:
        gekozen_info = st.selectbox("Klant", accounts_df["Klantinfo"].tolist(), index=0)
        klant = accounts_df.loc[accounts_df["Klantinfo"] == gekozen_info, "Klantnummer"].iloc[0]
    else:
        klant_opts = list(sap_prices_all.keys()) if sap_prices_all else ["100007"]
        klant = st.selectbox("Klantnummer (fallback)", klant_opts, index=0)

    # Metrics (alleen als SF-data beschikbaar is)
    if not accounts_df.empty:
        gekozen_klant = accounts_df.loc[accounts_df["Klantnummer"] == str(klant)].iloc[0]
        col1, col2 = st.columns(2)
        col1.metric("💶 Omzet klant", f"€ {gekozen_klant['Omzet klant (€)']:,.0f}".replace(",", "."))
        col2.metric("🏷️ Klantgrootte", gekozen_klant["Klantgrootte"])

    # Klantgrootte ophalen voor default prijsparameters
    klantgrootte = None
    if not accounts_df.empty:
        sel = accounts_df.loc[accounts_df["Klantnummer"] == str(klant)]
        if not sel.empty:
            klantgrootte = sel["Klantgrootte"].iloc[0]


    
    # Defaults per klantgrootte
    BASE_PRICE_BY_SIZE = {"A": 30, "B": 32, "C": 34, "D": 37}
    PER_MM_BY_SIZE     = {"A": 2.00, "B": 2.50, "C": 2.50, "D": 2.75}
    
    base_default = float(BASE_PRICE_BY_SIZE.get(klantgrootte, 30))
    permm_default = float(PER_MM_BY_SIZE.get(klantgrootte, 2.50))

    # # --- Productgroepen + S/M/L + coating-opslag in expander ---
    # allowed_pg = [
    #     "IsoPerform SS Zero (HR++)",
    #     "SolarControl SKN 154 (ZHR++)",
    #     "SolarControl SKN165 (ZHR++)",
    #     "IsoPerform Eclaz Zen (HR++)",
    #     "IP Energy 72/38 (ZHR++)",
    #     "IsoPerform ALFA (HR++)"
    # ]
    
    # alle_pg = [
    #     pg for pg in sorted(df_all["Productgroep"].dropna().unique().tolist())
    #     if pg in allowed_pg
    # ] if "Productgroep" in df_all.columns else []

    # --- Productgroepen + S/M/L + coating-opslag in expander ---
    allowed_pg_order = [
        "IsoPerform ALFA (HR++)",
        "IsoPerform Eclaz Zen (HR++)",
        "IsoPerform SS Zero (HR++)",
        "IP Energy 72/38 (ZHR++)",
        "SolarControl SKN 154 (ZHR++)",
        "SolarControl SKN165 (ZHR++)"
    ]
    
    if "Productgroep" in df_all.columns:
        unieke_pg = df_all["Productgroep"].dropna().unique().tolist()
        alle_pg = [pg for pg in allowed_pg_order if pg in unieke_pg]
    else:
        alle_pg = []
    
    # Init session state (éénmalig of bij nieuwe pg's)
    if "pg_show" not in st.session_state:
        st.session_state.pg_show = {pg: ("ja" if pg in ALWAYS_ON else "nee") for pg in alle_pg}
    if "pg_uplift" not in st.session_state or set(st.session_state.pg_uplift.keys()) != set(alle_pg):
        st.session_state.pg_uplift = {pg: float(DEFAULT_UPLIFTS.get(pg, 30)) for pg in alle_pg}
    
    with st.expander("Productgroepen & coating-toeslagen", expanded=False):
        st.caption("Zet per productgroep aan/uit en geef een opslag in €/m² op.")
        # kopregel
        header_cols = st.columns([4, 2, 3])
        header_cols[0].markdown("**Productgroep**")
        header_cols[1].markdown("**Tonen?**")
        header_cols[2].markdown("**Opslag coating (€/m²)**")

        for pg in alle_pg:
            c1, c2, c3 = st.columns([4, 2, 3])
            c1.write(pg)
            st.session_state.pg_show[pg] = c2.selectbox(
                label="",
                options=["ja", "nee"],
                index=0 if st.session_state.pg_show.get(pg, "nee") == "ja" else 1,
                key=f"pg_show_{pg}",
            )
            # toon & update opslag per productgroep
            current = float(st.session_state.pg_uplift.get(pg, 30.0))
            new_val = c3.number_input(
                label="",
                value=current,
                step=0.5,
                min_value=0.0,
                key=f"pg_uplift_{pg}",
            )
            st.session_state.pg_uplift[pg] = new_val

    # resultaatvariabelen voor gebruik in de rest van de app (ongewijzigde namen)
    sel_pg = [pg for pg, v in st.session_state.pg_show.items() if v == "ja"]
    sml_pick = st.radio(
        "S/M/L lijst",
        ["S", "M", "L"],
        index=1,  # standaard op "M"
        horizontal=True,
        help="S = hardlopers, M = S+M, L = alle artikelen"
    )

    col1, col2, col3 = st.columns([1, 0.2, 1])
    
    with col1:
        st.markdown("<div style='text-align: right; font-weight: bold;'>Build Up</div>", unsafe_allow_html=True)
    
    with col2:
        rsp_build_up = st.toggle(
            "",
            value=False,
            key="rsp_build_up"
        )
    
    with col3:
        st.markdown("<div style='text-align: left; font-weight: bold;'>RSP</div>", unsafe_allow_html=True)

    
    if not rsp_build_up:
        st.markdown("---")
        st.caption("Prijsparameters")
        base_price_alfa = st.number_input(
            "Basismateriaal (IsoPerform ALFA 04 - #04)",
            min_value=0.0, value=base_default, step=0.1,
        )
        per_mm_uplift = st.number_input(
            "Opslag per mm (€/mm)",
            min_value=0.0, value=permm_default, step=0.05
        )
        gelaagd_component = st.number_input(
            "Opslag per gelaagd component",
            min_value=0.0, value=20.0, step=0.5
        )
        
        # dict met opslag per productgroep voor RSP-berekening
        per_pg_uplift = dict(st.session_state.pg_uplift)

    else:
        st.markdown("---")
        st.caption("RSP-index")
        rsp_pct = st.number_input(
            "RSP-aanpassing (%)", min_value=50, max_value=150, value=100, step=1,
            help="Final price = RSP × dit percentage"
        )
        
    st.markdown("---")
    datum = datetime.now().strftime("%Y-%m-%d")
    export_name = f"Prijslijst_{gekozen_info}_{datum}"
    st.text_input("Bestandsnaam export (zonder extensie)", value=export_name)

# ============================================================
# HANDMATIGE PRIJS / FINAL PRIJS – OPSCHOONDE EN JUISTE VERSIE
# ============================================================

# --- Functies voor berekeningen ---
def compute_rsp(row, base_price_alfa: float, per_pg_uplift: dict, per_mm_uplift: float) -> float:
    pg = row.get("Productgroep", "")
    mm = float(row.get("mm", 0.0) or 0.0)
    pg_uplift = float(per_pg_uplift.get(pg, 0.0) or 0.0)
    return float(base_price_alfa) + pg_uplift + ((mm - 8) * float(per_mm_uplift))

def compute_rsp_with_matrix(r, accounts_df):
    klantnummer = str(r.get("Klantnummer", "") or "")
    if klantnummer and not accounts_df.empty and "Klantnummer" in accounts_df.columns:
        sel = accounts_df[accounts_df["Klantnummer"].astype(str) == klantnummer]
    else:
        sel = pd.DataFrame()

    if not sel.empty and "Klantgrootte" in sel.columns:
        klass = str(sel.iloc[0]["Klantgrootte"]).upper()
    else:
        omzet = bepaal_omzet(klantnummer) if klantnummer else 0
        klass = bepaal_klantgrootte(omzet)

    base_price_alfa = BASE_PRICE_ALFA_BY_CLASS.get(klass, 36)
    per_mm_uplift   = PER_MM_UPLIFT_BY_CLASS.get(klass, 2.5)
    return compute_rsp(r, base_price_alfa, per_pg_uplift, per_mm_uplift)

def _compute_handmatig_core(row, base_price_alfa, per_pg_uplift, per_mm_uplift, gelaagd_component):
    """Gebruikt in Build Up-modus voor berekening Handmatige prijs"""
    pg = row.get("Productgroep", "")
    mm = float(row.get("mm", 0.0) or 0.0)
    pg_uplift_val = float(per_pg_uplift.get(pg, 0.0) or 0.0)
    layers = str(row.get("Artikelnaam", "")).count(".")
    return float(base_price_alfa) + pg_uplift_val + ((mm - 8) * float(per_mm_uplift)) + (layers * float(gelaagd_component))


# ============================================================
# Pagina: Prijslijst
# ============================================================
if selected == "Prijslijst":
    st.title("📄 Prijslijst")

    # Filter op productgroep + S/M/L
    df = df_all.copy()
    if sel_pg:
        df = df[df["Productgroep"].isin(sel_pg)]
    df = sml_filter(df, sml_pick).copy()

    # Basis kolommen
    df["Material"] = df["Material"].astype("Int64")
    df["Artikelnummer"] = df["Material"].astype(str)
    df["Artikelnaam"] = df["Description"]
    df["Klantnummer"] = str(klant)
    if "mm" not in df.columns:
        df["mm"] = 0.0
    df["mm"] = pd.to_numeric(df["mm"], errors="coerce").fillna(0.0)

    # Huidige m2 prijs (SAP) met fallback naar default klant 100007
    sap_for_client = sap_prices_all.get(klant, {})
    sap_default = sap_prices_all.get("100007", {})
    
    # Zorg dat gelaagd_component altijd bestaat
    if "gelaagd_component" not in locals():
        gelaagd_component = 0.0

    def map_sap_price(artnr: str):
        art = str(artnr)
        return sap_for_client.get(art) or sap_default.get(art)

    df["Huidige m2 prijs"] = df["Artikelnummer"].map(map_sap_price)

    # Bereken RSP
    df["RSP"] = df.apply(
        lambda r: round(
            compute_rsp_with_matrix(r, accounts_df)
            + (str(r.get("Artikelnaam", "")).count(".") * gelaagd_component),
            2,
        ),
        axis=1,
    )

    # --- Handmatige prijs berekenen afhankelijk van modus ---
    if "Handmatige prijs" not in df.columns:
        df["Handmatige prijs"] = np.nan

    if not rsp_build_up:
        # BUILD UP → bereken o.b.v. mm/coating/gelaagd component
        compute_handmatige_prijs = partial(
            _compute_handmatig_core,
            base_price_alfa=base_price_alfa,
            per_pg_uplift=per_pg_uplift,
            per_mm_uplift=per_mm_uplift,
            gelaagd_component=gelaagd_component,
        )
        df["Handmatige prijs_auto"] = df.apply(
            lambda r: round(compute_handmatige_prijs(r), 2),
            axis=1,
        )
    else:
        # RSP-modus → Handmatige prijs = RSP × percentage
        df["Handmatige prijs_auto"] = (
            pd.to_numeric(df["RSP"], errors="coerce") * (rsp_pct / 100.0)
        ).round(2)

    # Alleen lege/NaN cellen vullen (handmatige invoer behouden)
    mask_leeg = df["Handmatige prijs"].isna() | (df["Handmatige prijs"] == "")
    df.loc[mask_leeg, "Handmatige prijs"] = df.loc[mask_leeg, "Handmatige prijs_auto"]

    # Final prijs volgt ALTIJD de Handmatige prijs
    df["Final prijs"] = pd.to_numeric(df["Handmatige prijs"], errors="coerce").round(2)

  

    # Prijskwaliteit = (Final prijs / RSP) * 100, afgerond op hele getallen
    final_vals = pd.to_numeric(df["Huidige m2 prijs"], errors="coerce")
    rsp_vals   = pd.to_numeric(df["RSP"], errors="coerce")
    
    ratio_pct = np.where((~np.isnan(rsp_vals)) & (rsp_vals > 0),
                         (final_vals / rsp_vals) * 100,
                         np.nan)
    df["Prijskwaliteit"] = np.round(ratio_pct)

    # Nieuwe Prijskwaliteit = (Final prijs / RSP) * 100, afgerond op hele getallen
    new_final_vals = pd.to_numeric(df["Final prijs"], errors="coerce")
    rsp_vals   = pd.to_numeric(df["RSP"], errors="coerce")
    
    new_ratio_pct = np.where((~np.isnan(rsp_vals)) & (rsp_vals > 0),
                         (new_final_vals / rsp_vals) * 100,
                         np.nan)
    df["New Prijskwaliteit"] = np.round(new_ratio_pct)

    
    


    # Tabel tonen (selectief editable)
    show_cols = [
        "Artikelnummer","Artikelnaam","Productgroep","mm",
        "Huidige m2 prijs","RSP","Handmatige prijs","Final prijs",
        "Prijskwaliteit","New Prijskwaliteit","Omzet conditie","Omzet totaal","Effect aanpassing"
    ]
    display_df = df.reindex(columns=show_cols).copy()

    display_df = display_df.reset_index(drop=True)
    edited = st.data_editor(
        display_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Artikelnummer": st.column_config.TextColumn(disabled=True),
            "Artikelnaam": st.column_config.TextColumn(disabled=True),
            "Productgroep": st.column_config.TextColumn(disabled=True),
            "mm": st.column_config.NumberColumn(disabled=True),
            "Huidige m2 prijs": st.column_config.NumberColumn(width="small",disabled=True, format="€ %.2f",),
            "RSP": st.column_config.NumberColumn(width="small",disabled=True, format="€ %.2f",),
            "Handmatige prijs": st.column_config.NumberColumn(width="small",help="Laat leeg om RSP te gebruiken", format="€ %.2f",),
            "Final prijs": st.column_config.NumberColumn(width=20,disabled=True,format="€ %.2f",),
            "Prijskwaliteit": st.column_config.TextColumn(width="small",disabled=True),
            "New Prijskwaliteit": st.column_config.TextColumn(width="small",disabled=True),
            "Omzet conditie": st.column_config.NumberColumn(disabled=True),
            "Omzet totaal": st.column_config.NumberColumn(disabled=True),
            "Effect aanpassing": st.column_config.NumberColumn(disabled=True),
        },
        key="prijs_editor",
        column_order=[c for c in df.columns if c != "mm"],
    )

    # Herberekenen op basis van editor
    # Herberekenen op basis van editor
    edited["Handmatige prijs"] = pd.to_numeric(edited["Handmatige prijs"], errors="coerce")
    
    if rsp_build_up:
        # RSP-modus: waar geen handmatige override is, vullen met RSP × %
        auto_val = (pd.to_numeric(edited["RSP"], errors="coerce") * (rsp_pct / 100.0)).round(2)
        mask_leeg = edited["Handmatige prijs"].isna()
        edited.loc[mask_leeg, "Handmatige prijs"] = auto_val[mask_leeg]
    
    # ✅ Final price = Handmatige prijs
    edited["Final prijs"] = edited["Handmatige prijs"].round(2)


    st.caption(f"Regels: {len(edited)}")

    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = edited.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name=f"{export_name}.csv", mime="text/csv")
    with c2:
        pdf_bytes = df_to_simple_pdf(edited, title=f"Prijslijst klant {klant}")
        if pdf_bytes:
            st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name=f"{export_name}.pdf", mime="application/pdf")
        else:
            st.info("PDF-export vereist `reportlab` → `pip install reportlab`")

# ---------------------------
# Pagina: Beheer
# ---------------------------
elif selected == "Beheer":
    st.title("🛠️ Beheer – artikelen")

    beheer_cols = ["Material","Description","Productgroep","mm","Min_prijs","Max_prijs","SML","UC_waarde","Min_afname"]
    base = df_all.reindex(columns=[c for c in beheer_cols if c in df_all.columns]).copy()

    beheer_edit = st.data_editor(
        base,
        use_container_width=True,
        column_config={
            "Material": st.column_config.NumberColumn(disabled=True),
            "Description": st.column_config.TextColumn(disabled=True),
            "Productgroep": st.column_config.TextColumn(disabled=True),
            "mm": st.column_config.NumberColumn(disabled=True),
            "Min_prijs": st.column_config.NumberColumn(help="Ondergrens per m²"),
            "Max_prijs": st.column_config.NumberColumn(help="Bovengrens per m²"),
            "SML": st.column_config.SelectboxColumn(options=["S","M","L"], help="S=hardloper, M=mid, L=volledige lijst"),
            "UC_waarde": st.column_config.TextColumn(disabled=True) if "UC_waarde" in base.columns else None,
            "Min_afname": st.column_config.NumberColumn(disabled=True) if "Min_afname" in base.columns else None,
        },
        key="beheer_editor",
    )

    st.caption("Alleen Min_prijs, Max_prijs en SML zijn bewerkbaar. Overige velden zijn vergrendeld.")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("💾 Wijzigingen opslaan (session)"):
            full = st.session_state.articles_df.set_index("Material")
            upd = beheer_edit.set_index("Material")
            for col in ["Min_prijs","Max_prijs","SML"]:
                if col in upd.columns and col in full.columns:
                    full.loc[upd.index, col] = upd[col]
            st.session_state.articles_df = full.reset_index()
            st.success("Wijzigingen opgeslagen in geheugen.")
    with c2:
        export_articles_csv = st.session_state.articles_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporteer CSV", data=export_articles_csv, file_name="articles_updated.csv", mime="text/csv")
    with c3:
        export_articles_json = st.session_state.articles_df.to_json(orient="records", force_ascii=False).encode("utf-8")
        st.download_button("⬇️ Exporteer JSON", data=export_articles_json, file_name="articles_updated.json", mime="application/json")
