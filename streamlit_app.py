import streamlit as st
import pandas as pd
from io import BytesIO
from streamlit_option_menu import option_menu

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

# ---------------------------
# Helper functies
# ---------------------------
def sml_filter(df: pd.DataFrame, pick: str) -> pd.DataFrame:
    if pick == "S":
        return df[df["SML"] == "S"]
    elif pick == "M":
        return df[df["SML"].isin(["S", "M"])]
    return df  # L = alles

def compute_rsp(row, base_price_alfa: float, per_pg_uplift: dict, per_mm_uplift: float) -> float:
    pg = row.get("Productgroep", "")
    mm = float(row.get("mm", 0.0) or 0.0)
    pg_uplift = float(per_pg_uplift.get(pg, 0.0) or 0.0)
    return float(base_price_alfa) + pg_uplift + (mm * float(per_mm_uplift))

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

def df_to_simple_pdf(df: pd.DataFrame, title: str = "Prijslijst") -> bytes:
    """Zeer eenvoudige PDF-export (optioneel). Vereist reportlab."""
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=landscape(A4))
        width, height = landscape(A4)

        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, height - 1.5*cm, title)

        cols = ["Material","Description","Productgroep","mm","Huidige m2 prijs","RSP","Handmatige prijs","Final prijs","Verwachte m2","Omzet conditie","Omzet totaal","Effect aanpassing"]
        show_df = df.reindex(columns=[c for c in cols if c in df.columns]).copy()

        x0, y0 = 1.2*cm, height - 3*cm
        line_h = 0.6*cm
        col_widths = [3*cm, 6.2*cm, 3.2*cm, 1.6*cm, 3*cm, 2.4*cm, 3*cm, 2.6*cm, 2.6*cm, 3*cm, 3*cm, 3*cm]

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
    
    accounts_df["Klantnummer"] = accounts_df["Klantnummer"].astype(str)  # eerst naar string
    accounts_df["Omzet klant (€)"] = accounts_df["Klantnummer"].apply(bepaal_omzet)
    accounts_df["Klantgrootte"] = accounts_df["Omzet klant (€)"].apply(bepaal_klantgrootte)
    accounts_df["Klantinfo"] = accounts_df["Klantnummer"] + " - " + accounts_df["Klantnaam"]
    
    # Verbind met Salesforce
    sf = None
    try:
        if SF_USERNAME and SF_PASSWORD and SF_SECURITY_TOKEN:
            session_id, instance = SalesforceLogin(
                username=SF_USERNAME,
                password=f"{SF_PASSWORD}{SF_SECURITY_TOKEN}",
                domain=SF_DOMAIN
            )
            sf = Salesforce(instance=instance, session_id=session_id)
            records = fetch_salesforce_accounts_direct(sf)
            if records:
                accounts_df = (
                    pd.DataFrame(records)
                    .drop(columns="attributes", errors="ignore")
                    .rename(columns={"Name": "Klantnaam", "ERP_Number__c": "Klantnummer"})
                )
    
                if "Klantnummer" in accounts_df.columns:
                    accounts_df["Klantnummer"] = accounts_df["Klantnummer"].astype(str)
                    accounts_df["Omzet klant (€)"] = accounts_df["Klantnummer"].apply(bepaal_omzet)
                    accounts_df["Klantgrootte"] = accounts_df["Omzet klant (€)"].apply(bepaal_klantgrootte)
                    accounts_df["Klantinfo"] = accounts_df["Klantnummer"] + " - " + accounts_df["Klantnaam"]
                else:
                    st.warning("Geen geldige klantnummers gevonden in Salesforce-data.")
    except Exception as e:
        st.warning(f"Fout bij het verbinden met Salesforce: {e}")
    
    # UI: kies klant (Salesforce → klantnummer), anders fallback
    if not accounts_df.empty:
        gekozen_info = st.selectbox("Klant", accounts_df["Klantinfo"].tolist(), index=0)
        klant = accounts_df.loc[accounts_df["Klantinfo"] == gekozen_info, "Klantnummer"].iloc[0]
    else:
        klant_opts = list(sap_prices_all.keys()) if sap_prices_all else ["100007"]
        klant = st.selectbox("Klantnummer (fallback)", klant_opts, index=0)

    if not accounts_df.empty:
        gekozen_klant = accounts_df.loc[accounts_df["Klantnummer"] == klant].iloc[0]
        col1, col2 = st.columns(2)
        col1.metric("💶 Omzet klant", f"€ {gekozen_klant['Omzet klant (€)']:,.0f}".replace(",", "."))
        col2.metric("🏷️ Klantgrootte", gekozen_klant["Klantgrootte"])

    # Klantgrootte ophalen voor de geselecteerde klant
    klantgrootte = None
    if 'accounts_df' in locals() and not accounts_df.empty:
        sel = accounts_df.loc[accounts_df["Klantnummer"] == str(klant)]
        if not sel.empty:
            klantgrootte = sel["Klantgrootte"].iloc[0]
    
    # Defaults per klantgrootte
    BASE_PRICE_BY_SIZE = {"A": 30, "B": 32, "C": 34, "D": 37}
    PER_MM_BY_SIZE     = {"A": 2.00, "B": 2.50, "C": 2.50, "D": 2.75}
    
    base_default = float(BASE_PRICE_BY_SIZE.get(klantgrootte, 30))
    permm_default = float(PER_MM_BY_SIZE.get(klantgrootte, 2.50))


    # --- Productgroepen + S/M/L + coating-opslag in expander ---
    alle_pg = sorted(df_all["Productgroep"].dropna().unique().tolist()) if "Productgroep" in df_all.columns else []
    
    # init state voor keuzes per productgroep
    ALWAYS_ON = {
        "IsoPerform Alfa (HR++)",
        "IsoPerform Eclaz Zen (HR++)",
        "IsoPerform SS Zero (HR++)",
        "TriplePerform 74/54 (HR++)",
    }
    
    if "pg_show" not in st.session_state:
        st.session_state.pg_show = {
            pg: ("ja" if pg in ALWAYS_ON else "nee") 
            for pg in alle_pg
        }

    
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
                index=0 if st.session_state.pg_show.get(pg, "ja") == "ja" else 1,
                key=f"pg_show_{pg}",
            )
            DEFAULT_UPLIFTS = {
                "IP SolarControl Sun (ZHR++)": 6,
                "IsoPerform SS Zero (HR++)": 12,
                "SolarControl SKN 154 (ZHR++)": 25,
                "SolarControl SKN165 (ZHR++)": 25,
                "IsoPerform Eclaz Zen (HR++)": 6,
                "IP Energy 72/38 (ZHR++)": 6,
                # overige niet genoemde productgroepen krijgen 30
            }
            
            if "pg_uplift" not in st.session_state:
                st.session_state.pg_uplift = {
                    pg: float(DEFAULT_UPLIFTS.get(pg, 30)) 
                    for pg in alle_pg
                }

    # resultaatvariabelen voor gebruik in de rest van de app (ongewijzigde namen)
    sel_pg = [pg for pg, v in st.session_state.pg_show.items() if v == "ja"]
    sml_pick = st.radio("S/M/L lijst", ["S", "M", "L"], horizontal=True,
                        help="S = hardlopers, M = S+M, L = alle artikelen")
    
    st.markdown("---")
    st.caption("Prijsparameters")
    base_price_alfa = st.number_input(
        "Basismateriaal (IsoPerform ALFA 04 - #04)",
        min_value=0.0, value=base_default, step=0.1,
        help="Startpunt voor RSP; Alfa heeft geen coatingtoeslag"
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
    
    st.markdown("---")
    export_name = st.text_input("Bestandsnaam export (zonder extensie)", value="prijslijst")

# ---------------------------
# Pagina: Prijslijst
# ---------------------------
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
    if "mm" not in df.columns:
        df["mm"] = 0.0
    df["mm"] = pd.to_numeric(df["mm"], errors="coerce").fillna(0.0)

    # Huidige m2 prijs (SAP)
    sap_for_client = sap_prices_all.get(klant, {})
    def map_sap_price(artnr: str):
        # SAP dict keys vaak als strings
        return sap_for_client.get(str(artnr), None)
    df["Huidige m2 prijs"] = df["Artikelnummer"].map(map_sap_price)

    # RSP
    df["RSP"] = df.apply(lambda r: round(compute_rsp(r, base_price_alfa, per_pg_uplift, per_mm_uplift), 2), axis=1)

    # Handmatige prijs + Verwachte m2 (editable)
    if "Handmatige prijs" not in df.columns:
        df["Handmatige prijs"] = None
    if "Verwachte m2" not in df.columns:
        df["Verwachte m2"] = 0.0

    # Final prijs
    def final_price_row(r):
        hp = r.get("Handmatige prijs", None)
        return round(float(hp), 2) if pd.notna(hp) and hp != "" else r["RSP"]
    df["Final prijs"] = df.apply(final_price_row, axis=1)

    # Prijskwaliteit binnen bandbreedtes
    df["Prijskwaliteit"] = df.apply(
        lambda r: prijs_kwaliteit(
            r["Final prijs"],
            r.get("Min_prijs", None),
            r.get("Max_prijs", None)
        ),
        axis=1
    )

    # Omzetberekening
    df["Omzet conditie"] = (
        pd.to_numeric(df["Huidige m2 prijs"], errors="coerce").fillna(0) *
        pd.to_numeric(df["Verwachte m2"], errors="coerce").fillna(0)
    ).round(2)
    df["Omzet totaal"] = (
        pd.to_numeric(df["Final prijs"], errors="coerce").fillna(0) *
        pd.to_numeric(df["Verwachte m2"], errors="coerce").fillna(0)
    ).round(2)
    df["Effect aanpassing"] = (df["Omzet totaal"] - df["Omzet conditie"]).round(2)

    # Tabel tonen (selectief editable)
    show_cols = [
        "Artikelnummer","Artikelnaam","Productgroep","mm",
        "Huidige m2 prijs","RSP","Handmatige prijs","Final prijs",
        "Verwachte m2","Prijskwaliteit","Omzet conditie","Omzet totaal","Effect aanpassing"
    ]
    display_df = df.reindex(columns=show_cols).copy()

    edited = st.data_editor(
        display_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Artikelnummer": st.column_config.TextColumn(disabled=True),
            "Artikelnaam": st.column_config.TextColumn(disabled=True),
            "Productgroep": st.column_config.TextColumn(disabled=True),
            "mm": st.column_config.NumberColumn(disabled=True),
            "Huidige m2 prijs": st.column_config.NumberColumn(disabled=True),
            "RSP": st.column_config.NumberColumn(disabled=True),
            "Handmatige prijs": st.column_config.NumberColumn(help="Laat leeg om RSP te gebruiken"),
            "Final prijs": st.column_config.NumberColumn(disabled=True),
            "Verwachte m2": st.column_config.NumberColumn(help="Gebruik voor omzetberekening"),
            "Prijskwaliteit": st.column_config.TextColumn(disabled=True),
            "Omzet conditie": st.column_config.NumberColumn(disabled=True),
            "Omzet totaal": st.column_config.NumberColumn(disabled=True),
            "Effect aanpassing": st.column_config.NumberColumn(disabled=True),
        },
        key="prijs_editor",
    )

    # Herberekenen op basis van editor
    edited["Final prijs"] = edited.apply(
        lambda r: round(float(r["Handmatige prijs"]), 2) if pd.notna(r["Handmatige prijs"]) and r["Handmatige prijs"] != "" else r["RSP"],
        axis=1
    )
    edited["Omzet conditie"] = (
        pd.to_numeric(edited["Huidige m2 prijs"], errors="coerce").fillna(0) *
        pd.to_numeric(edited["Verwachte m2"], errors="coerce").fillna(0)
    ).round(2)
    edited["Omzet totaal"] = (
        pd.to_numeric(edited["Final prijs"], errors="coerce").fillna(0) *
        pd.to_numeric(edited["Verwachte m2"], errors="coerce").fillna(0)
    ).round(2)
    edited["Effect aanpassing"] = (edited["Omzet totaal"] - edited["Omzet conditie"]).round(2)

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
