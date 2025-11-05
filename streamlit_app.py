import streamlit as st
st.set_page_config(page_icon="🎯", page_title="PricePilot", layout="wide")
from streamlit_option_menu import option_menu
import pandas as pd
from io import BytesIO

# --- Optionele PDF export (eenvoudig) ---
def df_to_simple_pdf(df: pd.DataFrame, title: str = "Prijslijst") -> bytes:
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=landscape(A4))
        width, height = landscape(A4)

        # Titel
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, height - 1.5*cm, title)

        # Tabel (simpel: header + max ~25 rijen per pagina)
        cols = ["Material","Description","Productgroep","mm","Huidige m2 prijs","RSP","Handmatige prijs","Final prijs","Verwachte m2","Omzet conditie","Omzet totaal","Effect aanpassing"]
        show_df = df[cols].copy()

        x0, y0 = 1.5*cm, height - 3*cm
        line_height = 0.6*cm
        max_rows_per_page = int((y0 - 1.5*cm) / line_height) - 1

        def draw_row(y, values, bold=False):
            c.setFont("Helvetica-Bold" if bold else "Helvetica", 8)
            x = x0
            col_widths = [3*cm, 6*cm, 3.2*cm, 1.6*cm, 3*cm, 2.4*cm, 3*cm, 2.6*cm, 2.6*cm, 3*cm, 3*cm, 3*cm]
            for v, w in zip(values, col_widths):
                c.drawString(x, y, str(v))
                x += w

        # Header
        draw_row(y0, cols, bold=True)
        y = y0 - line_height

        for i, row in show_df.iterrows():
            draw_row(y, [row.get(cn, "") for cn in cols], bold=False)
            y -= line_height
            if y < 2*cm:
                c.showPage()
                c.setFont("Helvetica-Bold", 14)
                c.drawString(2*cm, height - 1.5*cm, title)
                draw_row(y0, cols, bold=True)
                y = y0 - line_height

        c.save()
        buffer.seek(0)
        return buffer.read()
    except Exception:
        return b""  # Als reportlab niet geïnstalleerd is

# --- Data laden ---
def load_articles():
    try:
        from articles import articles
        return pd.DataFrame(articles)
    except Exception:
        # Fallback demo
        data = [
            {'Material': 1000055, 'Description': 'IsoStandard 04 - 04', 'Min_prijs': 27, 'Max_prijs': 35.1, 'Productgroep': 'IsoStandard', 'UC_waarde': 'IsoStandard: TL 82, g 0.79 Ug=2.7W/m².K (gebaseerd op 5-15L-4)', 'Min_afname': 0.65, 'mm': 8, 'SML': 'M'},
            {'Material': 1000056, 'Description': 'IsoStandard 05 - 04', 'Min_prijs': 28.5, 'Max_prijs': 37.1, 'Productgroep': 'IsoStandard', 'UC_waarde': 'IsoStandard: TL 82, g 0.79 Ug=2.7W/m².K', 'Min_afname': 0.65, 'mm': 9, 'SML': 'S'},
            {'Material': 1000058, 'Description': 'IsoStandard 05 - 05', 'Min_prijs': 30, 'Max_prijs': 39, 'Productgroep': 'IsoStandard', 'UC_waarde': 'IsoStandard: TL 82, g 0.79 Ug=2.7W/m².K', 'Min_afname': 0.65, 'mm': 10, 'SML': 'L'},
            {'Material': 1000061, 'Description': 'IsoStandard 33.1 - 04', 'Min_prijs': 40.5, 'Max_prijs': 52.7, 'Productgroep': 'IsoStandard', 'UC_waarde': 'IsoStandard: TL 82, g 0.79 Ug=2.7W/m².K', 'Min_afname': 0.65, 'mm': 10, 'SML': 'S'},
        ]
        return pd.DataFrame(data)

def load_sap_prices():
    try:
        from SAPprijs import sap_prices
        return sap_prices
    except Exception:
        # Fallback demo
        return {
            "100007": {
                "1000055": 32.4,
                "1000056": 34.2,
                "1000058": 36.0,
                "1000061": 38.4,
            },
            "100008": {
                "1000055": 33.1,
                "1000056": 35.0,
                "1000058": 36.4,
                "1000061": 39.2,
            }
        }

# --- Helpers ---
def sml_filter(df: pd.DataFrame, sml_pick: str) -> pd.DataFrame:
    if sml_pick == "S":
        return df[df["SML"] == "S"]
    elif sml_pick == "M":
        return df[df["SML"].isin(["S", "M"])]
    else:
        return df  # L = alles

def compute_rsp(row, base_price_alfa, per_pg_uplift, per_mm_uplift):
    # Alfa is standaard met toeslag 0
    pg = row.get("Productgroep", "")
    mm = float(row.get("mm", 0) or 0)
    pg_uplift = float(per_pg_uplift.get(pg, 0) or 0)
    return float(base_price_alfa) + pg_uplift + (mm * float(per_mm_uplift))

def prijs_kwaliteit(final_price, min_p, max_p):
    try:
        if pd.isna(final_price):
            return ""
        if final_price < min_p:
            return "Onder min"
        if final_price > max_p:
            return "Boven max"
        return "Binnen band"
    except Exception:
        return ""

# --- App ---
st.set_page_config(page_icon="🎯", page_title="PricePilot", layout="wide")
st.title("🎯 PricePilot – Prijslijsten per klant")

if "articles_df" not in st.session_state:
    st.session_state.articles_df = load_articles()
if "sap_prices" not in st.session_state:
    st.session_state.sap_prices = load_sap_prices()

tabs = st.tabs(["📄 Prijslijst", "🛠️ Beheer"])

with tabs[0]:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Selecties")
        # Klantselectie (vervang later door Salesforce picklist)
        klantnrs = list(st.session_state.sap_prices.keys())
        klant = st.selectbox("Klantnummer", klantnrs, index=0)

        # Productgroep selectie
        alle_pg = sorted(st.session_state.articles_df["Productgroep"].dropna().unique())
        sel_pg = st.multiselect("Productgroepen", alle_pg, default=alle_pg[:1])

        # S/M/L selectie
        sml_pick = st.radio("S/M/L lijst", ["S", "M", "L"], horizontal=True,
                            help="S = hardlopers, M = S+M, L = alle artikelen")

        st.divider()
        st.subheader("Prijsparameters")
        base_price_alfa = st.number_input(
            "Basismateriaalprijs (IsoPerform ALFA 04 - #04)",
            min_value=0.0, value=30.0, step=0.1,
            help="Startpunt voor RSP; Alfa heeft geen coatingtoeslag"
        )
        per_mm_uplift = st.number_input(
            "Opslag per mm (€/mm)",
            min_value=0.0, value=0.40, step=0.05
        )

        st.markdown("**Opslag per productgroep (€/m²)**")
        per_pg_uplift = {}
        for pg in sel_pg:
            per_pg_uplift[pg] = st.number_input(f"Toeslag {pg}", value=0.0, step=0.1, key=f"uplift_{pg}")

        st.divider()
        st.caption("Optioneel: export")
        export_name = st.text_input("Bestandsnaam (zonder extensie)", value="prijslijst")

    with right:
        # Filter op PG & S/M/L
        df = st.session_state.articles_df.copy()
        if sel_pg:
            df = df[df["Productgroep"].isin(sel_pg)]
        df = sml_filter(df, sml_pick).copy()

        # Kolommen voorbereiden
        df["Material"] = df["Material"].astype(int)
        df["Artikelnummer"] = df["Material"].astype(str)
        df["Artikelnaam"] = df["Description"]
        df["mm"] = pd.to_numeric(df.get("mm", 0), errors="coerce").fillna(0).astype(float)

        # Huidige m2 prijs uit SAP
        sap_for_client = st.session_state.sap_prices.get(klant, {})
        df["Huidige m2 prijs"] = df["Artikelnummer"].map(lambda x: sap_for_client.get(x, None))

        # RSP berekening (basis + coating + mm)
        df["RSP"] = df.apply(
            lambda r: round(compute_rsp(r, base_price_alfa, per_pg_uplift, per_mm_uplift), 2),
            axis=1
        )

        # Handmatige prijs (editable)
        if "Handmatige prijs" not in df.columns:
            df["Handmatige prijs"] = None

        # Verwachte m2 voor omzetberekening (editable)
        if "Verwachte m2" not in df.columns:
            df["Verwachte m2"] = 0.0

        # Final prijs = handmatig of RSP
        def coalesce_price(row):
            hp = row.get("Handmatige prijs", None)
            return round(float(hp), 2) if pd.notna(hp) and hp != "" else row["RSP"]

        df["Final prijs"] = df.apply(coalesce_price, axis=1)

        # Prijskwaliteit vs band
        df["Prijskwaliteit"] = df.apply(
            lambda r: prijs_kwaliteit(r["Final prijs"], r.get("Min_prijs", None), r.get("Max_prijs", None)),
            axis=1
        )

        # Omzet-conditie (huidig) en totaal (nieuw), en effect
        df["Omzet conditie"] = (pd.to_numeric(df["Huidige m2 prijs"], errors="coerce").fillna(0) *
                                pd.to_numeric(df["Verwachte m2"], errors="coerce").fillna(0)).round(2)
        df["Omzet totaal"] = (pd.to_numeric(df["Final prijs"], errors="coerce").fillna(0) *
                              pd.to_numeric(df["Verwachte m2"], errors="coerce").fillna(0)).round(2)
        df["Effect aanpassing"] = (df["Omzet totaal"] - df["Omzet conditie"]).round(2)

        # Volgorde en tonen in editor (selectief editable)
        show_cols = [
            "Artikelnummer","Artikelnaam","Productgroep","mm",
            "Huidige m2 prijs","RSP","Handmatige prijs","Final prijs",
            "Verwachte m2","Prijskwaliteit","Omzet conditie","Omzet totaal","Effect aanpassing"
        ]
        display_df = df[show_cols].copy()

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

        # Final prijs herberekenen vanuit editor
        edited["Final prijs"] = edited.apply(
            lambda r: round(float(r["Handmatige prijs"]),2) if pd.notna(r["Handmatige prijs"]) and r["Handmatige prijs"] != "" else r["RSP"],
            axis=1
        )
        edited["Omzet conditie"] = (pd.to_numeric(edited["Huidige m2 prijs"], errors="coerce").fillna(0) *
                                    pd.to_numeric(edited["Verwachte m2"], errors="coerce").fillna(0)).round(2)
        edited["Omzet totaal"] = (pd.to_numeric(edited["Final prijs"], errors="coerce").fillna(0) *
                                  pd.to_numeric(edited["Verwachte m2"], errors="coerce").fillna(0)).round(2)
        edited["Effect aanpassing"] = (edited["Omzet totaal"] - edited["Omzet conditie"]).round(2)

        st.caption(f"Regels: {len(edited)}")

        # Downloads
        c1, c2 = st.columns(2)
        with c1:
            csv_bytes = edited.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv_bytes, file_name=f"{export_name}.csv", mime="text/csv")
        with c2:
            pdf_bytes = df_to_simple_pdf(edited, title=f"Prijslijst klant {klant}")
            if pdf_bytes:
                st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name=f"{export_name}.pdf", mime="application/pdf")
            else:
                st.info("PDF-export vereist `reportlab`. Installeer met: `pip install reportlab`")

with tabs[1]:
    st.subheader("Beheer – artikelen")
    base = st.session_state.articles_df.copy()

    # Editor: alleen Min_prijs, Max_prijs, SML aanpasbaar
    beheer_cols = ["Material","Description","Productgroep","mm","Min_prijs","Max_prijs","SML","UC_waarde","Min_afname"]
    base = base[beheer_cols].copy()

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
            "SML": st.column_config.SelectboxColumn(options=["S","M","L"], help="S=hardloper, M=mid, L=lang"),
            "UC_waarde": st.column_config.TextColumn(disabled=True),
            "Min_afname": st.column_config.NumberColumn(disabled=True),
        },
        key="beheer_editor",
    )

    st.caption("Alleen Min_prijs, Max_prijs en SML zijn bewerkbaar. Overige velden zijn vergrendeld.")
    if st.button("💾 Wijzigingen toepassen"):
        # Schrijf wijzigingen terug naar session_state
        upd = beheer_edit.set_index("Material")[["Min_prijs","Max_prijs","SML"]]
        full = st.session_state.articles_df.set_index("Material")
        for col in ["Min_prijs","Max_prijs","SML"]:
            full.loc[upd.index, col] = upd[col]
        st.session_state.articles_df = full.reset_index()
        st.success("Wijzigingen opgeslagen (in geheugen). Exporteer om te bewaren buiten de app.")

    colx, coly = st.columns(2)
    with colx:
        # Export aangepaste articles naar CSV (of JSON)
        export_articles_csv = st.session_state.articles_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporteer artikelen (CSV)", data=export_articles_csv, file_name="articles_updated.csv", mime="text/csv")
    with coly:
        # JSON
        export_articles_json = st.session_state.articles_df.to_json(orient="records", force_ascii=False).encode("utf-8")
        st.download_button("⬇️ Exporteer artikelen (JSON)", data=export_articles_json, file_name="articles_updated.json", mime="application/json")

