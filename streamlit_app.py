import os
import io
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="BullsAI • Materials",
    page_icon="🧊",
    layout="wide",
)

DATA_PATH_DEFAULT = "materials_augmented_mm_sml.csv"
SESSION_KEY = "materials_df"

# -------------------------------
# Helpers
# -------------------------------

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Material": "Int64"}, sep=",")
    return normalize_df(df)

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Zorg voor robuuste kolommen en types
    rename_map = {
        "material": "Material",
        "description": "Description",
        "min_prijs": "Min_prijs",
        "max_prijs": "Max_prijs",
        "productgroep": "Productgroep",
        "uc_waarde": "UC_waarde",
        "min_afname": "Min_afname",
    }
    df = df.rename(columns=rename_map)

    # Ensure columns exist
    for col in ["Material", "Description", "Min_prijs", "Max_prijs", "Productgroep",
                "UC_waarde", "Min_afname", "mm", "SML"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Types
    for col in ["Min_prijs", "Max_prijs", "Min_afname", "mm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Material" in df.columns:
        df["Material"] = pd.to_numeric(df["Material"], errors="coerce").astype("Int64")

    # SML naar categorie volgorde
    if "SML" in df.columns:
        df["SML"] = df["SML"].astype("string")
        df["SML"] = pd.Categorical(df["SML"], categories=["S", "M", "L"], ordered=True)

    # Opschonen strings
    for col in ["Description", "Productgroep", "UC_waarde"]:
        df[col] = df[col].astype("string")

    return df

def save_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Materials")
    buf.seek(0)
    return buf.read()

def quick_text_filter(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if not q:
        return df
    q = q.strip().lower()
    mask = (
        df["Description"].fillna("").str.lower().str.contains(q)
        | df["Productgroep"].fillna("").str.lower().str.contains(q)
        | df["UC_waarde"].fillna("").str.lower().str.contains(q)
        | df["Material"].astype("string").str.contains(q)
    )
    return df[mask]

def apply_advanced_filters(
    df: pd.DataFrame,
    productgroepen: list,
    sml_sel: list,
    mm_range: tuple,
    minprijs_range: tuple,
    maxprijs_range: tuple,
) -> pd.DataFrame:
    out = df.copy()
    if productgroepen:
        out = out[out["Productgroep"].isin(productgroepen)]
    if sml_sel:
        out = out[out["SML"].astype("string").isin([str(x) for x in sml_sel])]
    if mm_range:
        lo, hi = mm_range
        out = out[(out["mm"].isna()) | ((out["mm"] >= lo) & (out["mm"] <= hi))]
    if minprijs_range:
        lo, hi = minprijs_range
        out = out[(out["Min_prijs"].isna()) | ((out["Min_prijs"] >= lo) & (out["Min_prijs"] <= hi))]
    if maxprijs_range:
        lo, hi = maxprijs_range
        out = out[(out["Max_prijs"].isna()) | ((out["Max_prijs"] >= lo) & (out["Max_prijs"] <= hi))]
    return out

# -------------------------------
# Data in session state laden
# -------------------------------

def ensure_data_in_session():
    if SESSION_KEY in st.session_state and isinstance(st.session_state[SESSION_KEY], pd.DataFrame):
        return
    if os.path.exists(DATA_PATH_DEFAULT):
        st.session_state[SESSION_KEY] = load_csv(DATA_PATH_DEFAULT)
    else:
        st.session_state[SESSION_KEY] = normalize_df(pd.DataFrame([]))

ensure_data_in_session()

# -------------------------------
# Sidebar: option menu
# -------------------------------

with st.sidebar:
    selected = option_menu(
        "BullsAI • Materials",
        [
            "Overzicht",
            "Filter & zoek",
            "Statistiek",
            "Import / Export",
            "Info",
        ],
        icons=["table", "funnel", "bar-chart", "upload", "info-circle"],
        menu_icon="boxes",
        default_index=0,
        orientation="vertical",
    )

# -------------------------------
# Pagina's
# -------------------------------

df = st.session_state[SESSION_KEY]

if selected == "Overzicht":
    st.title("Alle materialen")
    st.caption("Toont **alle** records uit de dataset (geen demo-subset).")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        q = st.text_input("Snel zoeken (materiaal, omschrijving, productgroep…)", placeholder="Typ om te filteren…")
    with c2:
        sort_col = st.selectbox("Sorteer op", ["Material", "Description", "Productgroep", "mm", "Min_prijs", "Max_prijs"], index=0)
    with c3:
        asc = st.toggle("Oplopend", value=True)

    view = quick_text_filter(df, q)
    if sort_col in view.columns:
        view = view.sort_values(by=sort_col, ascending=asc, kind="mergesort")

    st.write(f"**{len(view):,}** materialen gevonden.")
    st.dataframe(view, use_container_width=True, height=560)

elif selected == "Filter & zoek":
    st.title("Filter & zoek")
    st.caption("Combineer filters voor een gerichte selectie.")

    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            productgroepen = sorted([x for x in df["Productgroep"].dropna().unique().tolist()])
            pg_sel = st.multiselect("Productgroep", productgroepen)
            sml_sel = st.multiselect("SML (S≤6, M7–10, L≥11)", options=["S", "M", "L"])
        with col2:
            mm_min = int(pd.to_numeric(df["mm"], errors="coerce").min(skipna=True) or 0)
            mm_max = int(pd.to_numeric(df["mm"], errors="coerce").max(skipna=True) or 50)
            mm_range = st.slider("Totale dikte (mm)", min_value=mm_min, max_value=mm_max, value=(mm_min, mm_max))
            min_min = float(pd.to_numeric(df["Min_prijs"], errors="coerce").min(skipna=True) or 0.0)
            min_max = float(pd.to_numeric(df["Min_prijs"], errors="coerce").max(skipna=True) or 0.0)
            minprijs_range = st.slider("Min_prijs", min_value=float(round(min_min, 2)), max_value=float(round(min_max, 2)),
                                       value=(float(round(min_min, 2)), float(round(min_max, 2))))
        with col3:
            max_min = float(pd.to_numeric(df["Max_prijs"], errors="coerce").min(skipna=True) or 0.0)
            max_max = float(pd.to_numeric(df["Max_prijs"], errors="coerce").max(skipna=True) or 0.0)
            maxprijs_range = st.slider("Max_prijs", min_value=float(round(max_min, 2)), max_value=float(round(max_max, 2)),
                                       value=(float(round(max_min, 2)), float(round(max_max, 2))))
    q2 = st.text_input("Vrije tekst zoekterm (material/omschrijving/UC_waarde)", placeholder="Bijv. SKN165 of 44.2")

    filtered = apply_advanced_filters(df, pg_sel, sml_sel, mm_range, minprijs_range, maxprijs_range)
    filtered = quick_text_filter(filtered, q2)

    st.write(f"Resultaat: **{len(filtered):,}** materialen")
    st.dataframe(filtered, use_container_width=True, height=560)

    st.download_button("📥 Download selectie (CSV)", data=filtered.to_csv(index=False).encode("utf-8"),
                       file_name="materials_filtered.csv", mime="text/csv")

elif selected == "Statistiek":
    st.title("Statistiek")
    st.caption("Snelle inzichten per productgroep.")

    if df.empty:
        st.info("Geen data geladen.")
    else:
        grp = (
            df.groupby("Productgroep", dropna=False)
              .agg(
                  Aantal=("Material", "count"),
                  Gem_mm=("mm", "mean"),
                  Gem_MinPrijs=("Min_prijs", "mean"),
                  Gem_MaxPrijs=("Max_prijs", "mean"),
              )
              .reset_index()
              .sort_values("Aantal", ascending=False)
        )
        st.dataframe(grp, use_container_width=True, height=480)

        st.subheader("Aantal per productgroep")
        st.bar_chart(grp.set_index("Productgroep")["Aantal"])

        st.subheader("Gemiddelde mm per productgroep")
        st.bar_chart(grp.set_index("Productgroep")["Gem_mm"])

elif selected == "Import / Export":
    st.title("Import / Export")
    st.caption("Laad nieuwe data of exporteer de huidige tabel.")

    st.markdown("**Importeer** CSV of Excel met de verwachte kolommen.")
    up = st.file_uploader("Kies bestand", type=["csv", "xlsx"], accept_multiple_files=False)
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                new_df = pd.read_csv(up, dtype={"Material": "Int64"})
            else:
                new_df = pd.read_excel(up, dtype={"Material": "Int64"})
            new_df = normalize_df(new_df)
            st.session_state[SESSION_KEY] = new_df
            st.success(f"Ingeladen: {len(new_df):,} regels.")
        except Exception as e:
            st.error(f"Kon bestand niet verwerken: {e}")

    st.divider()
    st.markdown("**Export** huidige dataset:")
    cur = st.session_state[SESSION_KEY]
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📥 Download alles (CSV)",
            data=cur.to_csv(index=False).encode("utf-8"),
            file_name="materials_full.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "📥 Download alles (Excel)",
            data=save_to_excel_bytes(cur),
            file_name="materials_full.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

elif selected == "Info":
    st.title("Info")
    st.markdown(
        """
        **BullsAI • Materials**
        - Linker **option-menu** voor navigatie
        - **Overzicht**: alle materialen + snelle zoekbalk
        - **Filter & zoek**: productgroep/SML/mm/prijs en vrije tekst
        - **Statistiek**: aantallen en gemiddelden per productgroep
        - **Import / Export**: upload CSV/XLSX of exporteer actuele set

        **SML-definitie**  
        - **S**: mm ≤ 6  
        - **M**: 7 ≤ mm ≤ 10  
        - **L**: mm ≥ 11  
        """
    )

# Footer subtiel
st.markdown(
    "<div style='text-align: right; color: #888; font-size: 12px;'>BullsAI • Materials</div>",
    unsafe_allow_html=True,
)
