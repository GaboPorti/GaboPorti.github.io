# app.py
# --- KPIs Hospitalarios con pestañas, gráficos y selector de hoja ---
# Ejecuta con:  python -m streamlit run app.py

import math
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Utilidades de fechas / LOS
# ================================
def try_parse_date(s):
    if pd.isna(s):
        return pd.NaT
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y",
                "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return pd.to_datetime(s, format=fmt, errors="raise")
        except Exception:
            continue
    return pd.to_datetime(s, errors="coerce")

def ensure_datetime(col: pd.Series) -> pd.Series:
    if np.issubdtype(col.dtype, np.datetime64):
        return col
    return col.apply(try_parse_date)

def compute_los(admit_col: pd.Series, discharge_col: pd.Series, unit: str = "days") -> pd.Series:
    admit = ensure_datetime(admit_col)
    discharge = ensure_datetime(discharge_col)
    delta = (discharge - admit)
    if unit == "hours":
        return delta.dt.total_seconds() / 3600.0
    return delta.dt.total_seconds() / (3600.0 * 24.0)

# ================================
# Validación automática de dataset
# ================================
from datetime import date

def _pct_missing(s: pd.Series) -> float:
    return float((s.isna().mean() * 100).round(2))

def validate_dataset(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    """
    Devuelve un DataFrame con la matriz de chequeos:
    columnas: ['Chequeo','Resultado','Severidad','Sugerencia']
    Severidad: OK / ADVERTENCIA / CRÍTICO
    """
    rows = []
    today = pd.Timestamp(date.today())

    def add(check, result, severity="OK", hint=""):
        rows.append({"Chequeo": check, "Resultado": result, "Severidad": severity, "Sugerencia": hint})

    # 0) Filas
    add("Filas totales", f"{len(df):,}", "OK" if len(df) > 0 else "CRÍTICO",
        "El archivo está vacío" if len(df) == 0 else "")

    # 1) Columnas requeridas
    required = ["patient_id", "admit_dt", "discharge_dt"]
    for k in required:
        ok = bool(colmap.get(k)) and colmap[k] in df.columns
        add(f"Columna requerida: {k}", "OK" if ok else "Falta", "OK" if ok else "CRÍTICO",
            f"Selecciona una columna para '{k}' en el panel izquierdo" if not ok else "")

    # Si faltan columnas críticas, devolvemos hasta aquí
    if any(r["Severidad"] == "CRÍTICO" for r in rows):
        return pd.DataFrame(rows)

    # 2) Parseo fechas y LOS
    admit = ensure_datetime(df[colmap["admit_dt"]])
    disch = ensure_datetime(df[colmap["discharge_dt"]])
    los = (disch - admit).dt.total_seconds() / 86400

    add("Admisiones no legibles (%)", f"{_pct_missing(admit)}%",
        "ADVERTENCIA" if _pct_missing(admit) > 5 else "OK",
        "Revisa el formato de fecha en 'ingreso' si >5% NaT")

    add("Altas no legibles (%)", f"{_pct_missing(disch)}%",
        "ADVERTENCIA" if _pct_missing(disch) > 5 else "OK",
        "Revisa el formato de fecha en 'alta' si >5% NaT")

    # 3) Reglas de consistencia
    neg = int((los < 0).sum())
    add("Estadías negativas (alta < ingreso)", f"{neg}",
        "CRÍTICO" if neg > 0 else "OK",
        "Revisa registros con alta anterior al ingreso")

    zero = int((los == 0).sum())
    add("Estadías de 0 días", f"{zero}",
        "ADVERTENCIA" if zero > 0 else "OK",
        "Verifica egresos el mismo día; pueden ser válidos en observación")

    fut_in = int((admit > today + pd.Timedelta(days=1)).sum())
    fut_out = int((disch > today + pd.Timedelta(days=1)).sum())
    add("Ingresos en el futuro", f"{fut_in}",
        "ADVERTENCIA" if fut_in > 0 else "OK",
        "Posible error de digitación (año/mes)")

    add("Altas en el futuro", f"{fut_out}",
        "ADVERTENCIA" if fut_out > 0 else "OK",
        "Posible error de digitación (año/mes)")

    # 4) Duplicados por episodio
    dup_in = int(df.duplicated(subset=[colmap["patient_id"], colmap["admit_dt"]], keep=False).sum())
    dup_out = int(df.duplicated(subset=[colmap["patient_id"], colmap["discharge_dt"]], keep=False).sum())
    add("Duplicados (paciente + ingreso)", f"{dup_in}",
        "ADVERTENCIA" if dup_in > 0 else "OK",
        "Revisa episodios repetidos en la misma fecha de ingreso")

    add("Duplicados (paciente + alta)", f"{dup_out}",
        "ADVERTENCIA" if dup_out > 0 else "OK",
        "Revisa episodios repetidos en la misma fecha de alta")

    # 5) Outliers simples
    if los.notna().any():
        p99 = float(np.nanpercentile(los, 99))
        max_los = float(np.nanmax(los))
        gt60 = int((los > 60).sum())
        add("LOS p99 (días)", f"{p99:.2f}")
        add("LOS máximo (días)", f"{max_los:.2f}",
            "ADVERTENCIA" if max_los > p99 * 1.5 or max_los > 60 else "OK",
            "Hay estancias extremas; valida casos o usa IEMA/inliers")
        add("LOS > 60 días (conteo)", f"{gt60}",
            "ADVERTENCIA" if gt60 > 0 else "OK",
            "Muy prolongadas; confirmar vigencias/traslados")
    else:
        add("LOS calculable", "No", "ADVERTENCIA", "Fechas insuficientes para calcular LOS")

    # 6) Cobertura de variables opcionales
    for opt in ["service", "dx_group"]:
        if colmap.get(opt) and colmap[opt] in df.columns:
            miss = _pct_missing(df[colmap[opt]])
            add(f"{opt} faltante (%)", f"{miss}%",
                "ADVERTENCIA" if miss > 10 else "OK",
                "Completar/codificar para mejores desglose y KPIs" if miss > 10 else "")

    rep = pd.DataFrame(rows)
    # Ordenar por severidad
    sev_order = {"CRÍTICO": 0, "ADVERTENCIA": 1, "OK": 2}
    rep["__ord"] = rep["Severidad"].map(sev_order)
    rep = rep.sort_values(["__ord", "Chequeo"]).drop(columns="__ord").reset_index(drop=True)
    return rep


# ================================
# Indicadores
# ================================
def compute_readmission_72h(df, patient_col, admit_col, discharge_col, by_cols: List[str]):
    work = df.copy()
    work[admit_col] = ensure_datetime(work[admit_col])
    work[discharge_col] = ensure_datetime(work[discharge_col])
    work = work.sort_values([patient_col, admit_col])
    work["next_admit"] = work.groupby(patient_col)[admit_col].shift(-1)
    work["this_discharge"] = work[discharge_col]
    work["readmit_72h"] = (work["next_admit"] - work["this_discharge"]).dt.total_seconds()/(3600) <= 72
    work["readmit_72h"] = work["readmit_72h"].fillna(False)
    grp = work.groupby(by_cols, dropna=False)["readmit_72h"].mean().rename("Reingreso_72h").reset_index()
    return grp

# ================================
# Configuración app
# ================================
st.title("🏥 KPIs Hospitalarios – MVP")

st.markdown("""
Esta aplicación permite cargar datos hospitalarios (CSV/Excel) y calcular indicadores clave de **hospitalización, urgencia, economía y salud pública**.  
Incluye tablas y **gráficos automáticos** (barras, líneas y distribución) para facilitar la interpretación y la toma de decisiones.

### ¿Qué hace la app?
- Lee tu archivo **.csv** o **.xlsx** (si es Excel, puedes elegir la **hoja**).
- Te permite **mapear** las columnas principales (paciente, fechas, servicio, diagnóstico).
- Calcula KPIs y muestra **tablas** y **gráficos** por el/los **grupos** que tú elijas (puedes usar cualquier columna del dataset).

---

## Definiciones de indicadores

### 🏨 Hospitalización
- **ALOS (Average Length of Stay):** Promedio de días hospitalizados por episodio.  
- **Mediana LOS:** Valor central de la estadía hospitalaria (menos sensible a outliers).  
- **Mortalidad (%):** Proporción de egresos con fallecimiento (si existe columna `death/defuncion`).  
- **Ocupación de camas (%):** (bed-days usados / bed-days disponibles) × 100, si subes tabla de camas por día/servicio.  
- **Rotación de camas:** Altas / (camas disponibles promedio en el período).  
- **TOI – Tiempo Ocioso de Instalación (días):** (bed-days disponibles – bed-days usados) / altas.  

> Para ocupación/rotación/TOI debes cargar una **tabla de camas** por día y (opcionalmente) por servicio.

### 🚑 Urgencia
- **Reingreso < 72 horas:** % de pacientes que reingresan dentro de 72 h desde el alta anterior.  
- **Tiempos de espera (opcional):** Si existen columnas de llegada y primera evaluación, se grafican tiempos promedio.

### 💰 Economía
- **Ingreso total por GRD/diagnóstico:** Suma de pagos por caso/día (si subes tabla económica).  
- **Costo total por GRD/diagnóstico:** Suma de costos por caso/día (si subes tabla económica).  
- **Margen:** Ingreso – Costo (si ambos están disponibles).

### 🌍 Salud pública
- **Tasa de hospitalización por diagnóstico prevalente:** Proporción de hospitalizaciones por diagnóstico (ranking de los más frecuentes).  
- **Ocupación de camas críticas (opcional):** Requiere tabla de camas críticas por día/servicio.  
- **Referencias y contrarreferencias (opcional):** Si el dataset incluye columnas de referencia/derivación, se calculan tasas.

---

**Sugerencia:** usa el selector **“Agrupar indicadores por…”** para comparar por Servicio, Diagnóstico, Sexo, Tramo etario, etc.  
Si alguna métrica no aparece, probablemente falta esa columna o la tabla de soporte (p. ej., camas o economía).
""")

# ================================
# Sidebar – Carga de datos + mapeo
# ================================
with st.sidebar:
    st.header("1) Carga de datos")
    f = st.file_uploader("Archivo (.csv o .xlsx)", type=["csv", "xlsx"])
    df = None
    selected_sheet = None
    if f is not None:
        if f.name.lower().endswith(".csv"):
            df = pd.read_csv(f)
        else:
            # Selector de hoja para Excel
            excel = pd.ExcelFile(f)
            selected_sheet = st.selectbox("Hoja de Excel", options=excel.sheet_names)
            f.seek(0)  # volver al inicio del buffer por seguridad
            df = pd.read_excel(f, sheet_name=selected_sheet)

    st.header("2) Mapeo de columnas")
    colmap = {}
    if df is not None:
        cols = list(df.columns)
        colmap["patient_id"]  = st.selectbox("Identificador paciente", [None] + cols)
        colmap["admit_dt"]    = st.selectbox("Fecha/hora ingreso", [None] + cols)
        colmap["discharge_dt"]= st.selectbox("Fecha/hora alta", [None] + cols)
        colmap["service"]     = st.selectbox("Servicio/Unidad (opcional)", [None] + cols)
        colmap["dx_group"]    = st.selectbox("Diagnóstico/GRD (opcional)", [None] + cols)

ok_to_process = (
    df is not None and all(colmap[k] for k in ["patient_id", "admit_dt", "discharge_dt"])
)

if ok_to_process:
    # --- Validación automática ---
    st.subheader("🔎 Validación automática del dataset")
    report = validate_dataset(df, colmap)
    st.dataframe(report, use_container_width=True)

    criticos = (report["Severidad"] == "CRÍTICO").sum()
    advert   = (report["Severidad"] == "ADVERTENCIA").sum()

    if criticos > 0:
        st.error("Se detectaron **problemas críticos**. Corrige el archivo o el mapeo antes de continuar.")
        st.stop()
    elif advert > 0:
        st.warning("Se detectaron **advertencias**. Puedes continuar, pero revisa los puntos señalados.")
    else:
        st.success("Validación OK ✅")

    # --- Prepara base una vez superada la validación ---
    work = df.copy()
    work["LOS_days"] = compute_los(work[colmap["admit_dt"]], work[colmap["discharge_dt"]], unit="days")


    # ================================
    # Agrupaciones flexibles
    # ================================
    st.subheader("Agrupar indicadores por…")
    all_columns = list(df.columns)
    default_groups = []
    if colmap.get("service") and colmap["service"] in all_columns:
        default_groups.append(colmap["service"])
    if colmap.get("dx_group") and colmap["dx_group"] in all_columns:
        default_groups.append(colmap["dx_group"])

    group_cols = st.multiselect(
        "Elige 0..N columnas (puede ser cualquiera del dataset)",
        options=all_columns,
        default=default_groups
    )

    # ================================
    # Pestañas
    # ================================
    tab_hosp, tab_er, tab_econ, tab_sp = st.tabs(
        ["🏨 Hospitalización", "🚑 Urgencia", "💰 Economía", "🌍 Salud Pública"]
    )

    # -------- HOSPITALIZACIÓN --------
    with tab_hosp:
        st.header("Indicadores de Hospitalización")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("ALOS (días)", f"{work['LOS_days'].mean():.2f}")
        with c2: st.metric("Mediana LOS", f"{work['LOS_days'].median():.2f}")
        with c3: st.metric("Episodios", f"{len(work)}")

        # Tabla agregada
        if group_cols:
            hos_tbl = work.groupby(group_cols, dropna=False)["LOS_days"].agg(
                ALOS="mean", Mediana="median", Episodios="count"
            ).reset_index()
            hos_tbl["ALOS"] = hos_tbl["ALOS"].round(2)
            hos_tbl["Mediana"] = hos_tbl["Mediana"].round(2)
            st.dataframe(hos_tbl, use_container_width=True)
        else:
            st.info("Puedes seleccionar columnas para ver el desglose por grupos.")

        # Histograma LOS
        st.markdown("### Distribución de LOS")
        fig, ax = plt.subplots(figsize=(7,4))
        sns.histplot(work["LOS_days"], bins=30, kde=True, ax=ax)
        ax.set_xlabel("LOS (días)"); ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

        # Boxplot por primera columna seleccionada (si existe)
        if group_cols:
            st.markdown(f"### LOS por {group_cols[0]}")
            fig, ax = plt.subplots(figsize=(8,4))
            sns.boxplot(x=work[group_cols[0]], y=work["LOS_days"], ax=ax)
            ax.set_xlabel(group_cols[0]); ax.set_ylabel("LOS (días)")
            plt.xticks(rotation=45)
            st.pyplot(fig)

# =============== PESTAÑA URGENCIA ===============
with tab_er:
    st.header("Indicadores de Urgencia")

    # ---------- CONFIGURACIÓN (solo en esta pestaña) ----------
    with st.expander("⚙️ Configurar datos de Urgencias"):
        urg_file = st.file_uploader(
            "Tabla de urgencias (CSV/XLSX)", type=["csv", "xlsx"], key="urg"
        )
        urg_df, urg_map = None, {}
        if urg_file is not None:
            try:
                urg_df = pd.read_csv(urg_file) if urg_file.name.lower().endswith(".csv") else pd.read_excel(urg_file)
                st.caption(f"Columnas detectadas: {list(urg_df.columns)}")
                urg_map["patient"]      = st.selectbox("Identificador paciente", options=list(urg_df.columns))
                urg_map["arrival_dt"]   = st.selectbox("Fecha/hora de llegada", options=list(urg_df.columns))
                urg_map["triage_level"] = st.selectbox("Nivel de triage", options=list(urg_df.columns))
                urg_map["first_doc_dt"] = st.selectbox("Primera atención médica", options=list(urg_df.columns))
                urg_map["discharge_dt"] = st.selectbox("Fecha/hora de alta", options=list(urg_df.columns))
                urg_map["dx_group"]     = st.selectbox("Diagnóstico / GRD", options=list(urg_df.columns))
            except Exception as e:
                st.error(f"No pude leer la tabla de urgencias: {e}")
                urg_df = None

    # ---------- INDICADORES ----------
    if urg_df is None:
        st.info("Sube y mapea una tabla de urgencias para ver los KPIs de esta sección.")
    else:
        # Copia base y tiempos
        u = urg_df.copy()
        u["_arrive"]  = ensure_datetime(u[urg_map["arrival_dt"]])
        u["_first"]   = ensure_datetime(u[urg_map["first_doc_dt"]])
        u["_out"]     = ensure_datetime(u[urg_map["discharge_dt"]])
        u["wait_min"] = (u["_first"] - u["_arrive"]).dt.total_seconds() / 60.0
        u["stay_min"] = (u["_out"]   - u["_arrive"]).dt.total_seconds() / 60.0

        st.subheader("Resumen")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "Tiempo de espera medio (min)",
                f"{u['wait_min'].mean():.1f}" if u["wait_min"].notna().any() else "-"
            )
        with c2:
            st.metric(
                "Tiempo total medio (min)",
                f"{u['stay_min'].mean():.1f}" if u["stay_min"].notna().any() else "-"
            )
        with c3:
            st.metric("Atenciones", f"{len(u)}")

        st.dataframe(
            u[
                [
                    urg_map["patient"],
                    urg_map["arrival_dt"],
                    urg_map["first_doc_dt"],
                    urg_map["discharge_dt"],
                    urg_map["triage_level"],
                    urg_map["dx_group"],
                    "wait_min",
                    "stay_min",
                ]
            ],
            use_container_width=True,
        )

        # Gráfico: espera promedio por triage
        if urg_map["triage_level"] in u.columns:
            grp = (
                u.groupby(urg_map["triage_level"], dropna=False)["wait_min"]
                .mean()
                .reset_index()
                .sort_values("wait_min", ascending=False)
            )
            st.markdown("### Espera promedio por nivel de triage (min)")
            st.bar_chart(data=grp, x=urg_map["triage_level"], y="wait_min", use_container_width=True)

        # Gráfico: top diagnósticos
        if urg_map["dx_group"] in u.columns:
            vol = u[urg_map["dx_group"]].value_counts().head(10).reset_index()
            vol.columns = [urg_map["dx_group"], "casos"]
            st.markdown("### Top 10 diagnósticos en Urgencias")
            st.bar_chart(data=vol, x=urg_map["dx_group"], y="casos", use_container_width=True)


# -------- ECONOMÍA --------
with tab_econ:
    st.header("Indicadores Económicos (opcional)")
    st.caption("Sube una tabla económica para calcular ingresos/costos/margen por GRD.")
    econ_file = st.file_uploader("Tabla Económica (.csv/.xlsx)", type=["csv", "xlsx"], key="econ")
    if econ_file is not None:
        econ_df = pd.read_csv(econ_file) if econ_file.name.lower().endswith(".csv") else pd.read_excel(econ_file)
        st.write("Columnas detectadas:", list(econ_df.columns))
        grd_col     = st.selectbox("Columna GRD en tabla económica", options=list(econ_df.columns))
        ingreso_col = st.selectbox("Columna Ingreso por caso", options=[None] + list(econ_df.columns))
        costo_col   = st.selectbox("Columna Costo por caso",   options=[None] + list(econ_df.columns))

        if colmap.get("dx_group"):
            base = work.rename(columns={colmap["dx_group"]: "_GRD"}).copy()
            econ_tbl = econ_df.rename(
                columns={
                    grd_col: "_GRD",
                    **({ingreso_col: "_ingreso"} if ingreso_col else {}),
                    **({costo_col: "_costo"} if costo_col else {}),
                }
            )
            econ_tbl = econ_tbl[["_GRD"] + [c for c in ["_ingreso", "_costo"] if c in econ_tbl.columns]].drop_duplicates()
            merged = base.merge(econ_tbl, on="_GRD", how="left")
            merged["_ingreso"] = pd.to_numeric(merged.get("_ingreso", 0), errors="coerce").fillna(0)
            merged["_costo"]   = pd.to_numeric(merged.get("_costo",   0), errors="coerce").fillna(0)
            merged["_margen"]  = merged["_ingreso"] - merged["_costo"]

            out = (
                merged.groupby("_GRD", dropna=False)
                .agg(
                    episodios=("_GRD", "size"),
                    ingreso_total=("_ingreso", "sum"),
                    costo_total=("_costo", "sum"),
                    margen=("_margen", "sum"),
                )
                .reset_index()
                .rename(columns={"_GRD": colmap["dx_group"]})
            )
            st.dataframe(out.sort_values("margen", ascending=False), use_container_width=True)

            # Gráfico de barras
            fig, ax = plt.subplots(figsize=(10, 4))
            melt = out.melt(
                id_vars=[colmap["dx_group"]],
                value_vars=["ingreso_total", "costo_total", "margen"],
                var_name="tipo",
                value_name="monto",
            )
            sns.barplot(data=melt, x=colmap["dx_group"], y="monto", hue="tipo", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("Para economía, mapea primero la columna Diagnóstico/GRD en el panel izquierdo.")


# -------- SALUD PÚBLICA --------
with tab_sp:
    st.header("Salud Pública – Tasa de hospitalización por diagnóstico prevalente")
    if colmap.get("dx_group"):
        dx_tbl = work[colmap["dx_group"]].value_counts().reset_index()
        dx_tbl.columns = [colmap["dx_group"], "hospitalizaciones"]
        dx_tbl["tasa_%"] = (dx_tbl["hospitalizaciones"] / len(work) * 100).round(2)
        st.dataframe(dx_tbl, use_container_width=True)

        st.markdown("### Top diagnósticos")
        fig, ax = plt.subplots(figsize=(8, 4))
        top = dx_tbl.head(10)
        sns.barplot(data=top, x="hospitalizaciones", y=colmap["dx_group"], ax=ax)
        ax.set_xlabel("Casos"); ax.set_ylabel(colmap["dx_group"])
        st.pyplot(fig)
    else:
        st.info("Mapea la columna Diagnóstico/GRD para ver esta sección.")

# ================================
# 📤 Exportar reporte HTML completo (con estilo Bootstrap)
# ================================
import base64
from io import BytesIO

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

if ok_to_process:
    st.subheader("📤 Exportar reporte completo en HTML")

    # --- Hospitalización ---
    fig_hosp, ax = plt.subplots(figsize=(7,4))
    sns.histplot(work["LOS_days"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribución de LOS")
    hosp_img = fig_to_base64(fig_hosp)

    hosp_tbl = None
    if group_cols:
        hosp_tbl = work.groupby(group_cols, dropna=False)["LOS_days"].agg(
            ALOS="mean", Mediana="median", Episodios="count"
        ).reset_index().round(2).to_html(index=False, classes="table table-striped table-bordered")

    # --- Urgencia ---
    urg_section = ""
    if 'urg_df' in locals() and urg_df is not None:
        u = urg_df.copy()
        u["_arrive"] = ensure_datetime(u[urg_map["arrival_dt"]])
        u["_first"]  = ensure_datetime(u[urg_map["first_doc_dt"]])
        u["_out"]    = ensure_datetime(u[urg_map["discharge_dt"]])
        u["wait_min"] = (u["_first"] - u["_arrive"]).dt.total_seconds()/60.0
        u["stay_min"] = (u["_out"]   - u["_arrive"]).dt.total_seconds()/60.0

        urg_tbl = u[[urg_map["patient"], urg_map["arrival_dt"],
                     urg_map["first_doc_dt"], urg_map["discharge_dt"],
                     urg_map["triage_level"], urg_map["dx_group"],
                     "wait_min","stay_min"]].head(20).to_html(index=False, classes="table table-striped table-hover")

        fig_urg, ax = plt.subplots(figsize=(7,4))
        sns.histplot(u["wait_min"].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title("Distribución de tiempos de espera (min)")
        urg_img = fig_to_base64(fig_urg)

        urg_section = f"""
        <div class="card my-4">
          <div class="card-body">
            <h2>🚑 Urgencia</h2>
            <p><b>Tiempo de espera medio:</b> {u['wait_min'].mean():.1f} min</p>
            <p><b>Tiempo total medio:</b> {u['stay_min'].mean():.1f} min</p>
            <img class="img-fluid shadow my-3" src="data:image/png;base64,{urg_img}">
            {urg_tbl}
          </div>
        </div>
        """

    # --- Economía ---
    econ_section = ""
    if 'econ_df' in locals() and econ_df is not None and colmap.get("dx_group"):
        econ_section = """
        <div class="card my-4">
          <div class="card-body">
            <h2>💰 Economía</h2>
            <p>Resultados cargados desde tabla económica (solo ejemplo visual).</p>
          </div>
        </div>
        """

    # --- Salud Pública ---
    sp_section = ""
    if colmap.get("dx_group"):
        dx_tbl = work[colmap["dx_group"]].value_counts().reset_index()
        dx_tbl.columns = [colmap["dx_group"], "hospitalizaciones"]
        dx_tbl["tasa_%"] = (dx_tbl["hospitalizaciones"] / len(work) * 100).round(2)

        fig_sp, ax = plt.subplots(figsize=(7,4))
        sns.barplot(data=dx_tbl.head(10), x="hospitalizaciones", y=colmap["dx_group"], ax=ax)
        ax.set_title("Top 10 diagnósticos")
        sp_img = fig_to_base64(fig_sp)

        sp_section = f"""
        <div class="card my-4">
          <div class="card-body">
            <h2>🌍 Salud Pública</h2>
            {dx_tbl.head(20).to_html(index=False, classes="table table-striped")}
            <img class="img-fluid shadow my-3" src="data:image/png;base64,{sp_img}">
          </div>
        </div>
        """

    # --- Ensamblar HTML con Bootstrap ---
    html_report = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>Reporte KPIs Hospitalarios</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="container my-4">
      <h1 class="mb-4">📊 Reporte de KPIs Hospitalarios</h1>
      <p><i>Generado el {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</i></p>

      <div class="card my-4">
        <div class="card-body">
          <h2>🏨 Hospitalización</h2>
          <ul>
            <li><b>ALOS:</b> {work['LOS_days'].mean():.2f} días</li>
            <li><b>Mediana LOS:</b> {work['LOS_days'].median():.2f} días</li>
            <li><b>Total episodios:</b> {len(work)}</li>
          </ul>
          <img class="img-fluid shadow my-3" src="data:image/png;base64,{hosp_img}">
          {hosp_tbl if hosp_tbl is not None else ""}
        </div>
      </div>

      {urg_section}
      {econ_section}
      {sp_section}

    </body>
    </html>
    """

    st.download_button(
        "💾 Descargar reporte HTML con estilo",
        data=html_report,
        file_name="reporte_kpis.html",
        mime="text/html"
    )
