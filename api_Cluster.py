import os
import json
import random
from datetime import date, datetime
from typing import Optional, List

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from database.connection import get_db_connection
from database.init_db import init_db

load_dotenv() 


PIPELINE_FILE = "pipeline_cluster_full.pkl"
CATEGORY_MAP_FILE = "category_map.json"
DATASET_FILE = "dataset_ingredientes_full.csv"


DEFAULT_CATEGORY_MAP = {
    "1": "Verduras",
    "2": "Líquidos",
    "3": "Agua",
    "4": "Sólidos"
}


unidad_tipo = {
    "kilogramos": ("masa", 1000),
    "gramos": ("masa", 1),
    "litros": ("volumen", 1000),
    "mililitros": ("volumen", 1),
    "pieza": ("unidad", 1),
    "paquete": ("unidad", 1),
    "caja": ("unidad", 1)
}

def normalizar_cantidad(unidad: str, cantidad: float):
    unidad = unidad.lower()
    if unidad not in unidad_tipo:
        return "unidad", float(cantidad), "u"
    tipo, factor = unidad_tipo[unidad]
    if tipo == "masa":
        return "masa", float(cantidad) * factor, "g"
    if tipo == "volumen":
        return "volumen", float(cantidad) * factor, "ml"
    return "unidad", float(cantidad), "u"


def asignar_unidad_realista(nombre: str):
    nombre_low = nombre.lower()
    if any(x in nombre_low for x in ["papa","tomate","cebolla","manzana","platano","naranja","zanahoria","lechuga","brocoli","espinaca","pimiento","limon"]):
        return "kilogramos"
    if any(x in nombre_low for x in ["pollo","carne","atun","sardina"]):
        return "kilogramos"
    if any(x in nombre_low for x in ["leche","yogur","queso","mantequilla"]):
        return "litros"
    if any(x in nombre_low for x in ["arroz","frijol","lenteja","azucar","harina","pasta","cacao","cafe"]):
        return "gramos"
    if "lata" in nombre_low:
        return "pieza"
    if any(x in nombre_low for x in ["pan","paquete","caja"]):
        return "paquete"
    return random.choice(["kilogramos","gramos","pieza","paquete","litros"])


def generar_dataset_sintetico(n=500, category_map=None, random_state=123):
    random.seed(random_state)
    np.random.seed(random_state)

    ingredientes_base = [
        "Papa","Cebolla","Tomate","Leche","Huevo","Arroz","Azúcar","Aceite",
        "Pollo","Carne molida","Harina","Mantequilla","Queso","Limón","Plátano",
        "Manzana","Zanahoria","Lechuga","Pasta","Atún","Sardina","Pan","Yogur",
        "Cacao","Café","Pimiento","Ajo","Jengibre","Perejil","Champiñones",
        "Espinaca","Brocoli","Naranja","Frijol","Lenteja"
    ]

    rows = []
    cat_ids = list(category_map.keys()) if category_map else list(DEFAULT_CATEGORY_MAP.keys())

    for _ in range(n):
        name = random.choice(ingredientes_base)
        unidad = asignar_unidad_realista(name)

        if unidad == "kilogramos":
            cantidad_unidad = round(np.random.uniform(0.25, 5.0), 2)
        elif unidad == "gramos":
            cantidad_unidad = int(np.random.choice([100,250,500,1000]))
        elif unidad == "litros":
            cantidad_unidad = round(np.random.uniform(0.25, 3.0), 2)
        elif unidad == "mililitros":
            cantidad_unidad = int(np.random.choice([200,330,500,1000]))
        elif unidad == "pieza":
            cantidad_unidad = int(np.random.choice([1,6,12]))
        elif unidad == "paquete":
            cantidad_unidad = int(np.random.choice([1,2,3,5]))
        else:
            cantidad_unidad = 1

        tipo_base, cantidad_norm, unidad_base = normalizar_cantidad(unidad, cantidad_unidad)
        mu = 80 if tipo_base == "masa" else 60
        cantidad_compras = int(np.random.poisson(mu) + np.random.randint(0, 20))
        tasa_recompra = float(np.round(np.random.beta(3, 2), 3))
        dias_promedio = max(1, int(np.random.normal(10, 4)))
        categoria_id = random.choice(cat_ids)

        rows.append({
            "ingrediente": name,
            "categoria_id": int(categoria_id),
            "unidad_medida": unidad,
            "cantidad_unidad": cantidad_unidad,
            "cantidad_compras": cantidad_compras,
            "tasa_recompra": tasa_recompra,
            "dias_promedio": dias_promedio,
            "tipo_base": tipo_base,
            "cantidad_normalizada": cantidad_norm,
            "unidad_base": unidad_base
        })

    df = pd.DataFrame(rows).sort_values("cantidad_compras", ascending=False).reset_index(drop=True)
    return df


def load_or_init_category_map():
    if os.path.exists(CATEGORY_MAP_FILE):
        with open(CATEGORY_MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(CATEGORY_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CATEGORY_MAP, f, ensure_ascii=False, indent=2)
    return DEFAULT_CATEGORY_MAP


def save_pipeline(pipeline):
    joblib.dump(pipeline, PIPELINE_FILE)

def load_pipeline():
    if not os.path.exists(PIPELINE_FILE):
        return None
    return joblib.load(PIPELINE_FILE)

def entrenar_pipeline(category_map=None, dataset_file=DATASET_FILE, n_default=500):
    category_map = category_map or load_or_init_category_map()

    if os.path.exists(dataset_file):
        df = pd.read_csv(dataset_file)
    else:
        df = generar_dataset_sintetico(n=n_default, category_map=category_map)

    df["categoria_id"] = df["categoria_id"].astype(str)

    cat_features = ["unidad_medida", "categoria_id"]
    num_features = ["cantidad_unidad", "cantidad_normalizada", "tasa_recompra", "dias_promedio"]

    preprocessor = ColumnTransformer([
        ("onehot_cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features)
    ])

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("kmeans", KMeans(n_clusters=4, random_state=42))
    ])

    X = df[cat_features + num_features]
    pipeline.fit(X)

    df["cluster"] = pipeline.named_steps["kmeans"].labels_

    df.to_csv(dataset_file, index=False, encoding="utf-8")
    save_pipeline(pipeline)

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        for _, row in df.iterrows():

            cursor.execute(
                "SELECT id FROM ingredientes WHERE ingrediente = %s AND unidad_medida = %s LIMIT 1",
                (row["ingrediente"], row["unidad_medida"])
            )
            r = cursor.fetchone()

            if r:
                ingrediente_id = r[0]
                cursor.execute("""
                    UPDATE ingredientes
                    SET categoria_id=%s,
                        cantidad_unidad=%s,
                        cantidad_compras=%s,
                        tasa_recompra=%s,
                        dias_promedio=%s,
                        tipo_base=%s,
                        cantidad_normalizada=%s,
                        unidad_base=%s,
                        cluster=%s
                    WHERE id=%s
                """, (
                    int(row["categoria_id"]),
                    float(row["cantidad_unidad"]),
                    int(row["cantidad_compras"]),
                    float(row["tasa_recompra"]),
                    int(row["dias_promedio"]),
                    row["tipo_base"],
                    float(row["cantidad_normalizada"]),
                    row["unidad_base"],
                    int(row["cluster"]),
                    ingrediente_id
                ))
            else:
                cursor.execute("""
                    INSERT INTO ingredientes (
                        ingrediente, categoria_id, unidad_medida, cantidad_unidad,
                        cantidad_compras, tasa_recompra, dias_promedio, tipo_base,
                        cantidad_normalizada, unidad_base, cluster
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING id
                """, (
                    row["ingrediente"],
                    int(row["categoria_id"]),
                    row["unidad_medida"],
                    float(row["cantidad_unidad"]),
                    int(row["cantidad_compras"]),
                    float(row["tasa_recompra"]),
                    int(row["dias_promedio"]),
                    row["tipo_base"],
                    float(row["cantidad_normalizada"]),
                    row["unidad_base"],
                    int(row["cluster"])
                ))

                ingrediente_id = cursor.fetchone()[0]

            hoy = date.today()
            cursor.execute("""
                INSERT INTO historial_clusters_ingredientes (
                    ingrediente_id, ingrediente, fecha, cluster,
                    cantidad_compras, cantidad_normalizada, nota
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                ingrediente_id,
                row["ingrediente"],
                hoy,
                int(row["cluster"]),
                int(row["cantidad_compras"]),
                float(row["cantidad_normalizada"]),
                "initial"
            ))

        conn.commit()

    finally:
        cursor.close()
        conn.close()

    return df, pipeline

    category_map = category_map or load_or_init_category_map()
    if os.path.exists(dataset_file):
        df = pd.read_csv(dataset_file)
    else:
        df = generar_dataset_sintetico(n=n_default, category_map=category_map)

    df["categoria_id"] = df["categoria_id"].astype(str)

    cat_features = ["unidad_medida", "categoria_id"]
    num_features = ["cantidad_unidad", "cantidad_normalizada", "tasa_recompra", "dias_promedio"]

    preprocessor = ColumnTransformer([
        ("onehot_cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features)
    ])

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("kmeans", KMeans(n_clusters=4, random_state=42))
    ])

    X = df[cat_features + num_features]
    pipeline.fit(X)

    df["cluster"] = pipeline.named_steps["kmeans"].labels_

   
    df.to_csv(dataset_file, index=False, encoding="utf-8")
    save_pipeline(pipeline)

    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
       
        for _, row in df.iterrows():
            
            cursor.execute("SELECT id FROM ingredientes WHERE ingrediente = %s AND unidad_medida = %s LIMIT 1",
                           (row["ingrediente"], row["unidad_medida"]))
            r = cursor.fetchone()
            if r:
                ingrediente_id = r[0]
                cursor.execute("""
                    UPDATE ingredientes SET categoria_id=%s, cantidad_unidad=%s, cantidad_compras=%s,
                                          tasa_recompra=%s, dias_promedio=%s, tipo_base=%s,
                                          cantidad_normalizada=%s, unidad_base=%s, cluster=%s
                    WHERE id=%s
                """, (int(row["categoria_id"]), float(row["cantidad_unidad"]), int(row["cantidad_compras"]),
                      float(row["tasa_recompra"]), int(row["dias_promedio"]), row["tipo_base"],
                      float(row["cantidad_normalizada"]), row["unidad_base"], int(row["cluster"]), ingrediente_id))
            else:
                cursor.execute("""
                    INSERT INTO ingredientes (ingrediente, categoria_id, unidad_medida, cantidad_unidad,
                                              cantidad_compras, tasa_recompra, dias_promedio, tipo_base,
                                              cantidad_normalizada, unidad_base, cluster)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (row["ingrediente"], int(row["categoria_id"]), row["unidad_medida"], float(row["cantidad_unidad"]),
                      int(row["cantidad_compras"]), float(row["tasa_recompra"]), int(row["dias_promedio"]),
                      row["tipo_base"], float(row["cantidad_normalizada"]), row["unidad_base"], int(row["cluster"])))
                ingrediente_id = cursor.lastrowid

          
            hoy = date.today()
            cursor.execute("""
                INSERT INTO historial_clusters_ingredientes (ingrediente_id, ingrediente, fecha, cluster,
                                                             cantidad_compras, cantidad_normalizada, nota)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (ingrediente_id, row["ingrediente"], hoy, int(row["cluster"]), int(row["cantidad_compras"]),
                  float(row["cantidad_normalizada"]), "initial"))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return df, pipeline


class ProductoInput(BaseModel):
    ingrediente: str
    categoria_id: int
    unidad_medida: str
    cantidad_unidad: float
    cantidad_compras: int
    tasa_recompra: float
    dias_promedio: int

class PredictResponse(BaseModel):
    ingrediente: str
    cluster: int
    etiqueta: str
    sugerido: str


app = FastAPI(title="API Clustering Ingredientes", version="1.0")
app.add_middleware(CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/train")
def train_endpoint():
    category_map = load_or_init_category_map()
    df, pipeline = entrenar_pipeline(category_map)
    return {"status": "trained", "n_items": len(df)}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: ProductoInput):
    pipeline = load_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline no encontrado. Ejecuta /train primero.")

    tipo_base, cantidad_norm, unidad_base = normalizar_cantidad(payload.unidad_medida, payload.cantidad_unidad)

    X_row = pd.DataFrame([{
        "unidad_medida": payload.unidad_medida,
        "categoria_id": str(payload.categoria_id),
        "cantidad_unidad": payload.cantidad_unidad,
        "cantidad_normalizada": cantidad_norm,
        "tasa_recompra": payload.tasa_recompra,
        "dias_promedio": payload.dias_promedio
    }])

    cluster = int(pipeline.predict(X_row)[0])

    
    df_ref = pd.read_csv(DATASET_FILE) if os.path.exists(DATASET_FILE) else pd.DataFrame()
    etiqueta = "BAJA PRIORIDAD"
    if not df_ref.empty and "cluster" in df_ref.columns:
        summary = df_ref.groupby("cluster")["cantidad_normalizada"].mean().sort_values(ascending=False)
        rank = summary.index.tolist()
        if cluster == rank[0]:
            etiqueta = "ALTA PRIORIDAD (demanda alta)"
        elif len(rank) > 1 and cluster == rank[1]:
            etiqueta = "PRIORIDAD MEDIA"


    sugerido = f"{payload.cantidad_unidad} {payload.unidad_medida}"

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
       
        cursor.execute("SELECT id FROM ingredientes WHERE ingrediente = %s AND unidad_medida = %s LIMIT 1",
                       (payload.ingrediente, payload.unidad_medida))
        r = cursor.fetchone()
        if r:
            ingrediente_id = r[0]
            cursor.execute("UPDATE ingredientes SET categoria_id=%s, cantidad_unidad=%s, cantidad_compras=%s, tasa_recompra=%s, dias_promedio=%s, cantidad_normalizada=%s, cluster=%s WHERE id=%s",
                           (payload.categoria_id, payload.cantidad_unidad, payload.cantidad_compras, payload.tasa_recompra, payload.dias_promedio, cantidad_norm, cluster, ingrediente_id))
        else:
            cursor.execute("""INSERT INTO ingredientes (ingrediente, categoria_id, unidad_medida, cantidad_unidad,
                                                      cantidad_compras, tasa_recompra, dias_promedio, tipo_base,
                                                      cantidad_normalizada, unidad_base, cluster)
                              VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                           (payload.ingrediente, payload.categoria_id, payload.unidad_medida, payload.cantidad_unidad,
                            payload.cantidad_compras, payload.tasa_recompra, payload.dias_promedio, tipo_base,
                            cantidad_norm, unidad_base, cluster))
            ingrediente_id = cursor.lastrowid

      
        hoy = date.today()
        cursor.execute("""INSERT INTO historial_clusters_ingredientes (ingrediente_id, ingrediente, fecha, cluster, cantidad_compras, cantidad_normalizada, nota)
                          VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                       (ingrediente_id, payload.ingrediente, hoy, cluster, payload.cantidad_compras, cantidad_norm, "predict"))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return PredictResponse(
        ingrediente=payload.ingrediente,
        cluster=cluster,
        etiqueta=etiqueta,
        sugerido=sugerido
    )

@app.get("/recluster")
def recluster_endpoint():
    category_map = load_or_init_category_map()
    df, pipeline = entrenar_pipeline(category_map)
    return {"status": "reclustered", "n_items": len(df)}

@app.get("/dataset")
def dataset_endpoint():
    if os.path.exists(DATASET_FILE):
        df = pd.read_csv(DATASET_FILE)
        return {"n_items": len(df), "sample": df.head(10).to_dict(orient="records")}
    return {"n_items": 0, "sample": []}

@app.get("/evolucion/ingrediente/{ingrediente}")
def evolucion_ingrediente(ingrediente: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT fecha, cluster, cantidad_compras, cantidad_normalizada
            FROM historial_clusters_ingredientes
            WHERE ingrediente = %s
            ORDER BY fecha ASC
        """, (ingrediente,))
        rows = cursor.fetchall()
        historial = [{"fecha": r[0].strftime("%Y-%m-%d"), "cluster": int(r[1]), "cantidad_compras": int(r[2]), "cantidad_normalizada": float(r[3])} for r in rows]
    finally:
        cursor.close()
        conn.close()
    return {"ingrediente": ingrediente, "historial": historial}
