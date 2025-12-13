from database.connection import get_db_connection

def init_db():
    print("🔌 Intentando conectar a la base de datos...")

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        print("📦 Creando tablas si no existen...")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ingredientes (
            id SERIAL PRIMARY KEY,
            ingrediente VARCHAR(255),
            categoria_id INT,
            unidad_medida VARCHAR(50),
            cantidad_unidad DOUBLE PRECISION,
            cantidad_compras INT,
            tasa_recompra DOUBLE PRECISION,
            dias_promedio INT,
            tipo_base VARCHAR(20),
            cantidad_normalizada DOUBLE PRECISION,
            unidad_base VARCHAR(10),
            cluster INT,
            creado_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS historial_clusters_ingredientes (
            id SERIAL PRIMARY KEY,
            ingrediente_id INT REFERENCES ingredientes(id),
            ingrediente VARCHAR(255),
            fecha DATE,
            cluster INT,
            cantidad_compras INT,
            cantidad_normalizada DOUBLE PRECISION,
            nota VARCHAR(255),
            creado_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        conn.commit()
        print("✅ Tablas verificadas / creadas correctamente")

    finally:
        cursor.close()
        conn.close()
        print("🔒 Conexión cerrada")