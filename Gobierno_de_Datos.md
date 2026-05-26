# Gobierno de Datos — Proyecto VIF Colombia 2015–2024

**Dataset:** `V1.csv` · 236,840 registros · 36 columnas  
**Fuente:** Instituto Nacional de Medicina Legal y Ciencias Forenses (INMLCF)  
**Proyecto:** Modelos Predictivos sobre Violencia Intrafamiliar en Colombia  
**Fecha:** Mayo 2026

---

## 1. Dueños de los datos

### Propietario institucional primario

El dataset proviene del **INMLCF — Instituto Nacional de Medicina Legal y Ciencias Forenses**, entidad adscrita a la Fiscalía General de la Nación de Colombia. El INMLCF es la autoridad técnico-científica en Colombia para la recolección y publicación de estadísticas forenses, incluyendo violencia intrafamiliar.

| Rol | Entidad | Responsabilidad |
|---|---|---|
| **Dueño primario (Data Owner)** | INMLCF | Recolecta, valida, y publica los datos forenses de VIF. Define qué se registra y cómo. |
| **Sistema de recolección** | SIEDCO (Sistema de Información Estadístico Delincuencial, Contravencional y Operativo) | Plataforma técnica donde se ingresan los registros de los casos |
| **Distribuidor oficial** | [datos.gov.co](https://www.datos.gov.co) — Portal de Datos Abiertos de Colombia | Publica el dataset para acceso público bajo Ley 1712/2014 |
| **Custodio académico** | Universidad — Equipo del proyecto | Responsable del uso ético, reproducible y no comercial de los datos para investigación académica |
| **Responsable de análisis** | Estudiantes del proyecto | Preprocesamiento, modelado y comunicación de resultados |

### Marco legal aplicable

- **Ley 1712 de 2014** (Transparencia y Acceso a la Información Pública): el dataset es de dominio público, acceso libre y gratuito.
- **Ley 1581 de 2012** (Protección de Datos Personales — equivalente colombiano al GDPR): aunque el dataset fue anonimizado por el INMLCF, contiene variables sensibles (género, discapacidad, menores) que requieren manejo cuidadoso.
- **Convención sobre los Derechos del Niño (ONU)**: los registros de víctimas menores de edad tienen protección especial; no deben usarse para identificación individual.

---

## 2. Organización de los datos dentro del modelo

### Diagrama de flujo de datos

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FUENTE: INMLCF / SIEDCO                               │
│  Médicos forenses registran cada caso de VIF atendido en Colombia        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ARCHIVO RAW: V1.csv                                   │
│  • 236,840 filas  • 36 columnas  • Separador: ;  • Encoding: UTF-8      │
│  • No hay NaN reales; los nulos son: "Sin información", "No aplica",     │
│    "No Sabe / No Informa"                                                │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 PREPROCESAMIENTO (Paso 3 del pipeline)                   │
│                                                                          │
│  Columnas originales (texto)        Columnas derivadas (numéricas)       │
│  ─────────────────────────          ──────────────────────────────       │
│  Días de Incapacidad Medicolegal ──▶ severidad_n (0=Leve, 1=Mod, 2=Gr.) │
│  Escolaridad                    ──▶ edu_n (0–6, ordinal)                │
│  Sexo de la victima             ──▶ sexo_n (binaria)                    │
│  Ciclo Vital                    ──▶ ciclo_n (0–5, ordinal)              │
│  Grupo Mayor Menor de Edad      ──▶ es_menor (binaria)                  │
│  Tipo de Discapacidad           ──▶ tiene_discapacidad (binaria)         │
│  Estado Civil                   ──▶ civil_n (0–4, ordinal)              │
│  Rango de Hora del Hecho        ──▶ hora_n (0–7) + es_noche (binaria)   │
│  Zona del Hecho                 ──▶ zona_n (binaria)                    │
│  Escenario del Hecho            ──▶ escenario_n (LabelEncoder top-8)    │
│  Presunto Agresor Detallado     ──▶ agresor_n (1–5, ordinal)            │
│  Factor Desencadenante          ──▶ factor_n (1–3, ordinal)             │
│  Mecanismo Causal               ──▶ mecanismo_n (1–3, ordinal)          │
│  Mes del hecho + Año            ──▶ mes_sin, mes_cos, t, lag_1/2/3     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌────────────┐  ┌────────────┐  ┌─────────────┐
       │   df_q2    │  │   df_q3    │  │   ts_nn     │
       │ FEATS_Q2   │  │ FEATS_Q3   │  │ FEATS_Q1_NN │
       │ (n~203k)   │  │ (n~150k)   │  │ (~117 pts)  │
       └─────┬──────┘  └─────┬──────┘  └──────┬──────┘
             │               │                │
             ▼               ▼                ▼
       ┌───────────────────────────────────────────────┐
       │              MODELOS PREDICTIVOS              │
       │  Clásicos: RF, SVM, KNN, Regresión Logística  │
       │  Redes neuronales: Keras (backend JAX)         │
       └────────────────────┬──────────────────────────┘
                            │
                            ▼
       ┌───────────────────────────────────────────────┐
       │               SALIDAS                         │
       │  • Métricas: F1, R², MAE, RMSE               │
       │  • Gráficas: graficas/*.png                   │
       │  • Importancia de features                    │
       └───────────────────────────────────────────────┘
```

### Categorización de las 36 columnas

| Categoría | Columnas | Uso en el modelo |
|---|---|---|
| **Identificadores** | `ID`, `Código Dane Municipio`, `Código Dane Departamento` | Ninguno (llave primaria, sin valor predictivo) |
| **Temporales** | `Año del hecho`, `Mes del hecho`, `Dia del hecho`, `Rango de Hora del Hecho` | Q1 (regresión temporal), Q3 (hora_n, es_noche) |
| **Perfil víctima** | `Sexo de la victima`, `Grupo de edad quinquenal`, `Grupo Mayor Menor de Edad`, `Ciclo Vital`, `Escolaridad`, `Estado Civil`, `Tipo de Discapacidad` | Q2 (perfil víctima) |
| **Geográficas** | `Municipio del hecho DANE`, `Departamento del hecho DANE`, `Localidad del Hecho`, `Zona del Hecho` | zona_n (Q3); Departamento sin explotar |
| **Contextuales del hecho** | `Escenario del Hecho`, `Actividad Durante el Hecho`, `Circunstancia del Hecho Detallada`, `Contexto del Hecho` | Q3 (escenario_n) |
| **Agresor** | `Sexo del Agresor`, `Presunto Agresor Detallado` | Q3 (agresor_n) |
| **Mecanismo** | `Mecanismo Causal de la Lesión no Fatal`, `Diagnostico Topográfico de la Lesión no Fatal`, `Factor Desencadenante de la Agresión` | Q3 (mecanismo_n, factor_n) |
| **Target** | `Días de Incapacidad Medicolegal` | Derivado como `severidad_n` (Q2, Q3) |
| **Alta cobertura de nulos** | `Orientación Sexual` (22%), `Identidad de Género` (79%), `Transgénero` (99%), `Localidad del Hecho` (79%), `Pueblo Indígena` (57%) | No usadas como features por alta tasa de valores faltantes |
| **Étnicas/identidad** | `Pertenencia Étnica`, `Pertenencia Grupal`, `País de Nacimiento`, `Grupo de Edad judicial` | No usadas en modelos actuales |

---

## 3. Distribución de los datos (Pasos 2 y 3 del EDA)

### Paso 2 — Distribución del target y valores faltantes

**Distribución real de `Días de Incapacidad Medicolegal`:**

| Valor original | Categoría (severidad_n) | N casos | % del total |
|---|---|---|---|
| "1 a 30" | 1 — Moderado | 182,400 | 77.0% |
| "Cero días" | 0 — Leve | 11,202 | 4.7% |
| "Sin días de incapacidad" | 0 — Leve | 4,983 | 2.1% |
| "31 a 90" | 2 — Grave | 4,792 | 2.0% |
| "Cero" | 0 — Leve | 41 | 0.0% |
| "Más de 90" | 2 — Grave | 118 | 0.1% |
| "Sin información" / "Cero días y sin información" | NaN (excluido) | 33,304 | 14.1% |

**Distribución del target entre casos válidos (203,536 registros):**
- Leve (0): 16,226 → **8.0%**
- Moderado (1): 182,400 → **89.6%**
- Grave (2): 4,910 → **2.4%**

> **Implicación:** La clase dominante es "Moderado" (~90%). Un modelo naive que prediga siempre "Moderado" tendría 89.6% de accuracy sin aprender nada. Esto justifica el uso de F1 ponderado y pesos de clase en lugar de accuracy simple.

**Variables con alta tasa de valores faltantes (>20%):**

| Variable | % Sin información | Impacto |
|---|---|---|
| Transgénero | 99.4% | No usable |
| Localidad del Hecho | 78.7% | No usable |
| Identidad de Género | 78.5% | No usable |
| Pueblo Indígena | 56.7% | Descartada |
| Orientación Sexual | 21.6% | Descartada |

### Paso 3 — Ingeniería de características

La conversión de texto a números sigue estas estrategias:

**Codificación ordinal:** Para variables con orden natural establecido.

| Feature | Escala | Lógica |
|---|---|---|
| `edu_n` | 0–6 | Sin escolaridad → Doctorado (nivel de protección creciente) |
| `agresor_n` | 1–5 | Familiar lejano (1) → Pareja íntima (5): vínculo determina severidad |
| `ciclo_n` | 0–5 | Primera infancia (0) → Adulto mayor (5) |
| `factor_n` | 1–3 | Económico/intolerancia (1) → Alcohol/drogas (3): riesgo creciente |
| `mecanismo_n` | 1–3 | Abrasivo (1) → Cortante (3): letalidad creciente |

**Codificación cíclica:** Para variables circulares como los meses.
- Sin este encoding, diciembre (12) y enero (1) parecen extremos opuestos cuando son consecutivos.
- `mes_sin = sin(2π × mes/12)` y `mes_cos = cos(2π × mes/12)` los ubican en el círculo anual correctamente.

**Lag features (solo Q1):** Para que la red neuronal tenga memoria temporal.
- `lag_1`, `lag_2`, `lag_3`: casos de los meses anteriores
- `rolling3`: media móvil de los últimos 3 meses

---

## 4. Propuesta de almacenamiento en base de datos

### Servicio recomendado: PostgreSQL en Amazon RDS (o Google Cloud SQL)

**Justificación:**
- Los datos son **relacionales y estructurados**: tablas con esquema fijo → PostgreSQL es la opción natural.
- **Escala:** 236k registros caben holgadamente en una instancia pequeña (db.t3.micro); el costo anual estimado es <$200 USD.
- **Compatibilidad:** PostgreSQL tiene conectores nativos para Python (psycopg2), R, y Power BI.
- **Backup automático:** RDS ofrece backups diarios con retención de 7 días sin configuración adicional.

### Esquema propuesto

```sql
-- Tabla principal: un registro por caso
CREATE TABLE vif_hechos (
    id                    BIGINT PRIMARY KEY,
    ano_hecho             SMALLINT NOT NULL,
    mes_hecho             VARCHAR(20),
    dia_hecho             VARCHAR(20),
    rango_hora            VARCHAR(30),
    sexo_victima          VARCHAR(20),
    ciclo_vital           VARCHAR(40),
    escolaridad           VARCHAR(60),
    estado_civil          VARCHAR(40),
    tipo_discapacidad     VARCHAR(60),
    zona_hecho            VARCHAR(30),
    escenario_hecho       VARCHAR(60),
    municipio_dane        VARCHAR(60),
    departamento_dane     VARCHAR(40),
    presunto_agresor      VARCHAR(60),
    factor_desencadenante VARCHAR(60),
    mecanismo_causal      VARCHAR(60),
    dias_incapacidad      VARCHAR(40),
    -- ... resto de columnas
    fecha_carga           TIMESTAMP DEFAULT NOW()
) PARTITION BY RANGE (ano_hecho);

-- Particiones por año (mejora rendimiento en consultas temporales)
CREATE TABLE vif_hechos_2015 PARTITION OF vif_hechos FOR VALUES FROM (2015) TO (2016);
CREATE TABLE vif_hechos_2016 PARTITION OF vif_hechos FOR VALUES FROM (2016) TO (2017);
-- ... hasta 2024

-- Tabla de features procesadas (listas para modelado)
CREATE TABLE vif_features (
    id             BIGINT REFERENCES vif_hechos(id),
    severidad_n    SMALLINT,   -- target: 0, 1, 2
    edu_n          SMALLINT,
    sexo_n         REAL,
    es_menor       SMALLINT,
    ciclo_n        SMALLINT,
    tiene_discapacidad SMALLINT,
    civil_n        SMALLINT,
    hora_n         SMALLINT,
    es_noche       SMALLINT,
    zona_n         SMALLINT,
    escenario_n    SMALLINT,
    agresor_n      SMALLINT,
    factor_n       SMALLINT,
    mecanismo_n    SMALLINT,
    PRIMARY KEY (id)
);

-- Tabla de predicciones del modelo
CREATE TABLE modelo_predicciones (
    pred_id         SERIAL PRIMARY KEY,
    hecho_id        BIGINT REFERENCES vif_hechos(id),
    modelo          VARCHAR(50),    -- 'random_forest', 'keras_mlp', etc.
    pred_severidad  SMALLINT,       -- predicción (0, 1, 2)
    prob_leve       REAL,           -- probabilidad clase 0
    prob_moderado   REAL,           -- probabilidad clase 1
    prob_grave      REAL,           -- probabilidad clase 2
    timestamp_pred  TIMESTAMP DEFAULT NOW()
);

-- Índices para mejorar rendimiento en consultas frecuentes
CREATE INDEX idx_vif_ano ON vif_hechos(ano_hecho);
CREATE INDEX idx_vif_dpto ON vif_hechos(departamento_dane);
CREATE INDEX idx_vif_severidad ON vif_features(severidad_n);
```

### Pipeline de actualización

```
Nuevo reporte INMLCF (mensual/anual)
         │
         ▼
  Validación automática
  (verificar formato, rangos, columnas)
         │
         ▼
  Carga a tabla staging (sin tocar producción)
         │
         ▼
  Ejecución de preprocesamiento
  (generar vif_features desde staging)
         │
         ▼
  Re-entrenamiento del modelo
  (si han pasado ≥3 meses de datos nuevos)
         │
         ▼
  Promoción a producción + notificación
```

---

## 5. Problemas potenciales de seguridad

### Datos sensibles identificados

| Variable | Tipo de sensibilidad | Riesgo | Mitigación |
|---|---|---|---|
| `Sexo de la victima`, `Identidad de Género`, `Orientación Sexual` | Dato personal sensible (Ley 1581) | Re-identificación + discriminación | Agregación; nunca publicar a nivel individual |
| `Grupo Mayor Menor de Edad` | Dato de menor de edad | Protección especial (Conv. Derechos del Niño) | Acceso restringido; no publicar registros de menores |
| `Municipio del hecho`, `Localidad` | Geolocalización | Triangulación para re-identificar a víctima o agresor | Usar solo a nivel de departamento en publicaciones |
| `Días de Incapacidad Medicolegal` | Dato médico/forense | Revela estado de salud | Solo usar en análisis agregado |
| `Presunto Agresor Detallado` | Información sobre tercero | Presunción de inocencia; dato no definitivo | Anonimizar antes de cualquier exposición pública |
| `Diagnóstico Topográfico de la Lesión` | Dato médico detallado | Altamente sensible | Acceso solo para investigadores autorizados |

### Riesgos de seguridad informática

1. **Acceso no autorizado al CSV:** El archivo `V1.csv` contiene 236k registros individuales. Si se expone en un repositorio público (GitHub), viola la privacidad de las víctimas aunque hayan sido "anonimizadas" por el INMLCF.
   - **Mitigación:** `.gitignore` incluye `V1.csv`; el archivo no se sube al repositorio. ✓ (ya implementado)

2. **Re-identificación por combinación de variables:** Combinar municipio + día + hora + agresor puede identificar un caso específico en municipios pequeños.
   - **Mitigación:** Nunca publicar datos a nivel de fila individual; solo resultados agregados.

3. **SQL injection en la base de datos propuesta:** Si se construye una API sobre la BD, las queries deben usar parámetros preparados.
   - **Mitigación:** Usar ORMs (SQLAlchemy) o consultas parametrizadas; nunca concatenar strings de usuario en queries SQL.

4. **Exposición de credenciales de BD:** Claves de acceso a AWS RDS nunca deben estar en el código fuente.
   - **Mitigación:** Variables de entorno (`.env`) + AWS Secrets Manager para producción.

5. **Transferencia de datos sin cifrado:** Si se transfiere el CSV por email o servicios no seguros.
   - **Mitigación:** Solo transferir via SFTP o S3 con SSE (Server-Side Encryption).

### Clasificación del dato

Siguiendo el marco de clasificación de información de Colombia:
- **Nivel:** Información pública con restricciones de uso (no comercial, no re-identificación)
- **Acceso:** Investigadores académicos con protocolo de uso ético firmado

---

## 6. Propuesta de limpieza de datos

### Estado ANTES de la limpieza

```
Dataset bruto V1.csv: 236,840 filas × 36 columnas

Problemas identificados:
├── Valores pseudonulos: "Sin información", "No aplica", "No Sabe / No Informa"
│   (representan datos faltantes pero NO son NaN reales)
│   Ejemplo: df['Orientación Sexual'].value_counts()
│     Sin información    178,741  (75.5%)
│     Heterosexual        41,032  (17.3%)
│     Bisexual             9,204   (3.9%)
│     Homosexual           7,863   (3.3%)
│
├── Inconsistencias en strings: "Cero días" vs "Cero días y sin información"
│   vs "Cero días y Sin información" (diferente mayúscula)
│
├── Variables ordinales como texto libre (no comparables entre sí):
│   "Básica primaria" vs "Educación básica primaria" → mismo nivel, texto diferente
│
└── Outliers: 'Año del hecho' fuera del rango 2015-2024 (si los hay)
```

**Ejemplo de columna problemática — `Escolaridad`:**

| Valor en CSV | Interpretación correcta |
|---|---|
| "Ninguna" | Sin escolaridad |
| "Sin escolaridad" | Sin escolaridad (duplicado semántico) |
| "No aplica" | Pseudonulo |
| "Sin información" | Pseudonulo |
| "Básica primaria" | Primaria |
| "Educación básica primaria" | Primaria (texto diferente, mismo nivel) |

### Estado DESPUÉS de la limpieza

```
Dataset limpio df_clean: 236,840 filas × 36 columnas
  + columnas numéricas derivadas: severidad_n, edu_n, sexo_n, etc.

Transformaciones aplicadas:
├── Pseudonulos → NaN: reemplazar "Sin información", "No aplica", etc. por np.nan
│   Antes: df['Escolaridad'].value_counts()['Sin información'] = 14,293
│   Después: df_clean['Escolaridad'].isna().sum() = 14,293 (ahora es NaN real)
│
├── Consolidación de valores equivalentes:
│   "Básica primaria" y "Educación básica primaria" → edu_n = 2
│   "Cero días y Sin información" y "Cero días y sin información" → NaN
│
├── Creación de features numéricas (ver Paso 3):
│   edu_n, sexo_n, ciclo_n, severidad_n, hora_n, agresor_n...
│
└── Tratamiento de outliers temporales:
│   Verificación de rango: df['Año del hecho'].between(2015, 2024).all()
│   Resultado: True (sin outliers temporales)
```

**Comparativa antes/después para el dataset de modelado:**

| Métrica | Antes | Después |
|---|---|---|
| Registros totales | 236,840 | 236,840 (sin eliminar filas) |
| NaN reales en target | 0 | 33,304 (antes eran strings; ahora NaN) |
| Registros usables para Q2 | ~100k (estimado) | 203,536 (con NaN explícitos) |
| Valores "Sin información" en Escolaridad | 14,293 strings | 14,293 NaN (excluidos del modelo) |
| Inconsistencias de texto en Escolaridad | 2 formas distintas para "primaria" | 1 sola categoría (edu_n = 2) |
| Tipo de dato de Escolaridad | `object` (texto) | `float64` (edu_n, 0–6) |

### Regla de oro: No eliminar del dataset original

Los registros con valores faltantes se excluyen **solo para el modelado** (`df.dropna(subset=FEATS + [TARGET])`), pero se mantienen en el dataset original `df` para conteos estadísticos generales (p.ej., el total de casos por año no debe excluir registros por falta de escolaridad).

---

## 7. Propuesta de maximizar efectividad mediante integración con Power BI

### Arquitectura de integración

```
┌──────────────────────┐     ┌─────────────────────┐     ┌────────────────────┐
│  PostgreSQL en RDS   │────▶│  Power BI Desktop    │────▶│  Power BI Service  │
│  (base de datos)     │     │  (modelado + DAX)    │     │  (publicación)     │
│                      │     │                      │     │  Dashboards en la  │
│  vif_hechos          │     │  Tablas importadas   │     │  web para entidades│
│  vif_features        │     │  + medidas DAX       │     │  gubernamentales   │
│  modelo_predicciones │     │                      │     └────────────────────┘
└──────────────────────┘     └─────────────────────┘
```

### Dashboards propuestos

**Dashboard 1 — Mapa geográfico de incidencia**
- Mapa de Colombia con intensidad de casos por departamento
- Filtros: año, mes, tipo de agresor, severidad
- KPIs: casos totales, % grave, departamento con mayor incidencia

**Dashboard 2 — Serie temporal y predicciones**
- Gráfico de línea: casos mensuales históricos (2015–2024) + predicción del modelo para los próximos 3 meses
- Bandas de confianza del modelo
- Marcador visual del efecto pandemia (2020–2021)

**Dashboard 3 — Perfil de víctimas**
- Distribución por sexo, edad, escolaridad, estado civil
- Treemap de circunstancias del hecho
- Comparativa por período (pre-pandemia vs. post-pandemia)

**Dashboard 4 — Alertas operativas (alta severidad)**
- Lista de combinaciones de alto riesgo (escenario + agresor + mecanismo) ordenadas por probabilidad de gravedad
- Umbral configurable: alertar cuando predicción de "Grave" supera X%
- Exportar a PDF automáticamente cada mes

### Integración del modelo ML en Power BI

Power BI soporta Python y R scripts directamente en el flujo de transformación de datos (Power Query):

```python
# Script Python dentro de Power BI (Power Query → Transformar → Ejecutar script Python)
import pandas as pd
import pickle

# Cargar modelo entrenado (guardado previamente con joblib/pickle)
with open('modelo_rf_q3.pkl', 'rb') as f:
    modelo = pickle.load(f)

# 'dataset' es el DataFrame que Power BI pasa al script
X = dataset[['hora_n', 'escenario_n', 'agresor_n', 'factor_n', 'mecanismo_n',
             'zona_n', 'es_noche']]

# Predicción
dataset['pred_severidad'] = modelo.predict(X)
dataset['prob_grave'] = modelo.predict_proba(X)[:, 2]  # P(grave)
```

### Beneficios esperados de la integración

| Beneficio | Impacto |
|---|---|
| Visualización accesible para no-técnicos | Funcionarios del ICBF o Fiscalía pueden explorar los datos sin código |
| Actualización automática | Con Direct Query a la BD, los dashboards se actualizan al cargar nuevos datos |
| Alertas proactivas | Notificaciones cuando la probabilidad de casos graves supera un umbral |
| Reportes regulatorios | Exportar a PDF/Excel en formato requerido por entidades de control |
| Democratización del análisis | Permite que más áreas de gobierno usen los modelos sin necesitar Python |

---

## 8. (OPCIONAL) Propuesta de gestión de documentos relacionados

### Sistema de control de versiones para datos

Dado que el CSV puede actualizarse anualmente con nuevos reportes del INMLCF, se propone el siguiente esquema de gestión documental:

**Git LFS (Large File Storage) o S3 versionado:**
```
datos/
├── V1.csv          → Datos 2015-2024 (versión actual)
├── V1_2023.csv     → Versión anterior (archivada)
└── CHANGELOG_DATOS.md
```

**Registro de versiones del dataset:**

| Versión | Fecha | Registros | Cambios |
|---|---|---|---|
| V1.0 | 2024-01 | 210,500 | Dataset inicial (2015-2023) |
| V1.1 | 2025-01 | 236,840 | Añadidos datos 2024 |
| V1.2 (propuesta) | 2026-01 | ~260,000 | Añadir datos 2025 |

**Metadata registry:** Un archivo `metadata.json` que acompañe cada versión:

```json
{
  "version": "V1.1",
  "fecha_creacion": "2025-01-15",
  "fuente": "INMLCF / datos.gov.co",
  "url_descarga": "https://www.datos.gov.co/...",
  "registros": 236840,
  "columnas": 36,
  "rango_temporal": "2015-2024",
  "checksum_md5": "a3f8b2c1...",
  "responsable": "INMLCF - División de Estadística",
  "licencia": "Datos Abiertos - Ley 1712/2014",
  "notas": "Los registros de 2024 son preliminares y pueden variar en la siguiente versión"
}
```

---

## 9. N/A

*Este ítem no aplica al presente proyecto.*

---

## 10. Informe de metadatos

### Descripción de cada campo del dataset

| # | Columna | Origen | Formato | Uso en modelo | Propósito |
|---|---|---|---|---|---|
| 1 | `ID` | SIEDCO (sistema INMLCF) | Entero único | No (llave primaria) | Identificador único del caso forense |
| 2 | `Año del hecho` | Registro forense | Entero (2015–2024) | Q1 (serie temporal) | Año calendario en que ocurrió el hecho |
| 3 | `Sexo de la victima` | Evaluación forense | Texto (Mujer/Hombre/Intersexual) | Q2 (sexo_n) | Sexo biológico registrado de la víctima |
| 4 | `Grupo de edad quinquenal` | Cálculo desde fecha nacimiento | Texto ("(20 a 24)", etc.) | No directamente | Rango quinquenal de edad |
| 5 | `Grupo Mayor Menor de Edad` | Derivado de edad | Texto (Mayor/Menor) | Q2 (es_menor) | Clasifica si la víctima es menor de 18 años |
| 6 | `Grupo de Edad judicial` | Derivado de edad | Texto | No | Clasificación legal por edad |
| 7 | `Ciclo Vital` | Clasificación del ICBF | Texto (Primera infancia, Adolescencia...) | Q2 (ciclo_n) | Etapa del ciclo vital de la víctima |
| 8 | `País de Nacimiento` | Declaración de la víctima | Texto | No (baja varianza) | País donde nació la víctima |
| 9 | `Escolaridad` | Declaración de la víctima | Texto (Ninguna → Doctorado) | Q2 (edu_n) | **Feature principal Q2** — nivel educativo alcanzado |
| 10 | `Estado Civil` | Declaración de la víctima | Texto (Soltero, Casado...) | Q2 (civil_n) | Estado civil al momento del hecho |
| 11 | `Tipo de Discapacidad` | Evaluación médico-forense | Texto | Q2 (tiene_discapacidad) | Discapacidad reportada o ausencia de ella |
| 12 | `Pertenencia Étnica` | Autodeclaración | Texto | No | Grupo étnico al que pertenece la víctima |
| 13 | `Orientación Sexual` | Autodeclaración | Texto | No (22% sin info) | Orientación sexual declarada |
| 14 | `Identidad de Género` | Autodeclaración | Texto | No (79% sin info) | Identidad de género declarada |
| 15 | `Transgénero` | Autodeclaración | Texto | No (99% sin info) | Si la persona es transgénero |
| 16 | `Pertenencia Grupal` | Registro institucional | Texto | No | Pertenencia a grupos especiales (FFMM, policía, etc.) |
| 17 | `Mes del hecho` | Registro forense | Texto (Enero–Diciembre) | Q1 (mes_sin, mes_cos) | Mes en que ocurrió el hecho (encoding cíclico) |
| 18 | `Dia del hecho` | Registro forense | Texto (lunes–domingo) | No utilizado aún | Día de la semana del hecho |
| 19 | `Rango de Hora del Hecho X 3 Horas` | Registro forense | Texto ("(0:00 a 2:59)", etc.) | Q3 (hora_n, es_noche) | Rango horario de 3 horas en que ocurrió el hecho |
| 20 | `Código Dane Municipio` | DANE | Entero (código oficial) | No (reemplazado por nombre) | Código único del municipio según DANE |
| 21 | `Municipio del hecho DANE` | DANE | Texto | No directamente | Nombre del municipio donde ocurrió el hecho |
| 22 | `Departamento del hecho DANE` | DANE | Texto (33 departamentos) | No (potencial futuro) | Departamento — alta cardinalidad, requiere encoding especial |
| 23 | `Código Dane Departamento` | DANE | Entero | No | Código oficial del departamento |
| 24 | `Localidad del Hecho` | Registro forense (Bogotá principalmente) | Texto | No (79% sin info) | Localidad dentro del municipio |
| 25 | `Zona del Hecho` | Registro forense | Texto (Cabecera/Rural) | Q3 (zona_n) | Zona urbana o rural del hecho |
| 26 | `Escenario del Hecho` | Registro forense | Texto (Vivienda, Vía pública...) | Q3 (escenario_n) | **Feature clave Q3** — lugar físico donde ocurrió |
| 27 | `Actividad Durante el Hecho` | Registro forense | Texto | No utilizado | Qué hacía la víctima cuando ocurrió el hecho |
| 28 | `Circunstancia del Hecho Detallada` | Relato médico-forense | Texto | No (texto libre, NLP potencial) | Descripción de la circunstancia del caso |
| 29 | `Contexto del Hecho` | Registro forense | Texto (Violencia física, psicológica...) | No | Tipo de violencia |
| 30 | `Mecanismo Causal de la Lesión no Fatal` | Evaluación forense | Texto (Contundente, Cortante...) | Q3 (mecanismo_n) | **Feature clave Q3** — cómo se produjo la lesión |
| 31 | `Diagnostico Topográfico de la Lesión no Fatal` | Evaluación médico-forense | Texto | No (alta tasa nulos) | Parte del cuerpo lesionada — potencial feature para Q3 |
| 32 | `Sexo del Agresor` | Registro forense | Texto | No (alta correlación con Presunto Agresor) | Sexo del presunto agresor |
| 33 | `Presunto Agresor Detallado` | Registro forense | Texto (Pareja, Padre, Hijo...) | Q3 (agresor_n) | **Feature clave Q3** — vínculo del agresor con la víctima |
| 34 | `Factor Desencadenante de la Agresión` | Relato de la víctima | Texto (Celos, Alcohol, Económico...) | Q3 (factor_n) | Factor que precipitó la agresión — 56% sin info |
| 35 | `Días de Incapacidad Medicolegal` | Dictamen forense | Texto ordinal | **Target Q2 y Q3** (severidad_n) | Variable objetivo: gravedad de las lesiones producidas |
| 36 | `Pueblo Indígena` | Autodeclaración | Texto | No (57% sin info) | Pueblo indígena de pertenencia |

---

## 11. Justificación de no uso de datos no estructurados

### ¿Por qué no se usan PDFs, imágenes ni artículos científicos?

Esta es una decisión deliberada y fundamentada en cuatro razones:

#### Razón 1: La naturaleza del problema no lo requiere

El problema es de **clasificación supervisada y regresión** sobre registros administrativos forenses. El objetivo es predecir la severidad de un caso de VIF a partir de las características registradas en ese caso. Toda la información relevante ya está en las 36 columnas del dataset tabular. Añadir texto no estructurado no añadiría información nueva sobre el caso individual; añadiría ruido.

#### Razón 2: La fuente ya provee datos estructurados

El INMLCF ya realizó la labor de **estructuración de datos** al registrar cada caso en un formulario estandarizado (SIEDCO). Ese proceso transforma relatos narrativos en categorías discretas (escenario, mecanismo, agresor). Si intentáramos procesar los relatos originales (que no están disponibles públicamente), estaríamos duplicando trabajo ya hecho de forma más confiable por profesionales forenses.

#### Razón 3: Complejidad desproporcionada al beneficio

| Tipo de dato no estructurado | Costo de incorporación | Beneficio esperado |
|---|---|---|
| **PDFs de informes INMLCF** | OCR + parsing + limpieza de texto = 4–6 semanas | Datos ya disponibles en el CSV; redundante |
| **Artículos científicos** | NLP (topic modeling, RAG) = alta complejidad | Información contextual, no predictiva a nivel de caso |
| **Imágenes forenses** | CNN + privacidad ética extrema | No disponibles públicamente; uso restringido |
| **Notas de texto libre** (`Circunstancia del Hecho`) | NLP (BERT en español) = 2–3 semanas | Potencial en una v2 del proyecto |

La relación costo/beneficio no justifica el uso de datos no estructurados para los objetivos del proyecto actual.

#### Razón 4: El dataset tabular ya tiene alta capacidad predictiva

- Q3 ya logra F1 ≈ 0.97 con solo 7 variables estructuradas.
- Q2 logra F1 ≈ 0.83 con 6 variables del perfil de la víctima.
- Añadir datos no estructurados incrementaría la complejidad sin una mejora significativa en métricas que ya son altas.

#### Trabajo futuro donde sí tendría sentido

La única columna de texto libre en el dataset con potencial real es `Circunstancia del Hecho Detallada`. En una versión 2 del proyecto, procesarla con un modelo de lenguaje en español (BERT multilingüe o `dccuchile/bert-base-spanish-wwm-cased`) podría capturar información semántica adicional (e.g., menciones de alcohol, armas, patrones de control) que los campos categóricos no capturan. Esto justificaría el uso de NLP como **complemento** al análisis tabular, no como reemplazo.

---

*Documento preparado por el equipo del proyecto — Mayo 2026*  
*Para consultas metodológicas: ver `Documento_Explicativo.md`*
