FROM python:3.11-slim-bookworm

# ── Dependencias del sistema (ODBC + fuentes para matplotlib) ────────────────
# Detecta arquitectura para seleccionar el repo correcto de Microsoft
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl gnupg2 apt-transport-https unixodbc-dev fonts-dejavu-core && \
    # Microsoft ODBC Driver 18 — soporta amd64 (Intel) y arm64 (Apple Silicon)
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | \
        gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    ARCH=$(dpkg --print-architecture) && \
    echo "deb [arch=${ARCH} signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencias Python ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Codigo fuente ────────────────────────────────────────────────────────────
COPY api/ ./api/
COPY OEE/ ./OEE/
COPY templates/ ./templates/
COPY static/ ./static/
COPY server.py .

# ── Volumenes para datos persistentes ────────────────────────────────────────
VOLUME ["/app/data", "/app/informes"]

EXPOSE 8000

ENV MPLBACKEND=Agg

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
