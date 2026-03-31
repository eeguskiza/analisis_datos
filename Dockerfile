FROM python:3.11-slim

# ── Dependencias del sistema (ODBC + fuentes para matplotlib) ────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl gnupg2 apt-transport-https unixodbc-dev fonts-dejavu-core && \
    # Microsoft ODBC Driver 17 for SQL Server
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | \
        gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 && \
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

# Matplotlib sin GUI
ENV MPLBACKEND=Agg

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
