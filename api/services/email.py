"""Envío de informes por email via SMTP."""
from __future__ import annotations

import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List

from OEE.db.connector import load_config as _load_cfg


def _get_smtp_config() -> dict:
    cfg = _load_cfg()
    return cfg.get("smtp", {})


def test_smtp() -> str:
    """Prueba la conexión SMTP. Devuelve 'OK' o mensaje de error."""
    smtp_cfg = _get_smtp_config()
    if not smtp_cfg.get("email"):
        return "ERROR: No hay config SMTP. Configura email en Ajustes."
    try:
        with smtplib.SMTP(smtp_cfg["server"], int(smtp_cfg.get("port", 587)), timeout=15) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(smtp_cfg["email"], smtp_cfg["password"])
        return "OK"
    except Exception as exc:
        return f"ERROR: {exc}"


def enviar_informes(
    destinatarios: List[str],
    asunto: str,
    cuerpo: str,
    pdf_paths: List[Path],
) -> str:
    """
    Envía email con PDFs adjuntos.
    Devuelve 'OK' o mensaje de error.
    """
    smtp_cfg = _get_smtp_config()
    if not smtp_cfg.get("email"):
        return "ERROR: No hay config SMTP"

    remitente = smtp_cfg["email"]

    msg = MIMEMultipart()
    msg["From"] = remitente
    msg["To"] = ", ".join(destinatarios)
    msg["Subject"] = asunto
    msg.attach(MIMEText(cuerpo, "plain", "utf-8"))

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            continue
        part = MIMEBase("application", "pdf")
        part.set_payload(pdf_path.read_bytes())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{pdf_path.name}"')
        msg.attach(part)

    try:
        with smtplib.SMTP(smtp_cfg["server"], int(smtp_cfg.get("port", 587)), timeout=30) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(remitente, smtp_cfg["password"])
            s.sendmail(remitente, destinatarios, msg.as_string())
        return "OK"
    except Exception as exc:
        return f"ERROR: {exc}"
