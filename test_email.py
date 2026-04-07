"""Test rápido de envío de email via SMTP Office 365."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.services.email import test_smtp

print("Probando conexión SMTP a smtp.office365.com...")
resultado = test_smtp()
print(f"Resultado: {resultado}")

if resultado == "OK":
    print("\nConexión OK. Ahora probando enviar email de prueba...")
    from api.services.email import enviar_informes
    res = enviar_informes(
        destinatarios=["e.eguskiza@ecsmobility.com"],
        asunto="Test OEE Planta - Email funciona",
        cuerpo="Si recibes esto, el envío de informes por email está configurado correctamente.",
        pdf_paths=[],  # sin adjuntos, solo texto de prueba
    )
    print(f"Envío: {res}")
