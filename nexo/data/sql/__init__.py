"""Paquete con el SQL loader y los ``.sql`` versionados (D-01).

Los archivos ``.sql`` viven bajo ``nexo/data/sql/<engine>/<method>.sql``
y se cargan con ``nexo.data.sql.loader.load_sql(name)``. Este módulo
paquete (``__init__.py``) es obligatorio para que
``importlib.resources.files('nexo.data.sql')`` resuelva.
"""
