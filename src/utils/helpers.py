"""Utilidades transversales del proyecto.

Funciones de configuración de logging, carga de configuración YAML
y helpers misceláneos reutilizables en todo el pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def setup_logger(
    name: str = "factor_arb",
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configura y retorna un logger estándar para el proyecto.

    Args:
        name: Nombre del logger (por defecto 'factor_arb').
        level: Nivel de logging (ej. logging.DEBUG, logging.INFO).
        log_file: Ruta opcional a un archivo de log. Si es None, sólo stdout.

    Returns:
        Logger configurado con formato estructurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


def load_config(config_path: Path | str) -> Dict[str, Any]:
    """Carga la configuración del proyecto desde un archivo YAML.

    Args:
        config_path: Ruta al archivo de configuración (config/config.yaml).

    Returns:
        Diccionario con la configuración del proyecto.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        yaml.YAMLError: Si el archivo tiene errores de sintaxis YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    return config
