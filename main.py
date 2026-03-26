"""Punto de entrada principal del motor de Factor Arbitraje RMT.

Uso:
    # Correr backtest completo
    python main.py backtest --config config/config.yaml

    # Escanear señales en tiempo real (último día disponible)
    python main.py scan --config config/config.yaml --output results/signals.parquet

    # Forzar re-descarga de datos
    python main.py backtest --config config/config.yaml --refresh-data
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
