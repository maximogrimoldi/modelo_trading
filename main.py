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

from src.backtest.engine import BacktestConfig, BacktestEngine
from src.data_io.ingestion import DataIngestion
from src.models import FactorModel, JumpDetector, RMTCleaner
from src.utils.helpers import load_config, setup_logger


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comando.

    Returns:
        Namespace con los argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        prog="factor-arb-rmt",
        description="Motor de Factor Arbitraje basado en Random Matrix Theory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Ruta al archivo de configuración YAML (default: config/config.yaml).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nivel de logging (default: INFO).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Subcomando: backtest ---
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Ejecutar backtest walk-forward completo.",
    )
    backtest_parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Forzar re-descarga de datos ignorando caché.",
    )
    backtest_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/backtest_result.parquet"),
        help="Ruta de salida para los resultados del backtest.",
    )

    # --- Subcomando: scan ---
    scan_parser = subparsers.add_parser(
        "scan",
        help="Escanear señales activas en el universo (último período).",
    )
    scan_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/signals.parquet"),
        help="Ruta de salida para las señales detectadas.",
    )
    scan_parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Número de señales top a reportar (default: 10).",
    )

    return parser.parse_args()


def run_backtest(args: argparse.Namespace, config: dict, logger: logging.Logger) -> None:
    """Ejecuta el pipeline completo de backtest.

    Args:
        args: Argumentos de línea de comando parseados.
        config: Configuración cargada desde YAML.
        logger: Logger configurado del proyecto.
    """
    logger.info("=== MODO: BACKTEST ===")

    # 1. Ingesta de datos
    tickers_file = Path(config["universe"]["tickers_file"])
    tickers = [
        line.strip()
        for line in tickers_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    logger.info("Universo cargado: %d tickers", len(tickers))

    ingestion = DataIngestion(
        tickers=tickers,
        cache_dir=Path(config["data"]["cache_dir"]),
        start_date=config["universe"]["start_date"],
        end_date=config["universe"]["end_date"],
    )
    prices = ingestion.fetch(force_refresh=args.refresh_data)
    log_returns = ingestion.get_log_returns(prices)
    logger.info("Log-retornos calculados: %s", log_returns.shape)

    # 2. Configurar y correr backtest
    bt_cfg = BacktestConfig(
        train_window=config["backtest"]["train_window"],
        rebalance_freq=config["backtest"]["rebalance_freq"],
        n_factors=config["factor_model"]["n_factors"],
        zscore_entry=config["backtest"]["zscore_entry"],
        zscore_exit=config["backtest"]["zscore_exit"],
        transaction_cost_bps=config["backtest"]["transaction_cost_bps"],
        max_leverage=config["backtest"]["max_leverage"],
    )
    engine = BacktestEngine(config=bt_cfg)
    result = engine.run(log_returns)

    # 3. Reportar métricas
    metrics = engine.compute_metrics(result.portfolio_returns)
    logger.info("--- Métricas de Performance ---")
    for k, v in metrics.items():
        logger.info("  %-20s: %.4f", k, v)

    # 4. Persistir resultados
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.portfolio_returns.to_frame("portfolio_return").to_parquet(args.output)
    logger.info("Resultados guardados en %s", args.output)


def run_scan(args: argparse.Namespace, config: dict, logger: logging.Logger) -> None:
    """Escanea señales activas en el universo de activos.

    Args:
        args: Argumentos de línea de comando parseados.
        config: Configuración cargada desde YAML.
        logger: Logger configurado del proyecto.
    """
    logger.info("=== MODO: SCAN DE SEÑALES ===")
    # TODO: Implementar pipeline de escáner:
    #   1. Descargar datos recientes (últimos train_window días)
    #   2. Ajustar RMTCleaner, FactorModel, JumpDetector
    #   3. Calcular z-scores del último día disponible
    #   4. Filtrar señales con saltos detectados
    #   5. Ordenar por |z-score| descendente y reportar top-N
    raise NotImplementedError("run_scan() — pendiente de implementación.")


def main() -> None:
    """Función principal — punto de entrada del motor."""
    args = parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(level=log_level)

    # Cargar configuración
    config = load_config(args.config)
    logger.info("Configuración cargada desde %s", args.config)

    # Despachar al subcomando correspondiente
    dispatch = {
        "backtest": run_backtest,
        "scan": run_scan,
    }
    try:
        dispatch[args.command](args, config, logger)
    except NotImplementedError as exc:
        logger.error("Funcionalidad no implementada: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Error inesperado: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
