#!/usr/bin/env python3
"""Daily ML trading pipeline for Railway deployment.

Runs once daily (triggered by cron):
1. Download latest daily bars via yfinance (stocks + macro)
2. Compute features (62 stock + 22 macro = 84 total)
3. Run LightGBM predictions → rank stocks
4. Rebalance portfolio via Alpaca API (long top 20)
5. Log full run report

Config: H5_LongOnly20 — 5-day horizon, long top 20, no shorts.
Rebalances every 5 trading days.

Usage:
    python pipeline.py                  # Full run
    python pipeline.py --dry-run        # Predict only, no orders
    python pipeline.py --force          # Force rebalance even if not due
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

TOP_N = 20              # Number of stocks to hold long
HORIZON = 5             # Prediction horizon (trading days)
MIN_HISTORY_DAYS = 250  # Minimum days of history needed for features
LOOKBACK_DAYS = 300     # Days of history to download for feature computation

# Alpaca
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
# Fix common typo: ensure URL starts with https://
if ALPACA_BASE_URL and not ALPACA_BASE_URL.startswith("http"):
    ALPACA_BASE_URL = "https://" + ALPACA_BASE_URL.lstrip("htps:/")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "ml_v4_model.pkl"

# Persistent data directory — mount a Railway volume at /app/data to survive redeploys
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
STATE_PATH = DATA_DIR / "state" / "pipeline_state.json"
LOG_DIR = DATA_DIR / "logs"

# Macro tickers needed for features
MACRO_TICKERS = [
    "^VIX", "SPY", "QQQ", "IWM", "TLT", "SHY", "HYG",
    "GLD", "USO", "UUP",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU",
]

MACRO_RENAME = {"^VIX": "VIX"}

# Symbols excluded from trading (IPO too recent, data quality issues)
EXCLUDED_SYMBOLS = {
    "RIVN", "LCID", "PLTR", "HOOD", "AFRM", "IONQ", "JOBY",
    "DNA", "GRAB", "NU", "RKLB", "VFS", "SMCI",
}

# Hardcoded universe — S&P 500 + Nasdaq 100 stocks we trained on.
# More reliable than scraping Wikipedia (which gets blocked by Railway).
# Update this list every few months when index composition changes.
STOCK_UNIVERSE = [
    "A", "AA", "AAL", "AAOI", "AAON", "AAPL", "ABBV", "ABNB", "ABT", "ACAD",
    "ACGL", "ACI", "ACM", "ACN", "ADBE", "ADC", "ADI", "ADM", "ADP", "ADSK",
    "AEE", "AEIS", "AEM", "AEO", "AEP", "AES", "AFG", "AFL", "AFRM", "AG",
    "AGCO", "AHR", "AIG", "AIT", "AIZ", "AJG", "AKAM", "ALB", "ALGM", "ALGN",
    "ALK", "ALL", "ALLE", "ALLY", "ALNY", "ALV", "AM", "AMAT", "AMCR", "AMD",
    "AME", "AMG", "AMGN", "AMH", "AMKR", "AMP", "AMPX", "AMT", "AMZN", "AN",
    "ANET", "ANF", "AON", "AOS", "APA", "APD", "APG", "APH", "APLS", "APO",
    "APP", "APPF", "APTV", "AR", "ARCT", "ARE", "ARES", "ARM", "ARMK", "ARW",
    "ARWR", "ASB", "ASH", "ASML", "ATI", "ATO", "ATR", "AVAV", "AVB", "AVGO",
    "AVNT", "AVT", "AVTR", "AVY", "AWK", "AXON", "AXP", "AXTA", "AYI", "AZO",
    "BA", "BAC", "BAH", "BALL", "BAX", "BBWI", "BBY", "BC", "BCO", "BCS",
    "BDC", "BDX", "BEAM", "BEN", "BF-B", "BG", "BHF", "BIIB", "BILL", "BIO",
    "BITF", "BJ", "BK", "BKH", "BKNG", "BKR", "BLD", "BLDR", "BLK", "BLKB",
    "BMNR", "BMRN", "BMY", "BN", "BP", "BR", "BRBR", "BRK-B", "BRKR", "BRO",
    "BROS", "BRX", "BSX", "BSY", "BTG", "BURL", "BVN", "BWA", "BWXT", "BX",
    "BXP", "BYD", "C", "CACI", "CAG", "CAH", "CAR", "CARR", "CART", "CASY",
    "CAT", "CAVA", "CB", "CBOE", "CBRE", "CBSH", "CBT", "CCEP", "CCI", "CCK",
    "CCL", "CDE", "CDNS", "CDP", "CDW", "CEG", "CELH", "CENX", "CF", "CFG",
    "CFR", "CG", "CGNX", "CHD", "CHDN", "CHE", "CHH", "CHRD", "CHRW", "CHTR",
    "CHWY", "CI", "CIEN", "CIFR", "CINF", "CL", "CLF", "CLH", "CLSK", "CLX",
    "CMC", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNH", "CNM", "CNO",
    "CNP", "CNQ", "CNX", "CNXC", "COF", "COHR", "COIN", "COKE", "COLB", "COLM",
    "COO", "COP", "COR", "COST", "COTY", "CPAY", "CPB", "CPRI", "CPRT", "CPT",
    "CR", "CRBG", "CRCL", "CRGY", "CRH", "CRL", "CRM", "CROX", "CRS", "CRSP",
    "CRUS", "CRWD", "CSCO", "CSGP", "CSL", "CSX", "CTAS", "CTRA", "CTRE", "CTSH",
    "CTVA", "CUBE", "CUZ", "CVLT", "CVNA", "CVS", "CVX", "CW", "CXT", "CYTK",
    "CZR", "D", "DAC", "DAL", "DAR", "DASH", "DAWN", "DBX", "DCI", "DD",
    "DDOG", "DE", "DECK", "DELL", "DG", "DGX", "DHI", "DHR", "DHT", "DINO",
    "DIS", "DKNG", "DKS", "DLB", "DLR", "DLTR", "DOC", "DOCS", "DOCU", "DOV",
    "DOW", "DPZ", "DRI", "DT", "DTE", "DTM", "DUK", "DUOL", "DVA", "DVN",
    "DXCM", "DY", "EA", "EBAY", "ECL", "ED", "EDIT", "EEFT", "EFX", "EG",
    "EGP", "EHC", "EIX", "EL", "ELAN", "ELF", "ELS", "ELV", "EME", "EMR",
    "ENS", "ENSG", "ENTG", "EOG", "EPAM", "EPR", "EQH", "EQIX", "EQNR", "EQR",
    "EQT", "EQX", "ERIE", "ES", "ESAB", "ESNT", "ESS", "ET", "ETN", "ETR",
    "EVR", "EVRG", "EW", "EWBC", "EXAS", "EXC", "EXE", "EXEL", "EXLS", "EXP",
    "EXPD", "EXPE", "EXPO", "EXR", "F", "FAF", "FANG", "FAST", "FATE", "FBIN",
    "FCFS", "FCN", "FCX", "FDS", "FDX", "FE", "FER", "FFIN", "FFIV", "FHI",
    "FHN", "FICO", "FIGR", "FIS", "FISV", "FITB", "FIVE", "FIX", "FLEX", "FLG",
    "FLO", "FLR", "FLS", "FN", "FNB", "FND", "FNF", "FOUR", "FOX", "FOXA",
    "FR", "FRO", "FRT", "FSLR", "FTI", "FTNT", "FTV", "G", "GAP", "GATX",
    "GBCI", "GD", "GDDY", "GE", "GEF", "GEHC", "GEN", "GEV", "GGG", "GHC",
    "GILD", "GIS", "GL", "GLPI", "GLW", "GM", "GME", "GMED", "GNRC", "GNTX",
    "GOLD", "GOOG", "GOOGL", "GPC", "GPK", "GPN", "GRMN", "GS", "GT", "GTLS",
    "GWRE", "GWW", "GXO", "H", "HAE", "HAL", "HALO", "HAS", "HBAN", "HCA",
    "HD", "HDB", "HGV", "HIG", "HII", "HIMS", "HL", "HLI", "HLNE", "HLT",
    "HOG", "HOLX", "HOMB", "HON", "HOOD", "HP", "HPE", "HPQ", "HQY", "HR",
    "HRB", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HUT", "HWC", "HWM",
    "HXL", "IBKR", "IBM", "IBOC", "ICE", "IDA", "IDCC", "IDXX", "IEX", "IFF",
    "ILMN", "INCY", "INGR", "INSM", "INSW", "INTC", "INTU", "INVH", "IONQ", "IOT",
    "IP", "IPGP", "IQV", "IR", "IRM", "IRT", "ISRG", "IT", "ITT", "ITW",
    "IVZ", "J", "JAZZ", "JBHT", "JBL", "JCI", "JEF", "JHG", "JKHY", "JLL",
    "JNJ", "JPM", "KBH", "KBR", "KD", "KDP", "KEX", "KEY", "KEYS", "KGC",
    "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KNF", "KNSL", "KNX", "KO",
    "KR", "KRC", "KRG", "KTOS", "KVUE", "L", "LAD", "LAMR", "LDOS", "LEA",
    "LECO", "LEN", "LFUS", "LH", "LHX", "LII", "LIN", "LITE", "LIVN", "LLY",
    "LMT", "LNT", "LNTH", "LOPE", "LOW", "LPX", "LRCX", "LSCC", "LSTR", "LULU",
    "LUV", "LVS", "LYB", "LYFT", "LYV", "M", "MA", "MAA", "MANH", "MAR",
    "MARA", "MAS", "MASI", "MAT", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT",
    "MEDP", "MELI", "MET", "META", "MGM", "MIDD", "MKC", "MKSI", "MLI", "MLM",
    "MMM", "MMS", "MNST", "MO", "MOG-A", "MORN", "MOS", "MP", "MPC", "MPWR",
    "MRK", "MRNA", "MRSH", "MRVL", "MS", "MSA", "MSCI", "MSFT", "MSI", "MSM",
    "MSTR", "MTB", "MTD", "MTDR", "MTG", "MTN", "MTSI", "MTZ", "MU", "MUD",
    "MUR", "MUSA", "NBIX", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NET",
    "NEU", "NFG", "NFLX", "NGD", "NI", "NIO", "NJR", "NKE", "NLY", "NNN",
    "NOC", "NOV", "NOVT", "NOW", "NRG", "NSA", "NSC", "NTAP", "NTLA", "NTNX",
    "NTRS", "NUE", "NVDA", "NVR", "NVST", "NVT", "NVTS", "NWE", "NWS", "NWSA",
    "NXPI", "NXST", "NXT", "NYT", "O", "OC", "ODFL", "OGE", "OGS", "OHI",
    "OKE", "OKTA", "OLED", "OLLI", "OLN", "OMC", "ON", "ONB", "ONON", "ONTO",
    "OPCH", "ORA", "ORCL", "ORI", "ORLY", "OSK", "OTIS", "OVV", "OWL", "OXY",
    "OZK", "PAAS", "PAG", "PANW", "PATH", "PAYX", "PB", "PBF", "PBR", "PCAR",
    "PCG", "PCTY", "PDD", "PEG", "PEGA", "PEN", "PENN", "PEP", "PFE", "PFG",
    "PFGC", "PG", "PGR", "PH", "PHM", "PII", "PINS", "PK", "PKG", "PLD",
    "PLNT", "PLTD", "PLTR", "PM", "PNC", "PNFP", "PNR", "PNW", "PODD", "POOL",
    "POR", "POST", "PPC", "PPG", "PPL", "PR", "PRI", "PRU", "PSA", "PSKY",
    "PSN", "PSTG", "PSX", "PTC", "PTEN", "PVH", "PWR", "PYPL", "QCOM", "QLYS",
    "QUBT", "R", "RARE", "RBA", "RBC", "RBLX", "RCAT", "RCL", "REG", "REGN",
    "REXR", "RF", "RGA", "RGEN", "RGLD", "RGTI", "RH", "RIOT", "RJF", "RKT",
    "RL", "RLI", "RMBS", "RMD", "RNR", "ROIV", "ROK", "ROKU", "ROL", "ROP",
    "ROST", "RPM", "RRC", "RRX", "RS", "RSG", "RTX", "RUN", "RVTY", "RYAN",
    "RYN", "SAIA", "SAIC", "SAM", "SARO", "SATS", "SAVA", "SBAC", "SBLK", "SBRA",
    "SBUX", "SCHW", "SCI", "SE", "SEIC", "SF", "SFM", "SGI", "SHC", "SHOP",
    "SHW", "SIGI", "SITM", "SJM", "SLAB", "SLB", "SLGN", "SLM", "SM", "SMCI",
    "SMG", "SNA", "SNAP", "SNOW", "SNPS", "SNX", "SO", "SOC", "SOFI", "SOLV",
    "SON", "SPG", "SPGI", "SPOT", "SPXC", "SR", "SRE", "SRPT", "SSB", "SSD",
    "SSRM", "ST", "STAG", "STE", "STLD", "STNG", "STRL", "STT", "STWD", "STX",
    "STZ", "SW", "SWK", "SWKS", "SWX", "SYF", "SYK", "SYNA", "SYY", "T",
    "TAP", "TCBI", "TDG", "TDY", "TEAM", "TECH", "TECK", "TEL", "TER", "TEVA",
    "TEX", "TFC", "TGT", "THC", "THG", "THO", "TJX", "TKO", "TKR", "TLN",
    "TMHC", "TMO", "TMUS", "TNK", "TNL", "TOL", "TPL", "TPR", "TREX", "TRGP",
    "TRI", "TRMB", "TROW", "TRU", "TRV", "TSCO", "TSLA", "TSLG", "TSM", "TSN",
    "TT", "TTC", "TTD", "TTEK", "TTMI", "TTWO", "TWLO", "TXN", "TXNM", "TXRH",
    "TXT", "TYL", "UAL", "UBER", "UBSI", "UDR", "UEC", "UFPI", "UGI", "UHS",
    "ULTA", "UMBF", "UNH", "UNM", "UNP", "UPS", "UPST", "URI", "USB", "USFD",
    "UTHR", "UUUU", "V", "VAL", "VALE", "VC", "VFC", "VG", "VICI", "VICR",
    "VLO", "VLTO", "VLY", "VMC", "VMI", "VNO", "VNOM", "VNT", "VOYA", "VRSK",
    "VRSN", "VRT", "VRTX", "VST", "VTR", "VTRS", "VVV", "VZ", "WAB", "WAL",
    "WAT", "WBD", "WBS", "WCC", "WDAY", "WDC", "WEC", "WELL", "WEX", "WFC",
    "WFRD", "WH", "WHR", "WING", "WLK", "WM", "WMB", "WMG", "WMS", "WMT",
    "WPC", "WPM", "WRB", "WSM", "WSO", "WST", "WTFC", "WTRG", "WTS", "WTW",
    "WWD", "WY", "WYNN", "XEL", "XOM", "XP", "XPEV", "XPO", "XRAY", "XYL",
    "XYZ", "YETI", "YUM", "ZBH", "ZBRA", "ZIM", "ZION", "ZM", "ZS", "ZTS",
]


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

class RunReport:
    """Collects all run data for a final summary."""

    def __init__(self):
        self.start_time = datetime.now()
        self.steps = {}
        self.warnings = []
        self.errors = []
        self.timings = {}
        self.data = {}

    def start_step(self, name: str):
        self.steps[name] = {"status": "running", "start": time.time()}

    def end_step(self, name: str, status: str = "ok"):
        if name in self.steps:
            elapsed = time.time() - self.steps[name]["start"]
            self.steps[name]["status"] = status
            self.steps[name]["elapsed"] = elapsed
            self.timings[name] = elapsed

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_error(self, msg: str):
        self.errors.append(msg)

    def set(self, key: str, value):
        self.data[key] = value

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def format_summary(self) -> str:
        total_time = (datetime.now() - self.start_time).total_seconds()
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  RUN SUMMARY")
        lines.append("=" * 70)
        lines.append(f"  Date:         {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Total time:   {total_time:.1f}s")
        lines.append("")

        # Step timings
        lines.append("  STEPS:")
        for name, info in self.steps.items():
            elapsed = info.get("elapsed", 0)
            status = info["status"]
            icon = "OK" if status == "ok" else "SKIP" if status == "skipped" else "FAIL"
            lines.append(f"    [{icon:4s}] {name:30s} {elapsed:6.1f}s")
        lines.append("")

        # Data quality
        if "universe_size" in self.data:
            lines.append("  DATA:")
            lines.append(f"    Universe requested:   {self.data.get('universe_size', '?')} symbols")
            lines.append(f"    Stocks downloaded:    {self.data.get('stocks_downloaded', '?')} symbols")
            lines.append(f"    Stocks failed:        {self.data.get('stocks_failed', '?')} symbols")
            lines.append(f"    Macro downloaded:     {self.data.get('macro_downloaded', '?')}/{len(MACRO_TICKERS)} tickers")
            lines.append(f"    Macro missing:        {', '.join(self.data.get('macro_missing', [])) or 'none'}")
            lines.append(f"    Feature columns:      {self.data.get('n_features', '?')}")
            lines.append(f"    Latest data date:     {self.data.get('latest_data_date', '?')}")
            lines.append("")

        # Predictions
        if "n_predictions" in self.data:
            lines.append("  PREDICTIONS:")
            lines.append(f"    Stocks predicted:     {self.data.get('n_predictions', '?')}")
            lines.append(f"    Prediction failures:  {self.data.get('prediction_failures', '?')}")
            pred_range = self.data.get("prediction_range", {})
            lines.append(f"    Pred range:           {pred_range.get('min', '?'):.2f}% to {pred_range.get('max', '?'):.2f}%")
            lines.append(f"    Pred mean:            {pred_range.get('mean', '?'):.2f}%")
            lines.append(f"    Pred std:             {pred_range.get('std', '?'):.2f}%")
            lines.append("")

        # Portfolio
        if "target_portfolio" in self.data:
            lines.append("  TARGET PORTFOLIO (top 20):")
            for i, (sym, pred) in enumerate(self.data["target_portfolio"]):
                lines.append(f"    {i+1:2d}. {sym:6s}  {pred:+6.2f}%")
            lines.append("")

        # Rebalance
        if "rebalance" in self.data:
            rb = self.data["rebalance"]
            lines.append("  REBALANCE:")
            lines.append(f"    Mode:                 {'DRY RUN' if rb.get('dry_run') else 'LIVE'}")
            lines.append(f"    Portfolio value:       ${rb.get('portfolio_value', 0):,.2f}")
            lines.append(f"    Cash available:        ${rb.get('cash', 0):,.2f}")
            lines.append(f"    Target weight/stock:   ${rb.get('target_weight', 0):,.2f}")
            lines.append(f"    Positions before:      {rb.get('positions_before', '?')}")
            lines.append(f"    Sells (exit):          {rb.get('n_sells', 0)}")
            lines.append(f"    Buys (new):            {rb.get('n_buys', 0)}")
            lines.append(f"    Rebalanced (adjust):   {rb.get('n_rebalanced', 0)}")
            lines.append(f"    Held (no change):      {rb.get('n_held', 0)}")
            lines.append(f"    Turnover:              {rb.get('turnover', 0):.0%}")
            if not rb.get("dry_run"):
                lines.append(f"    Orders executed:       {rb.get('executed', 0)}")
                lines.append(f"    Orders failed:         {rb.get('failed', 0)}")

            # Detail sold/bought
            if rb.get("sells_detail"):
                lines.append(f"    Sold:   {', '.join(rb['sells_detail'])}")
            if rb.get("buys_detail"):
                lines.append(f"    Bought: {', '.join(rb['buys_detail'])}")
            lines.append("")

        # Warnings / Errors
        if self.warnings:
            lines.append(f"  WARNINGS ({len(self.warnings)}):")
            for w in self.warnings[:20]:
                lines.append(f"    - {w}")
            if len(self.warnings) > 20:
                lines.append(f"    ... and {len(self.warnings) - 20} more")
            lines.append("")

        if self.errors:
            lines.append(f"  ERRORS ({len(self.errors)}):")
            for e in self.errors[:20]:
                lines.append(f"    - {e}")
            lines.append("")

        # Final status
        status = "SUCCESS" if not self.errors else "COMPLETED WITH ERRORS"
        lines.append(f"  STATUS: {status}")
        lines.append("=" * 70)
        return "\n".join(lines)


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]
    for h in handlers:
        h.setFormatter(fmt)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in handlers:
        logger.addHandler(h)

    return logging.getLogger("pipeline"), log_file


# ═══════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════

def get_tradeable_symbols(logger, report: RunReport) -> list[str]:
    """Return hardcoded stock universe (307 stocks from S&P 500 + Nasdaq 100)."""
    report.start_step("get_universe")
    symbols = [s for s in STOCK_UNIVERSE if s not in EXCLUDED_SYMBOLS]
    logger.info(f"  Universe: {len(symbols)} stocks (hardcoded, excludes {len(EXCLUDED_SYMBOLS)} blacklisted)")
    report.set("universe_size", len(symbols))
    report.end_step("get_universe")
    return symbols


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance column names (handles MultiIndex from newer versions)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df


def download_bars(symbols: list[str], days: int, logger,
                  report: RunReport) -> dict[str, pd.DataFrame]:
    """Download recent daily bars from Yahoo Finance."""
    report.start_step("download_stocks")
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))

    logger.info(f"  Date range: {start.date()} → {end.date()}")
    logger.info(f"  Symbols to download: {len(symbols)}")

    data = {}
    failed_syms = []
    batch_size = 50
    n_batches = (len(symbols) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(symbols), batch_size):
        batch_num = batch_idx // batch_size + 1
        batch = symbols[batch_idx:batch_idx + batch_size]
        tickers_str = " ".join(batch)
        batch_start = time.time()

        try:
            df = yf.download(
                tickers_str, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                group_by="ticker", auto_adjust=True, progress=False,
                threads=True,
            )
            batch_ok = 0
            for sym in batch:
                try:
                    if len(batch) == 1:
                        sym_df = df.copy()
                    else:
                        sym_df = df[sym].copy()
                    sym_df = _normalize_columns(sym_df)
                    sym_df = sym_df.dropna(subset=["close"])
                    if len(sym_df) >= MIN_HISTORY_DAYS:
                        data[sym] = sym_df
                        batch_ok += 1
                    else:
                        failed_syms.append(sym)
                except Exception:
                    failed_syms.append(sym)

            batch_time = time.time() - batch_start
            logger.info(f"  Batch {batch_num}/{n_batches}: "
                        f"{batch_ok}/{len(batch)} ok ({batch_time:.1f}s)")

        except Exception as e:
            logger.warning(f"  Batch {batch_num}/{n_batches} FAILED: {e}")
            report.add_warning(f"Stock batch {batch_num} failed: {e}")
            failed_syms.extend(batch)

        time.sleep(0.3)

    logger.info(f"  Downloaded: {len(data)} symbols, "
                f"failed: {len(failed_syms)} symbols")

    if data:
        sample_sym = next(iter(data))
        sample_df = data[sample_sym]
        logger.info(f"  Sample ({sample_sym}): {len(sample_df)} days, "
                    f"{sample_df.index[0].date()} → {sample_df.index[-1].date()}")
        report.set("latest_data_date", str(sample_df.index[-1].date()))

    report.set("stocks_downloaded", len(data))
    report.set("stocks_failed", len(failed_syms))
    if failed_syms and len(failed_syms) <= 20:
        report.add_warning(f"Failed stocks: {', '.join(failed_syms)}")
    report.end_step("download_stocks")
    return data


def download_macro(days: int, logger, report: RunReport) -> dict[str, pd.DataFrame]:
    """Download macro tickers."""
    report.start_step("download_macro")
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.5))

    macro = {}
    missing = []

    for ticker in MACRO_TICKERS:
        try:
            df = yf.download(
                ticker, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True, progress=False,
            )
            df = _normalize_columns(df)
            name = MACRO_RENAME.get(ticker, ticker)
            df_clean = df.dropna(subset=["close"])
            if len(df_clean) > 0:
                macro[name] = df_clean
                logger.info(f"  {name:6s}: {len(df_clean)} days, "
                            f"latest close: {df_clean['close'].iloc[-1]:.2f}")
            else:
                missing.append(ticker)
                logger.warning(f"  {ticker}: no data returned")
        except Exception as e:
            missing.append(ticker)
            logger.warning(f"  {ticker}: FAILED — {e}")
            report.add_warning(f"Macro {ticker} failed: {e}")
        time.sleep(0.1)

    logger.info(f"  Macro: {len(macro)}/{len(MACRO_TICKERS)} downloaded")
    if missing:
        logger.warning(f"  Missing: {', '.join(missing)}")

    report.set("macro_downloaded", len(macro))
    report.set("macro_missing", missing)
    report.end_step("download_macro")
    return macro


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (same as train_ml_v4.py)
# ═══════════════════════════════════════════════════════════════════════════

def compute_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stock daily features — identical to training pipeline."""
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    ret = c.pct_change()

    feats = {}

    # Returns at multiple horizons
    for d in [1, 2, 3, 5, 10, 20, 60, 120]:
        feats[f"ret_{d}d"] = c.pct_change(d) * 100

    # Moving average distances
    for p in [5, 10, 20, 50, 100, 200]:
        sma = c.rolling(p, min_periods=p).mean()
        feats[f"sma_{p}"] = (c - sma) / (sma + 1e-8) * 100
    for p in [20, 50, 100]:
        sma = c.rolling(p, min_periods=p).mean()
        feats[f"sma_{p}_slope"] = sma.pct_change(5) * 100

    # EMA distances
    for p in [12, 26, 50]:
        ema = c.ewm(span=p, adjust=False).mean()
        feats[f"ema_{p}"] = (c - ema) / (ema + 1e-8) * 100

    # RSI
    for p in [5, 14, 21]:
        delta = c.diff()
        up = delta.clip(lower=0)
        down = (-delta).clip(lower=0)
        rs = up.rolling(p).mean() / (down.rolling(p).mean() + 1e-8)
        feats[f"rsi_{p}"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    feats["macd_hist"] = (macd - macd_signal) / (c + 1e-8) * 100

    # Volatility
    for p in [5, 10, 20, 60]:
        feats[f"vol_{p}"] = ret.rolling(p).std() * np.sqrt(252)

    # Bollinger %B
    for p in [10, 20]:
        mid = c.rolling(p).mean()
        std = c.rolling(p).std()
        feats[f"bb_{p}"] = (c - (mid - 2 * std)) / (4 * std + 1e-8)

    # ATR
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    for p in [5, 14, 20]:
        feats[f"atr_{p}"] = tr.rolling(p).mean() / (c + 1e-8) * 100

    # Volume features
    v_sma20 = v.rolling(20).mean()
    feats["vol_ratio"] = v / (v_sma20 + 1e-8)
    feats["vol_trend"] = v.rolling(5).mean() / (v_sma20 + 1e-8)

    # OBV trend
    obv = (v * ret.apply(np.sign)).cumsum()
    obv_sma = obv.rolling(20).mean()
    feats["obv_trend"] = (obv - obv_sma) / (obv_sma.abs() + 1e-8)

    # Momentum / ROC
    for p in [5, 10, 20, 60]:
        feats[f"roc_{p}"] = c.pct_change(p) * 100

    # Statistical
    for p in [20, 60]:
        feats[f"skew_{p}"] = ret.rolling(p).skew()
        feats[f"kurt_{p}"] = ret.rolling(p).kurt()

    # Sharpe
    for p in [20, 60]:
        rm = ret.rolling(p)
        feats[f"sharpe_{p}"] = (rm.mean() / (rm.std() + 1e-8)) * np.sqrt(252)

    # Channel position
    for p in [10, 20, 40, 60]:
        ch_h = h.rolling(p).max()
        ch_l = l.rolling(p).min()
        feats[f"channel_{p}"] = (c - ch_l) / (ch_h - ch_l + 1e-8)

    # Distance from highs/lows
    for p in [20, 60, 120]:
        feats[f"dist_hi_{p}"] = (c - h.rolling(p).max()) / (c + 1e-8) * 100
        feats[f"dist_lo_{p}"] = (c - l.rolling(p).min()) / (c + 1e-8) * 100

    # Intraday features from daily bar
    feats["day_range"] = (h - l) / (c + 1e-8) * 100
    feats["close_loc"] = (c - l) / (h - l + 1e-8)

    # Calendar
    if hasattr(df.index, "dayofweek"):
        dow = df.index.dayofweek
        feats["dow_sin"] = pd.Series(np.sin(2 * np.pi * dow / 5), index=df.index)
        feats["dow_cos"] = pd.Series(np.cos(2 * np.pi * dow / 5), index=df.index)
    if hasattr(df.index, "month"):
        m = df.index.month
        feats["month_sin"] = pd.Series(np.sin(2 * np.pi * m / 12), index=df.index)
        feats["month_cos"] = pd.Series(np.cos(2 * np.pi * m / 12), index=df.index)

    result = pd.DataFrame(feats, index=df.index)
    for col in result.columns:
        result[col] = result[col].astype(np.float32)
    return result


def compute_macro_features(macro_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute market-level features — identical to training pipeline."""
    feats = {}

    if "VIX" in macro_data:
        vix = macro_data["VIX"]["close"]
        feats["vix"] = vix
        feats["vix_sma20"] = (vix - vix.rolling(20).mean()) / (vix.rolling(20).mean() + 1e-8) * 100
        feats["vix_change5"] = vix.pct_change(5) * 100
        feats["vix_rank20"] = vix.rolling(60).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
        )

    for name in ["SP500", "SPY"]:
        if name in macro_data:
            c = macro_data[name]["close"]
            feats[f"{name}_ret5"] = c.pct_change(5) * 100
            feats[f"{name}_ret20"] = c.pct_change(20) * 100
            feats[f"{name}_sma50"] = (c - c.rolling(50).mean()) / (c.rolling(50).mean() + 1e-8) * 100
            feats[f"{name}_sma200"] = (c - c.rolling(200).mean()) / (c.rolling(200).mean() + 1e-8) * 100
            feats[f"{name}_vol20"] = c.pct_change().rolling(20).std() * np.sqrt(252)
            break

    if "TLT" in macro_data and "SHY" in macro_data:
        ratio = macro_data["TLT"]["close"] / macro_data["SHY"]["close"]
        feats["yield_curve"] = (ratio - ratio.rolling(60).mean()) / (ratio.rolling(60).std() + 1e-8)
        feats["yield_curve_ret20"] = ratio.pct_change(20) * 100

    if "HYG" in macro_data:
        hyg = macro_data["HYG"]["close"]
        feats["credit_ret5"] = hyg.pct_change(5) * 100
        feats["credit_ret20"] = hyg.pct_change(20) * 100

    for name, prefix in [("UUP", "dollar"), ("GLD", "gold"), ("USO", "oil")]:
        if name in macro_data:
            c = macro_data[name]["close"]
            feats[f"{prefix}_ret5"] = c.pct_change(5) * 100
            feats[f"{prefix}_ret20"] = c.pct_change(20) * 100

    sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU"]
    available_sectors = [s for s in sector_etfs if s in macro_data]
    if len(available_sectors) >= 4:
        sector_rets = pd.DataFrame()
        for s in available_sectors:
            sector_rets[s] = macro_data[s]["close"].pct_change(20) * 100
        feats["sector_spread"] = sector_rets.max(axis=1) - sector_rets.min(axis=1)
        feats["sector_dispersion"] = sector_rets.std(axis=1)

    if "IWM" in macro_data and "SPY" in macro_data:
        ratio = macro_data["IWM"]["close"] / macro_data["SPY"]["close"]
        feats["small_vs_large"] = ratio.pct_change(20) * 100

    result = pd.DataFrame(feats)
    result.index = pd.to_datetime(result.index)
    for col in result.columns:
        result[col] = result[col].astype(np.float32)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def predict_rankings(
    stock_data: dict, macro_features: pd.DataFrame,
    model, feature_cols: list, logger, report: RunReport,
) -> list[tuple[str, float]]:
    """Generate predictions for all stocks, return sorted rankings."""
    report.start_step("predict")
    predictions = {}
    failures = []

    for sym, df in stock_data.items():
        try:
            stock_feats = compute_stock_features(df)

            # Merge with macro
            if not macro_features.empty:
                merged = stock_feats.join(macro_features, how="left")
                macro_cols = macro_features.columns
                merged[macro_cols] = merged[macro_cols].ffill()
            else:
                merged = stock_feats

            # Get latest row
            latest = merged.iloc[-1:]
            avail_cols = [c for c in feature_cols if c in merged.columns]

            if len(avail_cols) < len(feature_cols) * 0.9:
                failures.append((sym, "too many missing features"))
                continue

            row = latest[avail_cols]
            if row.isna().any(axis=1).iloc[0]:
                row = row.fillna(0)

            # Ensure column order matches training
            missing = [c for c in feature_cols if c not in row.columns]
            for mc in missing:
                row[mc] = 0
            row = row[feature_cols]

            pred = model.predict(row)[0]
            if np.isfinite(pred):
                predictions[sym] = float(pred)
            else:
                failures.append((sym, "non-finite prediction"))

        except Exception as e:
            failures.append((sym, str(e)))

    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"  Predictions: {len(ranked)} ok, {len(failures)} failed")
    if failures and len(failures) <= 10:
        for sym, reason in failures:
            logger.info(f"    SKIP {sym}: {reason}")

    # Prediction distribution stats
    if predictions:
        preds_arr = np.array(list(predictions.values()))
        report.set("prediction_range", {
            "min": float(np.min(preds_arr)),
            "max": float(np.max(preds_arr)),
            "mean": float(np.mean(preds_arr)),
            "std": float(np.std(preds_arr)),
            "median": float(np.median(preds_arr)),
        })
        logger.info(f"  Prediction range: {np.min(preds_arr):.2f}% to "
                    f"{np.max(preds_arr):.2f}% "
                    f"(mean: {np.mean(preds_arr):.2f}%, "
                    f"std: {np.std(preds_arr):.2f}%)")

    report.set("n_predictions", len(ranked))
    report.set("prediction_failures", len(failures))

    if ranked:
        logger.info(f"  Top 5:    {[(s, f'{p:+.2f}%') for s, p in ranked[:5]]}")
        logger.info(f"  Bottom 5: {[(s, f'{p:+.2f}%') for s, p in ranked[-5:]]}")

    report.end_step("predict")
    return ranked


# ═══════════════════════════════════════════════════════════════════════════
# ALPACA TRADING
# ═══════════════════════════════════════════════════════════════════════════

def get_alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }


def alpaca_request(method: str, endpoint: str, data=None, logger=None):
    """Make an Alpaca API request with logging."""
    import requests
    url = f"{ALPACA_BASE_URL}/{endpoint}"
    headers = get_alpaca_headers()

    if logger:
        logger.info(f"    API: {method} {endpoint}")

    if method == "GET":
        resp = requests.get(url, headers=headers)
    elif method == "POST":
        resp = requests.post(url, headers=headers, json=data)
    elif method == "DELETE":
        resp = requests.delete(url, headers=headers)
    else:
        raise ValueError(f"Unknown method: {method}")

    if resp.status_code not in (200, 204, 207):
        err_msg = f"Alpaca {method} {endpoint}: {resp.status_code} {resp.text}"
        if logger:
            logger.error(f"    {err_msg}")
        raise Exception(err_msg)

    if resp.status_code == 204:
        return {}
    return resp.json()


def get_account(logger, report: RunReport):
    """Get Alpaca account info."""
    acct = alpaca_request("GET", "v2/account", logger=logger)
    pv = float(acct["portfolio_value"])
    cash = float(acct["cash"])
    bp = float(acct["buying_power"])
    logger.info(f"  Account: portfolio=${pv:,.2f}, "
                f"cash=${cash:,.2f}, buying_power=${bp:,.2f}")
    report.data.setdefault("rebalance", {}).update({
        "portfolio_value": pv, "cash": cash, "buying_power": bp,
    })
    return acct


def get_positions(logger) -> dict[str, dict]:
    """Get current positions with details."""
    positions = alpaca_request("GET", "v2/positions", logger=logger)
    pos_dict = {}
    total_value = 0
    total_pl = 0

    for p in positions:
        mv = float(p["market_value"])
        pl = float(p["unrealized_pl"])
        pos_dict[p["symbol"]] = {
            "qty": float(p["qty"]),
            "market_value": mv,
            "unrealized_pl": pl,
            "unrealized_pl_pct": float(p.get("unrealized_plpc", 0)) * 100,
            "avg_entry": float(p.get("avg_entry_price", 0)),
            "current_price": float(p.get("current_price", 0)),
            "side": p["side"],
        }
        total_value += mv
        total_pl += pl

    logger.info(f"  Current positions: {len(pos_dict)} stocks, "
                f"value=${total_value:,.2f}, unrealized P&L=${total_pl:,.2f}")

    if pos_dict:
        # Show each position
        for sym, info in sorted(pos_dict.items()):
            logger.info(f"    {sym:6s}: {info['qty']:>8.2f} shares, "
                        f"${info['market_value']:>10,.2f}, "
                        f"P&L: ${info['unrealized_pl']:>+8,.2f} "
                        f"({info['unrealized_pl_pct']:>+5.1f}%)")

    return pos_dict


def rebalance_portfolio(
    target_symbols: list[str], logger, report: RunReport,
    dry_run: bool = False,
):
    """Rebalance to equal-weight long portfolio of target symbols."""
    report.start_step("rebalance")
    rb_data = report.data.setdefault("rebalance", {})
    rb_data["dry_run"] = dry_run

    if dry_run:
        # In dry-run mode, use mock account data so we don't need Alpaca credentials
        logger.info("  DRY RUN: using mock account (portfolio=$100,000)")
        portfolio_value = 100_000.0
        current_positions = {}
        rb_data["portfolio_value"] = portfolio_value
        rb_data["cash"] = portfolio_value
        rb_data["buying_power"] = portfolio_value
        rb_data["positions_before"] = 0
    else:
        acct = get_account(logger, report)
        portfolio_value = float(acct["portfolio_value"])
        current_positions = get_positions(logger)
        rb_data["positions_before"] = len(current_positions)

    target_set = set(target_symbols)
    current_set = set(current_positions.keys())

    to_sell = sorted(current_set - target_set)
    to_buy = sorted(target_set - current_set)
    to_hold = sorted(current_set & target_set)

    target_weight = portfolio_value / len(target_symbols)
    rb_data["target_weight"] = target_weight

    orders = []
    n_rebalanced = 0
    n_held_unchanged = 0

    # 1. Sell positions not in target
    logger.info(f"\n  Sells ({len(to_sell)} positions to exit):")
    for sym in to_sell:
        qty = current_positions[sym]["qty"]
        mv = current_positions[sym]["market_value"]
        pl = current_positions[sym]["unrealized_pl"]
        logger.info(f"    SELL {sym}: {qty:.2f} shares, "
                    f"${mv:,.2f}, P&L: ${pl:+,.2f}")
        if not dry_run:
            orders.append(("sell", sym, qty))

    # 2. Check held positions — rebalance if needed
    logger.info(f"\n  Holds ({len(to_hold)} positions to check):")
    for sym in to_hold:
        current_value = current_positions[sym]["market_value"]
        diff = target_weight - current_value
        drift_pct = abs(diff) / target_weight * 100

        if abs(diff) > target_weight * 0.1:
            direction = "BUY more" if diff > 0 else "TRIM"
            logger.info(f"    REBALANCE {sym}: ${current_value:,.0f} → "
                        f"${target_weight:,.0f} (drift: {drift_pct:.0f}%, {direction} ${abs(diff):,.0f})")
            n_rebalanced += 1
            if not dry_run:
                if diff > 0:
                    orders.append(("buy_notional", sym, diff))
                else:
                    orders.append(("sell_notional", sym, abs(diff)))
        else:
            logger.info(f"    HOLD {sym}: ${current_value:,.0f} "
                        f"(drift: {drift_pct:.0f}%, within 10% — no action)")
            n_held_unchanged += 1

    # 3. Buy new positions
    logger.info(f"\n  Buys ({len(to_buy)} new positions):")
    for sym in to_buy:
        logger.info(f"    BUY {sym}: ${target_weight:,.0f} (new position)")
        if not dry_run:
            orders.append(("buy_notional", sym, target_weight))

    # Turnover calculation
    total_positions = max(len(target_set) + len(current_set), 1)
    turnover = (len(to_sell) + len(to_buy)) / total_positions

    rb_data.update({
        "n_sells": len(to_sell), "n_buys": len(to_buy),
        "n_rebalanced": n_rebalanced, "n_held": n_held_unchanged,
        "sells_detail": to_sell, "buys_detail": to_buy,
        "turnover": turnover,
    })

    logger.info(f"\n  Summary: {len(to_sell)} sells, {len(to_buy)} buys, "
                f"{n_rebalanced} rebalanced, {n_held_unchanged} held, "
                f"turnover: {turnover:.0%}")

    if dry_run:
        logger.info("  MODE: DRY RUN — no orders sent")
        report.end_step("rebalance")
        return rb_data

    # Execute orders
    logger.info(f"\n  Executing {len(orders)} orders...")
    executed = 0
    failed = 0

    for action, sym, amount in orders:
        try:
            if action == "sell":
                alpaca_request("DELETE", f"v2/positions/{sym}", logger=logger)
                logger.info(f"    OK: closed {sym}")
            elif action == "buy_notional":
                resp = alpaca_request("POST", "v2/orders", {
                    "symbol": sym, "notional": round(amount, 2),
                    "side": "buy", "type": "market", "time_in_force": "day",
                }, logger=logger)
                logger.info(f"    OK: buy {sym} ${amount:,.2f} "
                            f"(order: {resp.get('id', '?')})")
            elif action == "sell_notional":
                resp = alpaca_request("POST", "v2/orders", {
                    "symbol": sym, "notional": round(amount, 2),
                    "side": "sell", "type": "market", "time_in_force": "day",
                }, logger=logger)
                logger.info(f"    OK: sell {sym} ${amount:,.2f} "
                            f"(order: {resp.get('id', '?')})")
            executed += 1
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"    FAIL: {action} {sym} — {e}")
            report.add_error(f"Order failed: {action} {sym} — {e}")
            failed += 1

    rb_data.update({"executed": executed, "failed": failed})
    logger.info(f"  Orders done: {executed} executed, {failed} failed")
    report.end_step("rebalance")
    return rb_data


# ═══════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    """Load pipeline state from JSON."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"last_rebalance": None, "last_run": None, "run_count": 0, "history": []}


def save_state(state: dict):
    """Save pipeline state to JSON."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def should_rebalance(state: dict, force: bool = False) -> bool:
    """Check if we should rebalance today."""
    if force:
        return True
    last = state.get("last_rebalance")
    if last is None:
        return True
    last_date = datetime.fromisoformat(last)
    days_since = (datetime.now() - last_date).days
    return days_since >= HORIZON


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(dry_run: bool = False, force: bool = False):
    """Execute the full daily pipeline."""
    logger, log_file = setup_logging()
    report = RunReport()

    logger.info("=" * 70)
    logger.info(f"  ML TRADING PIPELINE")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Config: H{HORIZON}_LongOnly{TOP_N}")
    logger.info(f"  Mode: {'DRY RUN' if dry_run else 'LIVE PAPER TRADING'}")
    logger.info(f"  Force rebalance: {force}")
    logger.info(f"  Log: {log_file}")
    logger.info("=" * 70)

    # Check API keys
    if not dry_run and (not ALPACA_API_KEY or not ALPACA_SECRET_KEY):
        logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set!")
        report.add_error("Missing API keys")
        logger.info(report.format_summary())
        sys.exit(1)

    # ── Step 1: Load model ──
    logger.info("\n[1/5] LOADING MODEL")
    report.start_step("load_model")
    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        report.add_error(f"Model not found: {MODEL_PATH}")
        logger.info(report.format_summary())
        sys.exit(1)

    model_bundle = pickle.load(open(MODEL_PATH, "rb"))
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    logger.info(f"  Model: {len(feature_cols)} features, "
                f"horizon={model_bundle['horizon']}d, "
                f"task={model_bundle.get('task', '?')}")
    logger.info(f"  Trained: {model_bundle.get('saved_at', '?')}")
    report.set("n_features", len(feature_cols))
    report.end_step("load_model")

    # ── Step 2: Check state ──
    state = load_state()
    logger.info(f"\n  State: run #{state.get('run_count', 0) + 1}, "
                f"last rebalance: {state.get('last_rebalance', 'never')}")

    if not should_rebalance(state, force):
        last = datetime.fromisoformat(state["last_rebalance"])
        days_since = (datetime.now() - last).days
        days_until = HORIZON - days_since
        logger.info(f"  Skipping — last rebalance {days_since}d ago, "
                     f"next in {days_until}d")
        state["last_run"] = datetime.now().isoformat()
        save_state(state)
        report.end_step("load_model")
        logger.info(report.format_summary())
        return

    # ── Step 3: Download data ──
    logger.info("\n[2/5] DOWNLOADING MARKET DATA")
    symbols = get_tradeable_symbols(logger, report)

    if not symbols:
        logger.error("No symbols found — cannot proceed")
        report.add_error("Universe is empty")
        logger.info(report.format_summary())
        sys.exit(1)

    stock_data = download_bars(symbols, LOOKBACK_DAYS, logger, report)
    macro_data = download_macro(LOOKBACK_DAYS, logger, report)

    if not stock_data:
        logger.error("No stock data downloaded — cannot proceed")
        report.add_error("No stock data available")
        logger.info(report.format_summary())
        sys.exit(1)

    # ── Step 4: Compute features ──
    logger.info("\n[3/5] COMPUTING FEATURES")
    report.start_step("compute_features")
    macro_features = compute_macro_features(macro_data)
    logger.info(f"  Macro features: {len(macro_features.columns)} columns, "
                f"{len(macro_features)} days")

    # Log latest macro values for context
    if not macro_features.empty:
        latest_macro = macro_features.iloc[-1]
        logger.info("  Latest macro snapshot:")
        for col in sorted(macro_features.columns):
            val = latest_macro[col]
            if np.isfinite(val):
                logger.info(f"    {col:25s}: {val:>10.3f}")
    report.end_step("compute_features")

    # ── Step 5: Generate predictions ──
    logger.info("\n[4/5] GENERATING PREDICTIONS")
    rankings = predict_rankings(
        stock_data, macro_features, model, feature_cols, logger, report,
    )

    if len(rankings) < TOP_N:
        logger.error(f"Only {len(rankings)} predictions — need at least {TOP_N}")
        report.add_error(f"Insufficient predictions: {len(rankings)} < {TOP_N}")
        logger.info(report.format_summary())
        sys.exit(1)

    target_symbols = [sym for sym, _ in rankings[:TOP_N]]
    report.set("target_portfolio", rankings[:TOP_N])

    logger.info(f"\n  Target portfolio ({TOP_N} stocks):")
    for i, (sym, pred) in enumerate(rankings[:TOP_N]):
        logger.info(f"    {i+1:2d}. {sym:6s}  {pred:+6.2f}%")

    # ── Step 6: Rebalance ──
    logger.info("\n[5/5] REBALANCING PORTFOLIO")
    result = rebalance_portfolio(target_symbols, logger, report, dry_run=dry_run)

    # ── Save state ──
    state["last_rebalance"] = datetime.now().isoformat()
    state["last_run"] = datetime.now().isoformat()
    state["run_count"] = state.get("run_count", 0) + 1
    state["history"].append({
        "date": datetime.now().isoformat(),
        "target_symbols": target_symbols,
        "predictions": {s: round(p, 4) for s, p in rankings[:TOP_N]},
        "result": {k: v for k, v in result.items()
                   if k not in ("sells_detail", "buys_detail")},
    })
    state["history"] = state["history"][-100:]
    save_state(state)

    # ── Final summary ──
    summary = report.format_summary()
    logger.info(summary)
    logger.info(f"Log saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Trading Pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Predict only, no orders")
    parser.add_argument("--force", action="store_true",
                        help="Force rebalance even if not due")
    args = parser.parse_args()

    try:
        run_pipeline(dry_run=args.dry_run, force=args.force)
    except Exception as e:
        logging.getLogger("pipeline").error(
            f"FATAL: {e}\n{traceback.format_exc()}"
        )
        sys.exit(1)
