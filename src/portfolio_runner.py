#!/usr/bin/env python3
"""
portfolio_runner.py
───────────────────
Runs the congressional-trade portfolio analysis every 30 minutes.

Usage
─────
  # Make executable (macOS / Linux)
  chmod +x portfolio_runner.py

  # Run directly
  ./portfolio_runner.py

  # Or via Python
  python3 portfolio_runner.py

  # Optional flags
  python3 portfolio_runner.py --interval 60        # run every 60 minutes
  python3 portfolio_runner.py --run-once           # single run then exit
  python3 portfolio_runner.py --log-dir ./logs     # custom log directory
  python3 portfolio_runner.py --output-dir ./out   # custom output directory

Stop
────
  Ctrl-C  (or send SIGTERM)  →  graceful shutdown after the current run.

Outputs (written after every run)
──────────────────────────────────
  <output-dir>/metrics_<timestamp>.json          — full metrics as JSON
  <output-dir>/trades_<timestamp>.csv            — processed trades DataFrame
  <output-dir>/latest_metrics.json               — always overwritten; latest run
  <output-dir>/report_<timestamp>.pdf            — full PDF report
  Google Drive → PortfolioReports/report_<ts>.pdf — auto-uploaded to Drive
  <log-dir>/portfolio_runner.log                 — rotating log (10 MB x 5 files)
"""

import argparse
import json
import logging
import logging.handlers
import os
import random
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for daemon use
import matplotlib.pyplot as plt
from scipy.stats import linregress

# reportlab — PDF generation (install: pip install reportlab)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, PageBreak,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults  (all overridable via CLI flags)
# ─────────────────────────────────────────────────────────────────────────────
INTERVAL_MINUTES = 30
DATA_URL = "https://drive.google.com/uc?export=download&id=1OR3ePHUjT9oI2tP_UaVosZR7xtgftm6i"
RISK_FREE_RATE   = 0.048    # annualised; US T-bill 2023 average
ULCER_PERIOD     = 14
DRIVE_FOLDER     = "PortfolioReports"   # subfolder inside My Drive


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("portfolio")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler — 10 MB max, keep 5 backups
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "portfolio_runner.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Yahoo Finance helpers  (per publicapis.io/blog/yahoo-finance-api-guide)
# ─────────────────────────────────────────────────────────────────────────────

class YFinanceClient:
    """
    Thin, cached wrapper around yfinance.

    Uses a persistent requests.Session with a real browser User-Agent so
    Yahoo Finance does not block requests from CI/datacenter IP ranges
    (GitHub Actions runners are on Azure IPs that YF rate-limits heavily
    when the default Python/yfinance UA is used).

    Each ticker's full history and sector are fetched once per run via the
    cache — redundant HTTP calls are eliminated automatically.
    """

    # A real browser UA is the single most reliable fix for YF blocking in CI.
    # Rotate this string if blocking resumes.
    _USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    _MAX_RETRIES = 3          # retry attempts per ticker on empty response
    _RETRY_DELAY = 5          # seconds to wait between retries

    def __init__(self, logger: logging.Logger):
        self._history_cache: dict[str, pd.Series] = {}
        self._sector_cache:  dict[str, str]       = {}
        self.log     = logger
        self._session = self._build_session()

    def _build_session(self):
        """
        Create a requests.Session with browser headers and mount it into
        yfinance so every HTTP call uses the same session automatically.
        """
        import requests
        session = requests.Session()
        session.headers.update({
            "User-Agent":      self._USER_AGENT,
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection":      "keep-alive",
        })
        return session

    def clear_cache(self):
        """Call at the start of each run so fresh data is fetched."""
        self._history_cache.clear()
        self._sector_cache.clear()
        # Rebuild session to clear any stale cookies
        self._session = self._build_session()

    def fetch_history(self, ticker: str, start: str = "2020-01-01") -> pd.Series:
        """
        Download full daily adjusted-close history for *ticker*.
        Uses ticker.history() with a browser session and exponential backoff.
        Returns pd.Series indexed by 'YYYY-MM-DD' date strings.
        """
        ticker = ticker.upper()
        if ticker in self._history_cache:
            return self._history_cache[ticker]

        self.log.debug("Fetching history for %s from Yahoo Finance...", ticker)
        series = pd.Series(dtype=float)

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                t  = yf.Ticker(ticker, session=self._session)
                df = t.history(
                    start=start, end=None, interval="1d", auto_adjust=True
                )
                if df.empty:
                    self.log.warning(
                        "No data returned for %s (attempt %d/%d).",
                        ticker, attempt, self._MAX_RETRIES,
                    )
                    if attempt < self._MAX_RETRIES:
                        time.sleep(self._RETRY_DELAY * attempt)
                        continue
                else:
                    series = df["Close"].copy()
                    series.index = pd.to_datetime(series.index).strftime("%Y-%m-%d")
                    self.log.debug(
                        "%s: %d trading days loaded (%s to %s).",
                        ticker, len(series),
                        series.index.min(), series.index.max(),
                    )
                    break
            except Exception as exc:
                self.log.error(
                    "Error fetching %s (attempt %d/%d): %s",
                    ticker, attempt, self._MAX_RETRIES, exc,
                )
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_DELAY * attempt)

        self._history_cache[ticker] = series
        return series

    def get_price(self, ticker: str, date: str) -> float:
        """
        Adjusted close price on *date* (YYYY-MM-DD).
        Searches up to 7 days forward for the next trading day.
        """
        series = self.fetch_history(ticker)
        target = datetime.strptime(date, "%Y-%m-%d")
        for offset in range(7):
            candidate = (target + timedelta(days=offset)).strftime("%Y-%m-%d")
            if candidate in series.index:
                return float(series[candidate])
        raise ValueError(
            f"No trading data for {ticker} within 7 days of {date}."
        )

    def get_sector(self, ticker: str) -> str:
        """Sector string via ticker.info['sector'] (per YF API guide)."""
        ticker = ticker.upper()
        if ticker in self._sector_cache:
            return self._sector_cache[ticker]
        try:
            sector = yf.Ticker(ticker).info.get("sector", "Unknown") or "Unknown"
        except Exception as exc:
            self.log.warning("Could not fetch sector for %s: %s", ticker, exc)
            sector = "Unknown"
        self._sector_cache[ticker] = sector
        return sector

    def get_spy(self, start: str, end: str) -> pd.Series:
        """SPY daily adjusted closes filtered to [start, end]."""
        spy = self.fetch_history("SPY", start=start)
        spy.index = pd.to_datetime(spy.index)
        mask = (spy.index >= pd.Timestamp(start)) & (spy.index <= pd.Timestamp(end))
        return spy[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Analyser
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioAnalyser:

    def __init__(self, yf_client: YFinanceClient, logger: logging.Logger,
                 output_dir: Path, reporter: "PDFReporter | None" = None,
                 uploader: "DriveUploader | None" = None):
        self.yf       = yf_client
        self.log      = logger
        self.out      = output_dir
        self.reporter = reporter
        self.uploader = uploader
        self._beta    = 0.0
        self.out.mkdir(parents=True, exist_ok=True)

    # ── Data loading ─────────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        self.log.info("Loading transaction data…")
        df = pd.read_csv(DATA_URL)
        df = df.drop(columns=[
            "disclosureYear", "owner", "disclosureDate",
            "assetDescription", "representative", "district",
        ], errors="ignore")
        df = df.rename(columns={"transactionDate": "Date"})
        df["Date"]   = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df["ticker"] = df["ticker"].str.replace(" ", "", regex=False)
        df["type"]   = df["type"].apply(self._convert_type)
        df["amount"] = df["amount"].apply(self._extract_avg_amount)
        return df

    @staticmethod
    def _extract_avg_amount(amount_range: str) -> float:
        parts = amount_range.replace("$", "").replace(",", "").split(" - ")
        return (int(parts[0]) + int(parts[1])) / 2

    @staticmethod
    def _convert_type(t: str) -> str:
        if t == "Purchase":            return "Buy"
        if t in ("Sale (Full)",
                 "Sale (Partial)"):    return "Sell"
        return t

    def _fetch_size_price(self, row) -> tuple[int, float]:
        try:
            price = self.yf.get_price(row["ticker"], row["Date"])
            size  = ceil(float(row["amount"]) / price)
            return size, price
        except Exception as exc:
            self.log.warning("Skipping %s on %s: %s", row["ticker"], row["Date"], exc)
            return 0, 0.0

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.info("Fetching prices for %d rows (%d unique tickers)…",
                      len(df), df["ticker"].nunique())
        df[["size", "price"]] = df.apply(
            self._fetch_size_price, axis=1, result_type="expand"
        )
        df = df[df["price"] > 0].copy()
        df = df.drop("amount", axis=1)
        df = df.rename(columns={"ticker": "Symbol", "type": "Side"})
        df = df[["Date", "Symbol", "Side", "size", "price"]]
        df = df.rename(columns={"size": "Size", "price": "Price"})
        df = df.sort_values("Date").reset_index(drop=True)
        self.log.info("DataFrame ready: %d rows after price fetch.", len(df))
        return df

    # ── Metrics ───────────────────────────────────────────────────────────────

    def total_return(self, df: pd.DataFrame) -> dict:
        invested_value = max_invested = long_ret = short_ret = 0.0
        long_pos = {}; short_pos = {}
        pct_rets = []; ret_vals = []

        for _, row in df.iterrows():
            txn_val = row["Size"] * row["Price"]
            sym, side = row["Symbol"], row["Side"]
            daily_ret = 0.0

            if side == "Buy":
                long_pos.setdefault(sym, []).append((row["Price"], row["Size"]))
                invested_value += txn_val
                if sym in short_pos:
                    rem = row["Size"]
                    while rem > 0 and short_pos[sym]:
                        sp, ss = short_pos[sym][0]
                        if ss <= rem:
                            daily_ret += (sp - row["Price"]) * ss; rem -= ss; short_pos[sym].pop(0)
                        else:
                            daily_ret += (sp - row["Price"]) * rem; short_pos[sym][0] = (sp, ss - rem); rem = 0
                    short_ret += daily_ret

            elif side == "Sell":
                short_pos.setdefault(sym, []).append((row["Price"], row["Size"]))
                invested_value -= txn_val
                if sym in long_pos:
                    rem = row["Size"]
                    while rem > 0 and long_pos[sym]:
                        bp, bs = long_pos[sym][0]
                        if bs <= rem:
                            daily_ret += (row["Price"] - bp) * bs; rem -= bs; long_pos[sym].pop(0)
                        else:
                            daily_ret += (row["Price"] - bp) * rem; long_pos[sym][0] = (bp, bs - rem); rem = 0
                    long_ret += daily_ret

            max_invested = max(max_invested, invested_value)
            pct_rets.append((daily_ret / invested_value * 100) if invested_value else 0.0)
            ret_vals.append(daily_ret)

        df["Percent Return"] = pct_rets
        df["Return Value"]   = ret_vals
        total = long_ret + short_ret
        return {
            "Return":         round(total, 2),
            "Return Percent": round((total / max_invested * 100) if max_invested else 0, 2),
        }

    def sharpe(self, df: pd.DataFrame, rf: float) -> float:
        s = df["Percent Return"]
        return abs(round((s.mean() * 255 - rf) / (s.std() * np.sqrt(255)), 2))

    def sector_allocation(self, df: pd.DataFrame, save_path: Path) -> dict:
        sector_map       = {t: self.yf.get_sector(t) for t in df["Symbol"].unique()}
        df["Sector"]     = df["Symbol"].map(sector_map)
        alloc            = df.groupby("Sector")["Size"].sum()

        fig, ax = plt.subplots(figsize=(10, 7))
        alloc.plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_title("Sector Allocation of the Portfolio")
        ax.set_ylabel("")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.log.info("Sector chart saved → %s", save_path)
        return alloc.to_dict()

    def alpha_beta(self, df: pd.DataFrame) -> tuple[float, float]:
        spy_price = self.yf.get_spy(df["Date"].min(), df["Date"].max())
        spy_ret   = spy_price.pct_change().dropna()
        port_ret  = df.set_index("Date")["Percent Return"]

        matched = [
            spy_ret.loc[min(spy_ret.index, key=lambda x: abs(x - pd.Timestamp(d)))]
            for d in port_ret.index
        ]
        spy_aligned = pd.Series(matched, index=port_ret.index)
        slope, intercept, *_ = linregress(spy_aligned.values, port_ret.values)
        return abs(round(intercept, 2)), abs(round(slope, 2))   # alpha, beta

    def std_dev(self, df: pd.DataFrame) -> float:
        return round(df["Percent Return"].std() * np.sqrt(252), 2)

    def var(self, df: pd.DataFrame) -> dict:
        r = df["Return Value"]
        return {
            "VaR 95%": abs(round(np.percentile(r, 5), 2)),
            "VaR 99%": abs(round(np.percentile(r, 1), 2)),
        }

    def max_drawdown(self, series: pd.Series) -> float:
        comp = (series + 1).cumprod()
        peak = comp.expanding(min_periods=1).max()
        return abs(((comp / peak) - 1).min())

    def calmar(self, df: pd.DataFrame) -> float:
        return abs((df["Return Value"].mean() * 255) / self.max_drawdown(df["Return Value"]))

    def ulcer_index(self, series: pd.Series, period: int) -> float:
        dd = []
        for i in range(len(series)):
            peak = series[: i + 1].max()
            dd.append(((peak - series[: i + 1].min()) / peak) ** 2 if peak else 0)
        return float(np.sqrt(np.mean(dd) * period))

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log.info("═" * 60)
        self.log.info("Run started  %s", ts)

        # 1. Load & prepare
        raw_df = self.load_data()
        df     = self.prepare_dataframe(raw_df)

        # 2. Guard: abort cleanly if no rows survived price fetch
        if df.empty:
            self.log.error(
                "DataFrame is empty after price fetch — all rows were skipped. "
                "This usually means Yahoo Finance blocked every request from this "
                "IP range. Check the warnings above for per-ticker details. "
                "Run will be marked as failed."
            )
            raise RuntimeError(
                "No trade data available — cannot compute metrics on empty DataFrame."
            )

        # 3. Compute metrics
        self.log.info("Computing metrics...")

        ret           = self.total_return(df)
        sharpe_ratio  = self.sharpe(df, RISK_FREE_RATE)
        chart_path    = self.out / f"sector_allocation_{ts}.png"
        sector_alloc  = self.sector_allocation(df, chart_path)
        alpha, beta   = self.alpha_beta(df)
        std           = self.std_dev(df)
        var_metrics   = self.var(df)
        max_dd        = self.max_drawdown(df["Percent Return"])
        calmar_ratio  = self.calmar(df)
        ulcer         = self.ulcer_index(df["Percent Return"], ULCER_PERIOD)

        metrics = {
            "run_timestamp": ts,
            "Profitability Indices": {
                "Total Return":       ret,
                "Sharpe Ratio":       sharpe_ratio,
                "Sector Allocation":  sector_alloc,
            },
            "Risk Measure Indices": {
                "Alpha":              alpha,
                "Beta":               beta,
                "Standard Deviation": std,
                "Value-at-Risk":      var_metrics,
            },
            "Drawdown Based Risk Measures": {
                "Maximum Drawdown": round(float(max_dd), 4),
                "Calmar Ratio":     round(float(calmar_ratio), 4),
                "Ulcer Index":      round(float(ulcer), 4),
            },
        }

        # 3. Save outputs
        metrics_path = self.out / f"metrics_{ts}.json"
        latest_path  = self.out / "latest_metrics.json"
        trades_path  = self.out / f"trades_{ts}.csv"

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        with open(latest_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        df.drop(columns=["Sector"], errors="ignore").to_csv(trades_path, index=False)

        self.log.info("Metrics saved  → %s", metrics_path)
        self.log.info("Trades saved   → %s", trades_path)

        # 4. Generate PDF and upload to Drive
        pdf_path = self.out / f"report_{ts}.pdf"
        if self.reporter:
            try:
                self.reporter.build(pdf_path, metrics, df, chart_path)
                if self.uploader:
                    drive_dest = self.uploader.upload(pdf_path)
                    if drive_dest:
                        self.log.info("Drive path    -> %s", drive_dest)
            except Exception as exc:
                self.log.error("PDF/Drive step failed: %s", exc)

        # 5. Print summary to console
        self.log.info("─" * 60)
        self.log.info("RESULTS")
        self.log.info("  Total Return:       %s", ret)
        self.log.info("  Sharpe Ratio:       %s", sharpe_ratio)
        self.log.info("  Alpha:              %s", alpha)
        self.log.info("  Beta:               %s", beta)
        self.log.info("  Std Dev:            %s", std)
        self.log.info("  VaR:                %s", var_metrics)
        self.log.info("  Max Drawdown:       %.4f", max_dd)
        self.log.info("  Calmar Ratio:       %.4f", calmar_ratio)
        self.log.info("  Ulcer Index:        %.4f", ulcer)
        self.log.info("Run complete.")

        return metrics



# ─────────────────────────────────────────────────────────────────────────────
# PDF Report Generator  (reportlab Platypus — per PDF skill)
# ─────────────────────────────────────────────────────────────────────────────

class PDFReporter:
    """
    Builds a multi-page portfolio report PDF using reportlab Platypus.

    Pages
    ─────
    1. Cover — title, timestamp, headline KPIs
    2. Profitability — Total Return, Sharpe Ratio table
    3. Risk Metrics   — Alpha, Beta, Std Dev, VaR table
    4. Drawdown Risk  — Max DD, Calmar, Ulcer table
    5. Sector Chart   — pie chart image embedded full-width
    6. Top Trades     — table of first 20 processed rows
    """

    PAGE_W, PAGE_H = A4
    MARGIN = 2 * cm

    # Brand colours
    C_DARK  = colors.HexColor("#0D1B2A")   # deep navy
    C_BLUE  = colors.HexColor("#1B4F72")   # header blue
    C_LIGHT = colors.HexColor("#D6EAF8")   # row tint
    C_GREEN = colors.HexColor("#1E8449")   # positive value
    C_RED   = colors.HexColor("#922B21")   # negative value
    C_LINE  = colors.HexColor("#2980B9")

    def __init__(self, logger: logging.Logger):
        self.log    = logger
        self.styles = getSampleStyleSheet()
        self._build_styles()

    def _build_styles(self):
        self.s = {
            "cover_title": ParagraphStyle(
                "cover_title",
                parent=self.styles["Title"],
                fontSize=28, textColor=self.C_DARK,
                spaceAfter=6, leading=34,
            ),
            "cover_sub": ParagraphStyle(
                "cover_sub",
                parent=self.styles["Normal"],
                fontSize=13, textColor=colors.HexColor("#5D6D7E"),
                spaceAfter=4,
            ),
            "section": ParagraphStyle(
                "section",
                parent=self.styles["Heading1"],
                fontSize=14, textColor=self.C_BLUE,
                spaceBefore=14, spaceAfter=6, leading=18,
            ),
            "body": ParagraphStyle(
                "body", parent=self.styles["Normal"],
                fontSize=10, leading=14, textColor=self.C_DARK,
            ),
            "kpi_label": ParagraphStyle(
                "kpi_label", parent=self.styles["Normal"],
                fontSize=10, textColor=colors.HexColor("#5D6D7E"),
            ),
            "kpi_value": ParagraphStyle(
                "kpi_value", parent=self.styles["Normal"],
                fontSize=18, textColor=self.C_DARK, leading=22,
            ),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _hr(self):
        return HRFlowable(
            width="100%", thickness=1, color=self.C_LINE,
            spaceAfter=8, spaceBefore=4,
        )

    def _section(self, title: str):
        return [
            Paragraph(title, self.s["section"]),
            self._hr(),
        ]

    @staticmethod
    def _fmt(value, pct=False) -> str:
        if isinstance(value, float):
            return f"{value:.2f}%" if pct else f"{value:,.4f}"
        if isinstance(value, dict):
            return "  |  ".join(f"{k}: {v}" for k, v in value.items())
        return str(value)

    def _metric_table(self, rows: list[tuple]) -> Table:
        """Two-column key/value table with alternating row shading."""
        data = [["Metric", "Value"]] + list(rows)
        col_w = [(self.PAGE_W - 2 * self.MARGIN) * f for f in (0.55, 0.45)]
        t = Table(data, colWidths=col_w, repeatRows=1)
        style = [
            # Header row
            ("BACKGROUND",  (0, 0), (-1, 0), self.C_BLUE),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            # Body rows
            ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 1), (-1, -1), 9),
            ("TOPPADDING",  (0, 1), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, self.C_LIGHT]),
            ("ALIGN",       (1, 0), (1, -1), "RIGHT"),
        ]
        t.setStyle(TableStyle(style))
        return t

    def _trades_table(self, df: pd.DataFrame, n: int = 20) -> Table:
        sample = df.head(n)
        header = list(sample.columns)
        data   = [header] + sample.astype(str).values.tolist()
        n_cols = len(header)
        col_w  = [(self.PAGE_W - 2 * self.MARGIN) / n_cols] * n_cols
        t = Table(data, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), self.C_BLUE),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 7),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#BDC3C7")),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, self.C_LIGHT]),
            ("ALIGN",         (3, 1), (-1, -1), "RIGHT"),
        ]))
        return t

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, pdf_path: Path, metrics: dict,
              df: pd.DataFrame, chart_path: Path) -> Path:
        """
        Render the full report to *pdf_path* and return it.
        """
        self.log.info("Building PDF report -> %s", pdf_path)

        doc   = SimpleDocTemplate(
            str(pdf_path), pagesize=A4,
            leftMargin=self.MARGIN, rightMargin=self.MARGIN,
            topMargin=self.MARGIN,  bottomMargin=self.MARGIN,
        )
        story = []
        ts    = metrics.get("run_timestamp", "")
        prof  = metrics.get("Profitability Indices", {})
        risk  = metrics.get("Risk Measure Indices", {})
        dd    = metrics.get("Drawdown Based Risk Measures", {})
        ret   = prof.get("Total Return", {})

        # ── Page 1: Cover ─────────────────────────────────────────────────────
        story += [
            Spacer(1, 1.5 * cm),
            Paragraph("Congressional Trade Portfolio", self.s["cover_title"]),
            Paragraph("Quantitative Performance Report", self.s["cover_sub"]),
            Paragraph(f"Generated: {ts.replace('_', '  ')}  |  Interval: every 30 min",
                      self.s["cover_sub"]),
            self._hr(),
            Spacer(1, 0.8 * cm),
        ]

        # Headline KPI strip
        kpis = [
            ("Total Return ($)",  f"${ret.get('Return', 0):,.2f}"),
            ("Total Return (%)",  f"{ret.get('Return Percent', 0):.2f}%"),
            ("Sharpe Ratio",      str(prof.get("Sharpe Ratio", "N/A"))),
            ("Alpha",             str(risk.get("Alpha", "N/A"))),
            ("Beta",              str(risk.get("Beta",  "N/A"))),
            ("Max Drawdown",      f"{dd.get('Maximum Drawdown', 0):.4f}"),
        ]
        kpi_data  = [[Paragraph(k, self.s["kpi_label"]) for k, _ in kpis],
                     [Paragraph(v, self.s["kpi_value"]) for _, v in kpis]]
        kpi_w     = [(self.PAGE_W - 2 * self.MARGIN) / len(kpis)] * len(kpis)
        kpi_table = Table(kpi_data, colWidths=kpi_w)
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), self.C_LIGHT),
            ("BOX",           (0, 0), (-1, -1), 1, self.C_BLUE),
            ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#AED6F1")),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ]))
        story += [kpi_table, PageBreak()]

        # ── Page 2: Profitability ──────────────────────────────────────────────
        story += self._section("Profitability Indices")
        story.append(self._metric_table([
            ("Total Return ($)",      f"${ret.get('Return', 0):,.2f}"),
            ("Total Return (%)",      f"{ret.get('Return Percent', 0):.2f}%"),
            ("Sharpe Ratio",          str(prof.get("Sharpe Ratio", "N/A"))),
        ]))
        story.append(PageBreak())

        # ── Page 3: Risk Metrics ───────────────────────────────────────────────
        story += self._section("Risk Measure Indices")
        var_d = risk.get("Value-at-Risk", {})
        story.append(self._metric_table([
            ("Alpha",             str(risk.get("Alpha", "N/A"))),
            ("Beta",              str(risk.get("Beta",  "N/A"))),
            ("Standard Deviation (annualised)", str(risk.get("Standard Deviation", "N/A"))),
            ("VaR 95%",           f"${var_d.get('VaR 95%', 0):,.2f}"),
            ("VaR 99%",           f"${var_d.get('VaR 99%', 0):,.2f}"),
        ]))
        story.append(PageBreak())

        # ── Page 4: Drawdown Risk ──────────────────────────────────────────────
        story += self._section("Drawdown Based Risk Measures")
        story.append(self._metric_table([
            ("Maximum Drawdown",  f"{dd.get('Maximum Drawdown', 0):.4f}"),
            ("Calmar Ratio",      f"{dd.get('Calmar Ratio', 0):.4f}"),
            ("Ulcer Index",       f"{dd.get('Ulcer Index', 0):.4f}"),
        ]))
        story.append(PageBreak())

        # ── Page 5: Sector Chart ───────────────────────────────────────────────
        story += self._section("Sector Allocation")
        if chart_path.exists():
            max_w   = self.PAGE_W - 2 * self.MARGIN
            story.append(RLImage(str(chart_path), width=max_w, height=max_w * 0.7))
        else:
            story.append(Paragraph("(Sector chart not available)", self.s["body"]))
        story.append(PageBreak())

        # ── Page 6: Trade Log ──────────────────────────────────────────────────
        story += self._section("Processed Trade Log (first 20 rows)")
        display_df = df.drop(columns=["Sector", "Percent Return",
                                      "Return Value"], errors="ignore")
        story.append(self._trades_table(display_df, n=20))

        doc.build(story)
        self.log.info("PDF ready: %s  (%.1f KB)", pdf_path, pdf_path.stat().st_size / 1024)
        return pdf_path


# ─────────────────────────────────────────────────────────────────────────────
# Google Drive Uploader
# ─────────────────────────────────────────────────────────────────────────────

class DriveUploader:
    """
    Uploads PDF reports to Google Drive using the Drive API v3 and a
    service-account credential.

    Credential resolution order (first match wins):
      1. GDRIVE_SERVICE_ACCOUNT_JSON env var  — JSON string (GitHub Actions secret)
      2. GDRIVE_KEY_FILE env var              — path to a local JSON key file
      3. google.colab mount fallback          — for interactive Colab use

    The target folder is set via:
      - GDRIVE_FOLDER_ID env var  (recommended — unambiguous)
      - or the folder name string passed at construction (Drive search fallback)
    """

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    def __init__(self, folder_name: str, logger: logging.Logger):
        self.folder_name = folder_name
        self.log         = logger
        self._service    = None
        self._folder_id  = os.environ.get("GDRIVE_FOLDER_ID", "")
        self._init_service()

    # ── Credential / service initialisation ───────────────────────────────────

    def _init_service(self):
        """Try all credential strategies in order."""
        if self._try_service_account_env():
            return
        if self._try_service_account_file():
            return
        self._try_colab_fallback()

    def _build_service(self, credentials):
        from googleapiclient.discovery import build
        self._service = build("drive", "v3", credentials=credentials)
        self.log.info("Drive API service initialised.")
        if not self._folder_id:
            self._folder_id = self._resolve_folder_id()

    def _try_service_account_env(self) -> bool:
        """Load credentials from the GDRIVE_SERVICE_ACCOUNT_JSON env var."""
        json_str = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON", "")
        if not json_str:
            return False
        try:
            from google.oauth2 import service_account
            info = json.loads(json_str)
            creds = service_account.Credentials.from_service_account_info(
                info, scopes=self.SCOPES
            )
            self.log.info("Drive: using service account from env var.")
            self._build_service(creds)
            return True
        except Exception as exc:
            self.log.error("Drive: service account env var failed: %s", exc)
            return False

    def _try_service_account_file(self) -> bool:
        """Load credentials from a local key file path in GDRIVE_KEY_FILE."""
        key_file = os.environ.get("GDRIVE_KEY_FILE", "")
        if not key_file or not Path(key_file).exists():
            return False
        try:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                key_file, scopes=self.SCOPES
            )
            self.log.info("Drive: using service account key file: %s", key_file)
            self._build_service(creds)
            return True
        except Exception as exc:
            self.log.error("Drive: key file auth failed: %s", exc)
            return False

    def _try_colab_fallback(self):
        """Last resort: mount Drive the Colab way and use shutil.copy2."""
        try:
            from google.colab import drive as colab_drive
            drive_root = Path("/content/drive/My Drive")
            if not drive_root.exists():
                self.log.info("Drive: mounting via Colab...")
                colab_drive.mount("/content/drive")
            dest_dir = drive_root / self.folder_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            self._colab_dest = dest_dir
            self.log.info("Drive: Colab mount ready at %s", dest_dir)
        except ModuleNotFoundError:
            self.log.warning(
                "Drive: no credentials found and not in Colab. "
                "PDFs will be saved locally only."
            )
            self._colab_dest = None

    # ── Folder resolution ──────────────────────────────────────────────────────

    def _resolve_folder_id(self) -> str:
        """Search Drive for the folder by name; return its ID."""
        try:
            q = (f"name='{self.folder_name}' "
                 "and mimeType='application/vnd.google-apps.folder' "
                 "and trashed=false")
            result = self._service.files().list(
                q=q, fields="files(id, name)", pageSize=1
            ).execute()
            files = result.get("files", [])
            if files:
                fid = files[0]["id"]
                self.log.info("Drive: resolved folder '%s' -> %s",
                              self.folder_name, fid)
                return fid
            self.log.warning(
                "Drive: folder '%s' not found. "
                "Set GDRIVE_FOLDER_ID or share the folder with the service account.",
                self.folder_name,
            )
        except Exception as exc:
            self.log.error("Drive: folder lookup failed: %s", exc)
        return ""

    # ── Upload ─────────────────────────────────────────────────────────────────

    def upload(self, pdf_path: Path) -> str | None:
        """
        Upload *pdf_path* to Drive.
        Returns the Drive file ID on success, None on failure.
        """
        # API path
        if self._service:
            return self._upload_via_api(pdf_path)

        # Colab shutil fallback
        if hasattr(self, "_colab_dest") and self._colab_dest:
            return self._upload_via_colab(pdf_path)

        self.log.warning("Drive: no upload method available; skipping.")
        return None

    def _upload_via_api(self, pdf_path: Path) -> str | None:
        from googleapiclient.http import MediaFileUpload
        try:
            meta = {
                "name": pdf_path.name,
                "mimeType": "application/pdf",
            }
            if self._folder_id:
                meta["parents"] = [self._folder_id]

            media = MediaFileUpload(str(pdf_path), mimetype="application/pdf",
                                    resumable=True)
            file_ = self._service.files().create(
                body=meta, media_body=media, fields="id,webViewLink"
            ).execute()
            fid  = file_.get("id")
            link = file_.get("webViewLink", "")
            self.log.info("Uploaded to Drive  id=%s  link=%s", fid, link)
            return fid
        except Exception as exc:
            self.log.error("Drive API upload failed: %s", exc)
            return None

    def _upload_via_colab(self, pdf_path: Path) -> str | None:
        import shutil
        dest = self._colab_dest / pdf_path.name
        try:
            shutil.copy2(str(pdf_path), str(dest))
            self.log.info("Copied to Drive (Colab): %s", dest)
            return str(dest)
        except Exception as exc:
            self.log.error("Colab Drive copy failed: %s", exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class Scheduler:
    """
    Runs *job* immediately, then repeats every *interval_minutes*.
    Uses threading.Event so the sleep is interruptible by Ctrl-C / SIGTERM.
    A threading.Lock prevents overlapping runs if a job takes longer than
    the interval.
    """

    def __init__(self, job, interval_minutes: int, logger: logging.Logger):
        self._job       = job
        self._interval  = interval_minutes * 60
        self._stop      = threading.Event()
        self._lock      = threading.Lock()
        self.log        = logger

    def _run_job(self):
        if not self._lock.acquire(blocking=False):
            self.log.warning(
                "Previous run still in progress — skipping this cycle."
            )
            return
        try:
            self._job()
        except Exception as exc:
            self.log.exception("Unhandled error during run: %s", exc)
        finally:
            self._lock.release()

    def start(self):
        self.log.info(
            "Scheduler started — running every %d minutes. "
            "Press Ctrl-C to stop.",
            self._interval // 60,
        )
        while not self._stop.is_set():
            thread = threading.Thread(target=self._run_job, daemon=True)
            thread.start()
            thread.join()                         # wait for run to finish
            if self._stop.is_set():
                break
            self.log.info(
                "Next run in %d minutes  (%s).",
                self._interval // 60,
                (datetime.now() + timedelta(seconds=self._interval))
                .strftime("%H:%M:%S"),
            )
            # Interruptible sleep: wakes immediately on stop signal
            self._stop.wait(timeout=self._interval)

        self.log.info("Scheduler stopped.")

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

# ── Config (edit these if not using CLI) ────────────────────────────────────
# When running in Colab / Jupyter, CLI flags are unavailable.
# Set these directly, or pass them when calling main() from a cell.
DEFAULT_INTERVAL   = INTERVAL_MINUTES   # minutes between runs
DEFAULT_LOG_DIR    = Path("./logs")
DEFAULT_OUTPUT_DIR = Path("./output")


def parse_args():
    p = argparse.ArgumentParser(
        description="Portfolio analysis daemon. Runs every N minutes."
    )
    p.add_argument("--interval",    type=int,  default=DEFAULT_INTERVAL)
    p.add_argument("--run-once",    action="store_true")
    p.add_argument("--log-dir",     type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    # parse_known_args silently ignores Colab's kernel args (-f kernel-xxx.json)
    args, _ = p.parse_known_args()
    return args


def _register_signals(scheduler, logger):
    """
    Register SIGINT/SIGTERM for graceful shutdown.
    Skipped silently inside Jupyter/Colab where signal handlers
    must live on the main thread and may already be claimed by IPython.
    """
    def _shutdown(signum, frame):
        logger.info("Signal %s received. Stopping...", signal.Signals(signum).name)
        scheduler.stop()

    try:
        signal.signal(signal.SIGINT,  _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
    except (OSError, ValueError):
        logger.warning(
            "Could not register OS signal handlers "
            "(running inside Jupyter/Colab -- use the Stop button to halt)."
        )


def main(interval=None, run_once=False, log_dir=None, output_dir=None):
    """
    Entry point. Can be called directly from a Colab cell:

        main(interval=30, run_once=True)   # single test run
        main(interval=30)                  # loop every 30 min
    """
    args = parse_args()

    # Keyword arguments override parsed CLI/defaults (useful from Colab cells)
    if interval   is not None: args.interval   = interval
    if run_once:               args.run_once   = True
    if log_dir    is not None: args.log_dir    = Path(log_dir)
    if output_dir is not None: args.output_dir = Path(output_dir)

    logger    = setup_logging(args.log_dir)
    yf_client = YFinanceClient(logger)

    reporter = PDFReporter(logger)
    uploader = DriveUploader(DRIVE_FOLDER, logger)

    def job():
        yf_client.clear_cache()
        PortfolioAnalyser(
            yf_client, logger, args.output_dir,
            reporter=reporter, uploader=uploader,
        ).run()

    if args.run_once:
        logger.info("run_once=True: executing a single analysis.")
        job()
        return

    scheduler = Scheduler(job, args.interval, logger)
    _register_signals(scheduler, logger)
    scheduler.start()


if __name__ == "__main__":
    main()
