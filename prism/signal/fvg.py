"""
Fair Value Gap (FVG) detector for PRISM.
FVG = 3-candle imbalance where price leaves an unfilled gap.

Bullish FVG: candle[i-2].high < candle[i].low  (demand zone)
Bearish FVG: candle[i-2].low > candle[i].high  (supply zone)

Strategy:
- Identify FVGs on H4 (regime zones)
- Enter on M5/M15 when price retests FVG zone
- Break & retest of FVG boundary = entry trigger
- SL below FVG zone (LONG) or above (SHORT)
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
FVG_STORE = Path("signals/fvg_zones.json")


@dataclass
class FVGZone:
    instrument: str
    timeframe: str          # "H4", "H1", "M15"
    direction: str          # "BULLISH" or "BEARISH"
    top: float              # Upper boundary of gap
    bottom: float           # Lower boundary of gap
    midline: float          # 50% of gap (primary entry target)
    formed_at: str          # ISO datetime when FVG formed
    formed_bar: int         # Bar index when formed
    mitigated: bool = False            # True if price closed through full zone
    partially_mitigated: bool = False  # Price hit midline but closed back
    age_bars: int = 0                  # How many bars since formation
    strength: float = 0.0             # Gap size relative to ATR (larger = stronger)


class FVGDetector:
    """Detects and manages Fair Value Gap zones across timeframes."""

    def __init__(self, instrument: str, timeframe: str = "H4"):
        self.instrument = instrument
        self.timeframe = timeframe
        self.zones: list = []

    def detect(self, df: pd.DataFrame, atr_col: str = "atr_14") -> list:
        """
        Scan OHLCV dataframe for FVG patterns.
        df must have columns: datetime, open, high, low, close
        Returns list of detected FVGZone objects.

        Detection: look at every 3-bar window [i-2, i-1, i]
        Bullish: df[i-2].high < df[i].low  -> gap between them
        Bearish: df[i-2].low > df[i].high  -> gap between them
        """
        zones = []
        # Reset index to ensure iloc == positional index
        df = df.reset_index(drop=True)
        atr = df[atr_col].values if atr_col in df.columns else np.ones(len(df))

        for i in range(2, len(df)):
            candle_prev2 = df.iloc[i - 2]
            candle_curr = df.iloc[i]
            formed_at = str(candle_curr["datetime"])

            # Bullish FVG: gap between prev2 high and curr low
            if candle_prev2["high"] < candle_curr["low"]:
                gap_size = candle_curr["low"] - candle_prev2["high"]
                if gap_size > 0:
                    atr_val = float(atr[i]) if atr[i] > 0 else 1.0
                    zones.append(FVGZone(
                        instrument=self.instrument,
                        timeframe=self.timeframe,
                        direction="BULLISH",
                        top=float(candle_curr["low"]),
                        bottom=float(candle_prev2["high"]),
                        midline=float((candle_curr["low"] + candle_prev2["high"]) / 2),
                        formed_at=formed_at,
                        formed_bar=i,
                        strength=round(gap_size / atr_val, 3),
                    ))

            # Bearish FVG: gap between prev2 low and curr high
            elif candle_prev2["low"] > candle_curr["high"]:
                gap_size = candle_prev2["low"] - candle_curr["high"]
                if gap_size > 0:
                    atr_val = float(atr[i]) if atr[i] > 0 else 1.0
                    zones.append(FVGZone(
                        instrument=self.instrument,
                        timeframe=self.timeframe,
                        direction="BEARISH",
                        top=float(candle_prev2["low"]),
                        bottom=float(candle_curr["high"]),
                        midline=float((candle_prev2["low"] + candle_curr["high"]) / 2),
                        formed_at=formed_at,
                        formed_bar=i,
                        strength=round(gap_size / atr_val, 3),
                    ))

        self.zones = zones
        self._update_mitigation(df)
        logger.info(f"Detected {len(zones)} FVG zones for {self.instrument} {self.timeframe}")
        return zones

    def _update_mitigation(self, df: pd.DataFrame):
        """Mark FVGs as mitigated if price has traded through them."""
        df = df.reset_index(drop=True)
        for zone in self.zones:
            if zone.mitigated:
                continue
            for i in range(zone.formed_bar + 1, len(df)):
                bar = df.iloc[i]
                if zone.direction == "BULLISH":
                    if bar["close"] < zone.bottom:
                        zone.mitigated = True
                        break
                    if bar["low"] <= zone.midline:
                        zone.partially_mitigated = True
                else:  # BEARISH
                    if bar["close"] > zone.top:
                        zone.mitigated = True
                        break
                    if bar["high"] >= zone.midline:
                        zone.partially_mitigated = True
            zone.age_bars = len(df) - zone.formed_bar

    def get_active_zones(self, max_age_bars: int = 50, min_strength: float = 0.1) -> list:
        """Return unmitigated zones within age limit and minimum strength."""
        return [
            z for z in self.zones
            if not z.mitigated
            and z.age_bars <= max_age_bars
            and z.strength >= min_strength
        ]

    def check_entry_trigger(
        self,
        price: float,
        direction: str,
        entry_type: str = "midline",  # "midline" or "boundary"
    ) -> Optional[FVGZone]:
        """
        Check if current price is triggering an entry in an active FVG zone.
        direction: "LONG" or "SHORT"
        Returns the matching zone if entry conditions are met, else None.
        """
        active = self.get_active_zones()
        for zone in active:
            if direction == "LONG" and zone.direction == "BULLISH":
                target = zone.midline if entry_type == "midline" else zone.bottom
                if price <= zone.top and price >= target:
                    return zone
            elif direction == "SHORT" and zone.direction == "BEARISH":
                target = zone.midline if entry_type == "midline" else zone.top
                if price >= zone.bottom and price <= target:
                    return zone
        return None

    def save(self):
        """Persist active zones to JSON store."""
        FVG_STORE.parent.mkdir(exist_ok=True)
        data = {
            "updated_at": datetime.utcnow().isoformat(),
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "zones": [vars(z) for z in self.get_active_zones()],
        }
        with open(FVG_STORE, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data['zones'])} active FVG zones to {FVG_STORE}")

    @classmethod
    def load(cls, instrument: str, timeframe: str = "H4") -> "FVGDetector":
        """Load zones from JSON store."""
        detector = cls(instrument, timeframe)
        if FVG_STORE.exists():
            with open(FVG_STORE) as f:
                data = json.load(f)
            detector.zones = [FVGZone(**z) for z in data.get("zones", [])]
        return detector
