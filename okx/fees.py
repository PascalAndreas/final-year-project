"""
Fees for OKX trading.

Standard fees were obtained from this link: https://www.okx.com/en-eu/fees
Options fees were obtained from this link: https://www.okx.com/en-eu/help/okx-to-adjust-options-trading-fees

Standard fee table bound were cited in EUR, while options fee table bounds were cited in USD.
For now, we're ignoring this discrepancy.

I'm not sure which fee table will be applied in practice.
"""

def get_fees(assets: float, volume: float) -> tuple[str, float, float]:
    """Get trading fees based on assets (EUR) or 30-day trading volume (EUR)."""
    if assets >= 5_000_001 or volume >= 50_000_001:
        return ("VIP 8", -0.0005, 0.0008)
    elif assets >= 2_000_001 or volume >= 20_000_001:
        return ("VIP 7", -0.0005, 0.0010)
    elif assets >= 1_000_001 or volume >= 10_000_001:
        return ("VIP 6", -0.0003, 0.0015)
    elif assets >= 500_001 or volume >= 5_000_001:
        return ("VIP 5", 0.0000, 0.0020)
    elif assets >= 250_001 or volume >= 2_500_001:
        return ("VIP 4", 0.0005, 0.0023)
    elif assets >= 50_001 or volume >= 500_001:
        return ("VIP 3", 0.0010, 0.0025)
    elif assets >= 25_001 or volume >= 250_001:
        return ("VIP 2", 0.0015, 0.0028)
    elif assets >= 20_001 or volume >= 100_001:
        return ("VIP 1", 0.0018, 0.0030)
    else:
        return ("Regular", 0.0020, 0.0035)


def get_options_fees(assets: float, volume: float, okb: float = 0) -> tuple[str, float, float]:
    """Get options trading fees based on assets (USD), 30-day trading volume (USD), and OKB holdings."""
    if volume >= 20_000_000_000:
        return ("VIP 8", -0.0001, 0.00013)
    elif volume >= 2_000_000_000:
        return ("VIP 7", -0.0001, 0.00015)
    elif volume >= 1_500_000_000:
        return ("VIP 6", -0.00005, 0.00015)
    elif assets >= 10_000_000 or volume >= 100_000_000:
        return ("VIP 5", 0.0001, 0.0002)
    elif assets >= 5_000_000 or volume >= 50_000_000:
        return ("VIP 4", 0.00015, 0.0002)
    elif assets >= 2_000_000 or volume >= 25_000_000:
        return ("VIP 3", 0.0002, 0.00025)
    elif assets >= 500_000 or volume >= 10_000_000:
        return ("VIP 2", 0.0002, 0.0003)
    elif assets >= 100_000 or volume >= 5_000_000:
        return ("VIP 1", 0.00025, 0.0003)
    elif okb >= 1_000:
        return ("Lvl 5", 0.00026, 0.0003)
    elif okb >= 500:
        return ("Lvl 4", 0.00027, 0.0003)
    elif okb >= 200:
        return ("Lvl 3", 0.00028, 0.0003)
    elif okb >= 100:
        return ("Lvl 2", 0.00029, 0.0003)
    else:
        return ("Lvl 1", 0.0003, 0.0003)

