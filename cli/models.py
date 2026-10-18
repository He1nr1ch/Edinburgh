from enum import StrEnum


class AnalystType(StrEnum):
    MARKET = "market"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"


class AssetType(StrEnum):
    STOCK = "stock"
    CRYPTO = "crypto"
