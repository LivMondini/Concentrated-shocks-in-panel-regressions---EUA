import zipfile
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PASTA = ROOT / "dados_brutos" / "sec"
SAIDA = ROOT / "dados_tratados" / "sec_longo.csv"

TAGS = {
    "Assets", "AssetsCurrent",
    "Liabilities",
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "Cash",
    "ShortTermInvestments", "MarketableSecuritiesCurrent",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    "PropertyPlantAndEquipmentNet",
    "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
    "IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet",
    "DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings",
    "LongTermDebtAndCapitalLeaseObligationsCurrent",
    "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
    "LongTermNotesPayable", "LongTermDebt",
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
    "OperatingIncomeLoss",
    "LongTermDebt", "NotesPayableCurrent", "LoansPayableCurrent",
    "ConvertibleNotesPayableCurrent", "ConvertibleDebtCurrent",
    "LineOfCredit", "LongTermLineOfCredit", "SecuredDebt", "SubordinatedDebt",
    "NotesPayable", "ConvertibleLongTermNotesPayable", "OtherBorrowings",
}

COLS_SUB = ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed"]
COLS_NUM = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "segments", "coreg", "value"]

arquivos = sorted(PASTA.glob("*.zip"))
print(f"{len(arquivos)} arquivos encontrados")

partes = []
for zp in arquivos:
    with zipfile.ZipFile(zp) as z:
        with z.open("sub.txt") as f:
            sub = pd.read_csv(f, sep="\t", dtype=str, usecols=COLS_SUB)
        with z.open("num.txt") as f:
            num = pd.read_csv(f, sep="\t", dtype=str, usecols=COLS_NUM)

    sub = sub[sub["form"].isin(["10-K", "10-Q"])]
    num = num[
        num["tag"].isin(TAGS)
        & num["segments"].isna()
        & num["coreg"].isna()
        & (num["uom"] == "USD")
        & num["version"].str.startswith("us-gaap", na=False)
    ]
    num = num.merge(sub, on="adsh")
    num = num[num["ddate"] == num["period"]]
    num["arquivo"] = zp.stem
    partes.append(num.drop(columns=["version", "uom", "segments", "coreg", "ddate"]))
    print(f"{zp.name}: {len(sub)} relatórios, {len(num)} valores mantidos")

longo = pd.concat(partes, ignore_index=True)
longo["value"] = pd.to_numeric(longo["value"], errors="coerce")
longo = longo.dropna(subset=["value"])
longo = longo.sort_values("filed").drop_duplicates(
    ["cik", "period", "tag", "qtrs"], keep="last"
)

SAIDA.parent.mkdir(parents=True, exist_ok=True)
longo.to_csv(SAIDA, index=False)
print(f"\nGravado {SAIDA}: {len(longo)} linhas, {longo['cik'].nunique()} firmas")
print("Períodos de", longo["period"].min(), "a", longo["period"].max())
