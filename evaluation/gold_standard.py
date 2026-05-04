import csv
import os
from typing import Dict, List

from config.settings import GOLD_STANDARD_PATH


def evaluation_available() -> bool:
    return bool(GOLD_STANDARD_PATH) and os.path.isfile(GOLD_STANDARD_PATH)


def load_gold_standard() -> List[Dict[str, str]]:
    if not evaluation_available():
        return []
    with open(GOLD_STANDARD_PATH, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]
