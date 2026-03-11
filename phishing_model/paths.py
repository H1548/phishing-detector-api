from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = str(BASE_DIR / "artifacts" / "tokenizer" / "tokenizer.json")

Advice_PATH = str(BASE_DIR / "phishing_model" / "Advice.json")

Params_Path = str(BASE_DIR / "artifacts" / "MultiFineBest2" / "BestModel.tar")