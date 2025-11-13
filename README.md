
## Installation
Create / activate a virtual environment (recommended) then install dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirement.txt
```
> NOTE: The file is named `requirement.txt` (singular). If you prefer the conventional name, copy/rename it to `requirements.txt`.

# Basic run with defaults
python predict_options_enhanced.py

# Conservative (high confidence only)
python predict_options_enhanced.py --min-confidence 0.75 --min-accuracy 0.60

# Aggressive (more picks, lower thresholds)
python predict_options_enhanced.py --min-confidence 0.55 --min-accuracy 0.52

# NIFTY 50 only, top 5 picks
python predict_options_enhanced.py --nifty50-only --top 5

# Extended history for better training
python predict_options_enhanced.py --period 2y