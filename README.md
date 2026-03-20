# 2026-sharpe-ratio

Code to reproduce the numeric examples 
in the paper "Sharpe ratio under GARCH returns".

Run the code as follows: 
```python
uv venv
uv pip install -r requirements.txt
uv run python functions.py
mkdir outputs
for notebook in *.ipynb
do
  uv run papermill "$notebook" outputs/"$notebook"
done
```
