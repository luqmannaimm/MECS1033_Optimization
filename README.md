# MECS1033_Optimization
MECS1033 Advanced Artificial Intelligence: Assignment 3 - Optimization
Runs an Ant Colony Optimization (ACO) solver to find a short route that visits several points of interest using straight line (Haversine) distance, then saves an interactive HTML map.

## Requirements
- Python 3.9+

## Install (Windows PowerShell)
From the project folder:

1) Create and activate a virtual environment
```powershell
python -m venv _venv
./_venv/Scripts/Activate.ps1
```

2) Install dependencies
```powershell
python -m pip install -r requirements.txt
```

## Run
```powershell
python .\optimization.py
```

When it finishes:
- The console prints the best route and total distance (km)
- The map is saved as `aco_best_route.html`
- Open `aco_best_route.html` in a browser to interact with the map.

## Notes
- The route cost is computed using straight line distance only, not actual road routes.
- `matplotlib` is required because the downloaded ACO code imports it.
