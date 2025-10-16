# python
import json
import optuna
import subprocess
import re
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Resolve JAR path relative to this script
JAR_PATH = Path(__file__).parent.joinpath(
    "PSO_thermoforming", "thermoforming_optimization", "target", "thermoforming_optimization-1.0.jar"
).resolve()

def objective(trial):
    swarmSize = trial.suggest_int('swarmSize', 10, 100)
    maxIters = trial.suggest_int('maxIters', 50, 1000)
    inertias = trial.suggest_float('inertias', 0.1, 1.0)
    c1s = trial.suggest_float('c1s', 0.1, 3.0)
    c2s = trial.suggest_float('c2s', 0.1, 3.0)

    cmd = [
        'java', '-jar', str(JAR_PATH),
        str(swarmSize), str(maxIters), str(inertias), str(c1s), str(c2s)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logging.warning("Subprocess timed out for params: %s", trial.params)
        return float('inf')
    except FileNotFoundError as e:
        logging.error("Failed to run subprocess: %s", e)
        return float('inf')

    if result.returncode != 0:
        logging.warning("Jar exited with code %s. stderr: %s", result.returncode, result.stderr.strip())

    combined = "\n".join([s for s in (result.stdout or "", result.stderr or "") if s is not None])
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    if not lines:
        logging.warning("No output to parse for params: %s", trial.params)
        return float('inf')

    last_line = lines[-1]
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', last_line)
    if not m:
        logging.warning("No numeric value found in last line: %s", last_line)
        return float('inf')

    try:
        fitness = float(m.group(0))
    except ValueError:
        logging.warning("Failed to convert extracted token to float: %s", m.group(0))
        return float('inf')

    return fitness

if __name__ == '__main__':
    # Verify JAR exists before starting tuning
    if not JAR_PATH.is_file():
        logging.error("JAR not found at `%s`. Please build the project or correct the path.", JAR_PATH)
        sys.exit(1)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print('Best hyperparameters:', study.best_params)

    output = {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "direction": study.direction,
    }

    out_path = "best_results.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Saved best results to {out_path}")
    except OSError as e:
        logging.error("Failed to write best results to %s: %s", out_path, e)

    print("Best hyperparameters:", study.best_params)
