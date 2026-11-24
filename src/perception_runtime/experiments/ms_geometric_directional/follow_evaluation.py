"""Start the frozen 40k evaluation once both training jobs pass."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def main():
    run=Path(sys.argv[1]).resolve()
    while True:
        phase=json.loads((run/"status.json").read_text())
        if phase["phase"]=="failed":
            raise RuntimeError(phase)
        if phase["phase"]=="training_complete":
            break
        time.sleep(15)
    env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    command=[sys.executable,"-u",str(HERE/"evaluate_directional.py"),"--run",str(run),
             "--step","40000","--output",str(run/"evaluation_40000")]
    with (run/"evaluation.log").open("w") as log:
        subprocess.run(command,check=True,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)

if __name__=="__main__":
    main()
