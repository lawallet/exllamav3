"""Unprofiled generation with CPU-placement snapshots outside timed regions."""
import json
import os
from pathlib import Path
import sys
import time

import profile_final_ir as driver
from profile_tp_scheduling import placement

original_report = driver.rank_report


def report(context):
    result = original_report(context)
    result['cpu_placement'] = placement()
    path = Path(os.environ['EXL3_R3_BASELINE_OUTPUT']) / f'placement-rank{context["rank"]}.jsonl'
    with path.open('a') as f:
        f.write(json.dumps(dict(epoch_ns=time.time_ns(),device=context['device'],**result['cpu_placement']))+'\n')
    return result


driver.rank_report = report


if __name__ == '__main__':
    os.environ['EXL3_R3_BASELINE_OUTPUT'] = str(Path(sys.argv[sys.argv.index('--output')+1]).resolve())
    driver.main()
