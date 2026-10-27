"""Matched native eager/stage-graph target replay via existing validation oracles.

The output field `native=False` means eager IR, `native=True` means stage graphs.
At TP3 the first timed trajectory supplies the identical inputs for later runs.
Use --bench-query 4 for target verification rows; this does not measure MTP
draft execution or acceptance. No profiling is active in timed regions.
"""
import validate_native_tp as validation
from profile_final_ir import set_mode as graph_mode, rank_report
from profile_tp_scheduling import placement

original_mode, original_report = validation.mode, validation.report


def mode(context, enabled, trace=False, reset=True):
    original_mode(context, True, trace, reset)
    graph_mode(context, True, enabled)


def report(context):
    result = original_report(context)
    graphs = rank_report(context)
    result['graph_captures'] = graphs['graph_captures']
    result['graph_replays'] = graphs['graph_replays']
    result['cpu_placement'] = placement()
    executions = sum(b['executions'] for b in result['blocks'])
    if context.get('final_ir_graphs') and executions:
        assert result['graph_replays'] == 2*executions
    return result


validation.mode, validation.report = mode, report


if __name__ == '__main__':
    import torch
    with torch.inference_mode(): validation.main()
