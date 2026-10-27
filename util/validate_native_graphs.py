"""Use established GLM TP logits/state and local-stage oracles with graphs enabled."""
import validate_native_tp as validation
import os
from profile_final_ir import set_mode as graph_mode, rank_report

original_mode, original_report = validation.mode, validation.report


def mode(context, enabled, trace=False, reset=True):
    original_mode(context, enabled, trace, reset)
    graph_mode(context, enabled, enabled, os.environ.get('EXL3_NATIVE_TP_HC_FUSION', '0') == '1')


def report(context):
    result = original_report(context)
    graphs = rank_report(context)
    result['graph_captures'] = graphs['graph_captures']
    result['graph_replays'] = graphs['graph_replays']
    result['fused_graph_captures'] = graphs['fused_graph_captures']
    result['fused_graph_replays'] = graphs['fused_graph_replays']
    executions = sum(b['executions'] for b in result['blocks'])
    if context.get('final_ir_graphs') and executions:
        assert result['graph_replays'] == 2 * executions, 'A native HC stage bypassed graph replay'
        fused = os.environ.get('EXL3_NATIVE_TP_HC_FUSION', '0') == '1'
        assert result['fused_graph_replays'] == (result['graph_replays'] if fused else 0)
    return result


validation.mode, validation.report = mode, report


if __name__ == '__main__':
    import torch
    with torch.inference_mode():
        validation.main()
