# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""Pure observed-graph/future-schedule relation adapter; no GT/files/network access."""
import hashlib
import torch
from application_types import Prediction, Bundle, AdapterInputError
from compact_graph import histories, observed_bank
from relation_model import RelationDynamics
from local_extensions import transform


def observed_logits(graph: dict) -> torch.Tensor:
    mask = graph['pair_frames'] == graph['pair_frames'].max()
    probabilities = torch.cat((graph['spatial_probabilities'][mask], graph['contact_probabilities'][mask]), -1)
    return torch.cat((graph['attention_logits'][mask], torch.logit(probabilities.clamp(1e-7, 1 - 1e-7))), -1)


def predict_relations(graph: dict, future_frame_indices: list[int], bundle: dict,
                      current_hold: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict]:
    history, coverage = histories(graph)
    n = len(future_frame_indices)
    history = history.repeat(n, 1, 1)
    history = transform(history, bundle['method'], bundle.get('memory_prior'))
    delta = torch.tensor([(f - graph['actual_relation_cutoff']) / 30 for f in future_frame_indices], dtype=torch.float32).repeat_interleave(coverage['candidates'])
    if (delta <= 0).any():
        raise AdapterInputError('Nonfuture time requested')
    with torch.random.fork_rng(devices=[]):
        model = RelationDynamics(bundle['residual'], bundle.get('anchor', False)).eval()
    model.load_state_dict(bundle['state_dict'], strict=True)
    seed = int(hashlib.sha256((str(bundle['seed']) + ':' + graph['provenance']['sample_id']).encode()).hexdigest()[:16], 16) % (2 ** 63 - 1)
    generator = torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        future, current = model(history, delta, torch.randn(8, len(history), 64, generator=generator), observed_logits(graph).repeat(n, 1))
    return (current if current_hold else future), current, coverage


def forecast(graph: dict, future_frame_indices: list[int], frozen_bundle: Bundle) -> Prediction:
    """Full benchmark entrypoint. All nonrelation tensors preserve the observed bank."""
    if frozen_bundle.get('mode', graph['mode']) != graph['mode']:
        raise AdapterInputError('Bundle mode does not match observed graph')
    expected_encoder = frozen_bundle.get('encoder_checkpoint_sha256')
    if expected_encoder is not None and graph['provenance']['encoder_checkpoint_sha256'] != expected_encoder:
        raise AdapterInputError('Bundle encoder checkpoint does not match graph')
    result = observed_bank(graph, future_frame_indices)
    logits, _, _ = predict_relations(graph, future_frame_indices, frozen_bundle,
                                    current_hold=frozen_bundle.get('inference_ablation') == 'joint_current_hold')
    result.update(attention_distribution=logits[:, :3], spatial_distribution=logits[:, 3:9].sigmoid(),
                  contacting_distribution=logits[:, 9:].sigmoid())
    if frozen_bundle.get('anchor', False):
        base = observed_logits(graph).repeat(len(future_frame_indices), 1)
        unchanged = logits == base
        mask = graph['pair_frames'] == graph['pair_frames'].max()
        spatial = graph['spatial_probabilities'][mask].repeat(len(future_frame_indices), 1)
        contact = graph['contact_probabilities'][mask].repeat(len(future_frame_indices), 1)
        result['spatial_distribution'] = torch.where(unchanged[:, 3:9], spatial, result['spatial_distribution'])
        result['contacting_distribution'] = torch.where(unchanged[:, 9:], contact, result['contacting_distribution'])
    return result


def reference_forecast(graph: dict, future_frame_indices: list[int], official_prediction: Prediction | None = None) -> Prediction:
    """Same bank official ODE or same encoder's observed-relation hold."""
    result = observed_bank(graph, future_frame_indices)
    if official_prediction is None:
        mask = graph['pair_frames'] == graph['pair_frames'].max()
        n = len(future_frame_indices)
        relation = {'attention_distribution': graph['attention_logits'][mask].repeat(n, 1),
                    'spatial_distribution': graph['spatial_probabilities'][mask].repeat(n, 1),
                    'contacting_distribution': graph['contact_probabilities'][mask].repeat(n, 1)}
    else:
        relation = {k: official_prediction[k].clone() for k in ('attention_distribution', 'spatial_distribution', 'contacting_distribution')}
        assert torch.equal(result['pred_labels'], official_prediction['pred_labels'] if graph['mode'] == 'sgcls' else official_prediction['labels'])
        assert torch.equal(result['pair_idx'], official_prediction['pair_idx'])
        assert torch.equal(result['pred_scores'], official_prediction['pred_scores'] if graph['mode'] == 'sgcls' else official_prediction['scores'])
    result.update(relation)
    return result
