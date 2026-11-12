"""Compact fixed-bin relation forecasts shared verbatim by both providers."""
import json
from typing import Final
VERSION: Final = 'ag_observation_fullrank_bins3_v1'
BIN_COUNT: Final = 3
RANK_ENCODING: Final = 'attention logits log((3-rank)/6); spatial/contact score1-rank/group_size'

def output_schema(object_count: int) -> dict:
    rank = lambda maximum: {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': maximum}, 'minItems': maximum + 1, 'maxItems': maximum + 1}
    segment = {'type': 'object', 'properties': {'attention': rank(2), 'spatial': rank(5), 'contact': rank(16)}, 'required': ['attention', 'spatial', 'contact'], 'additionalProperties': False}
    obj = {'type': 'object', 'properties': {'id': {'type': 'integer', 'minimum': 0, 'maximum': max(0, object_count - 1)}, 'class_id': {'type': 'integer', 'minimum': 2, 'maximum': 36}, 'bins': {'type': 'array', 'items': segment, 'minItems': 3, 'maxItems': 3}}, 'required': ['id', 'class_id', 'bins'], 'additionalProperties': False}
    return {'type': 'object', 'properties': {'objects': {'type': 'array', 'items': obj, 'minItems': object_count, 'maxItems': object_count}}, 'required': ['objects'], 'additionalProperties': False}

def parse(text: str, request: dict) -> dict:
    value = json.loads(text)
    if set(value) != {'objects'} or not isinstance(value['objects'], list):
        raise ValueError('invalid_root')
    expected = {o['object_id']: o for o in request['objects']}
    if len(value['objects']) != len(expected):
        raise ValueError('object_count')
    seen = set()
    for obj in value['objects']:
        if set(obj) != {'id', 'class_id', 'bins'} or type(obj['id']) is not int or obj['id'] not in expected or (obj['id'] in seen):
            raise ValueError('object_identity')
        seen.add(obj['id'])
        if type(obj['class_id']) is not int or not 2 <= obj['class_id'] <= 36:
            raise ValueError('class_range')
        if request['mode'] == 'predcls' and obj['class_id'] != expected[obj['id']]['class_id']:
            raise ValueError('given_class_changed')
        if not isinstance(obj['bins'], list) or len(obj['bins']) != BIN_COUNT:
            raise ValueError('bin_count')
        for segment in obj['bins']:
            if set(segment) != {'attention', 'spatial', 'contact'}:
                raise ValueError('attention')
            for (key, limit) in [('attention', 3), ('spatial', 6), ('contact', 17)]:
                values = segment[key]
                if not isinstance(values, list) or len(values) != limit or any((type(v) is not int or not 0 <= v < limit for v in values)) or (len(set(values)) != limit):
                    raise ValueError('unique_ranked_relations')
    return value
parse_prediction = parse
