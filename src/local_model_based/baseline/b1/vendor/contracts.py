"""Named shapes for saved metrics and scientific configuration, without a framework."""
from typing import TypeAlias,TypedDict


class Statistics(TypedDict):
    frames:int
    R_sum:list[float]
    class_sum:list[list[float]]
    class_frames:list[int]
    hits:list[list[int]]
    counts:list[int]
    object_hits:list[int]
    object_counts:list[int]
    top20_family:list[int]


class Score(TypedDict):
    R:float|None
    mR:float|None


Metrics:TypeAlias=dict[str,Score]
Subsets:TypeAlias=dict[str,Statistics]
CandidateGrid:TypeAlias=dict[str,tuple[float,float]|None]


class Complementarity(TypedDict):
    both:list[int]
    hold_only:list[int]
    sol_only:list[int]
    neither:list[int]
    count:list[int]


class A1Choice(TypedDict):
    name:str
    weights:tuple[float,float]|None
    R:float
    mR:float
    distance:float
    complexity:int


class A2Choice(TypedDict,total=False):
    alpha:float
    R:float
    mR:float
    reference:str
    reference_metrics:Score
    adopted:bool


class ModeSelection(TypedDict):
    a1:A1Choice
    a1_trials:list[A1Choice]
    a2:A2Choice
    a2_trials:list[A2Choice]


class Selection(TypedDict,total=False):
    sgcls:ModeSelection
    predcls:ModeSelection
    created_utc:str
    development_sha256:str
    freeze_sha256:str


class Result(TypedDict,total=False):
    sample_id:str
    mode:str
    video_id:str
    ratio:float
    future_count:int
    group:str
    split:str
    stage:str
    status:str
    methods:dict[str,dict[str,Subsets]]
    states:dict[str,str]
    seconds:float
    input_signature:str
    cutoff_s:float
    postprocess_signature:str
    complementarity:Complementarity
    a1_duplicate_original:bool
    a1_a2_duplicate_a2:bool
