"""개발 선택에서 동일 ADE·악화된 FDE/충돌·다른 제공률을 거절한다."""
import json
from adapter_io import ROOT
from select_methods import passes

base={'provided':100,'ADE':1.,'FDE':2.,'ColI_observed_GT_percent':3.}
cases=[('improved',{'ADE':.9},True),('equal_ADE',{},False),('worse_FDE',{'ADE':.9,'FDE':2.001},False),('worse_collision',{'ADE':.9,'ColI_observed_GT_percent':3.001},False),('different_coverage',{'ADE':.9,'provided':99},False)]
records=[]
for name,changes,expected in cases:
    value={**base,**changes};actual=passes(value,base);assert actual==expected,name;records.append({'case':name,'actual':actual,'expected':expected})
near={**base,'ADE':1.-5e-7};assert passes(near,base) and not passes(near,base,1e-6)
result={'passed':True,'cases':records,'legacy_epsilon_differs_from_strict_ours':True}
(ROOT/'pilot/selection_gate_test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
