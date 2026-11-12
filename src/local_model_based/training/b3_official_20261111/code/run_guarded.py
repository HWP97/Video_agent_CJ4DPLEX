"""동결 adapter 소스 변경 없이 최종 재추론의 원시 GT 접근까지 차단한다."""
import os
from pathlib import Path
import runpy
import sys
from adapter_io import ROOT,install_guard


def install(stage: str, fold: int) -> None:
    install_guard(stage,fold)
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            writing=(isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or bool(args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if path.name=='official_v4_train.zip' or path.suffix=='.ndjson' and not writing:raise PermissionError('raw_GT_archive_or_ndjson_forbidden')
            if not path.is_relative_to(ROOT) and path.suffix in ('.npz','.npy','.ndjson','.jsonl','.zip'):raise PermissionError('external_experiment_data_forbidden')
            if 'ground_truth' in path.parts or path.name=='futures.jsonl':raise PermissionError('future_ground_truth_forbidden')
            if writing and not any(path.is_relative_to(base) for base in (ROOT,Path('/tmp'),Path('/dev'))):raise PermissionError('write_outside_experiment_forbidden')
    sys.addaudithook(audit)


if __name__=='__main__':
    stage=sys.argv[1];fold=int(sys.argv[2]);target=Path(sys.argv[3]).resolve();assert target.is_relative_to(ROOT/'code')
    install(stage,fold);sys.argv=[str(target),*sys.argv[4:]];runpy.run_path(str(target),run_name='__main__')
