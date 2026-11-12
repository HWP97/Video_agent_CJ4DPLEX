"""선택을 바꾸지 않고 현재 인식오차와 미래 변화량 오차의 signed 항등식을 채점한다."""
import numpy as np
from common import ROOT,read,save,digest,guard
from learning import metric

def main():
    guard('score');before=digest(ROOT/'all_frozen.json');frozen=read(ROOT/'all_frozen.json');selected=frozen['selected']
    rows=read(ROOT/'manifest.json');splits=np.array([r['split'] for r in rows]);table={r['sample_id']:r for split in ['train','dev','eval'] for r in read(ROOT/f'labels_{split}.json')}
    y=np.array([table[r['sample_id']]['future'] for r in rows]);current_gt=np.array([table[r['sample_id']]['current'] for r in rows]);current=np.load(ROOT/'inputs.npz')['recognition'][:,-1]
    assert np.max(np.abs(y))<=1 and np.max(np.abs(current_gt))<=1
    error_current=current-current_gt;truth_change=y-current_gt[:,None,:]
    histories=[];decompositions=[];identity_max=0.
    for split in ['train','dev','official_test','additional']:
        mask=splits==split;a=current_gt[mask];p=current[mask]
        ccc=2*((a-a.mean(0))*(p-p.mean(0))).mean(0)/np.maximum(a.var(0)+p.var(0)+(a.mean(0)-p.mean(0))**2,1e-15)
        histories.append({'split':split,'n':int(mask.sum()),'current_MAE':float(np.abs(error_current[mask]).mean()),
            'current_MAE_axes':np.abs(error_current[mask]).mean(0).tolist(),'current_CCC_mean2':float(ccc.mean()),'current_CCC_axes':ccc.tolist(),
            'current_signed_bias_axes':error_current[mask].mean(0).tolist(),
            'GT_current_hold_oracle':metric(y[mask],np.repeat(current_gt[mask,None,:],4,1))})
    for path in sorted((ROOT/'predictions').glob('*.npz')):
        forecast=np.load(path)['prediction'];future_error=forecast-y
        change_error=(forecast-current[:,None,:])-truth_change
        difference=float(np.abs(future_error-error_current[:,None,:]-change_error).max());identity_max=max(identity_max,difference)
        assert difference<1e-12
        for split in ['train','dev','official_test','additional']:
            mask=splits==split;ce=np.broadcast_to(error_current[mask,None,:],future_error[mask].shape);de=change_error[mask];fe=future_error[mask]
            cross=2*float((ce*de).mean());identity=float((ce**2).mean()+(de**2).mean()+cross)
            assert abs(identity-float((fe**2).mean()))<1e-12
            decompositions.append({'method':path.stem,'split':split,'n':int(mask.sum()),'current_MAE':float(np.abs(ce).mean()),
                'change_error_MAE':float(np.abs(de).mean()),'future_error_MAE':float(np.abs(fe).mean()),
                'current_signed_mean_axes':ce.mean((0,1)).tolist(),'change_signed_mean_dimensions':de.mean(0).ravel().tolist(),
                'future_signed_mean_dimensions':fe.mean(0).ravel().tolist(),'current_MSE':float((ce**2).mean()),
                'change_MSE':float((de**2).mean()),'twice_cross_term':cross,'future_MSE':float((fe**2).mean())})
    assert digest(ROOT/'all_frozen.json')==before
    save(ROOT/'recognition_change_decomposition.json',{'selected_unchanged':selected,'all_frozen_sha256_before_after':before,
        'signed_identity_max_abs_error':identity_max,'equation':'forecast-futureGT = (currentEstimate-currentGT) + ((forecast-currentEstimate)-(futureGT-currentGT))',
        'current_history_metrics':histories,'methods':decompositions,'MAE_subtraction_used':False,'new_training_or_inference':False,
        'current_estimate':'raw final recognizer output, not clipped','forecast':'same bounded predictions used by primary scores'})
    lines=['# 현재 인식과 미래 변화의 실패 분석','',f'개발에서 동결한 `{selected}`와 기존 선택 파일을 유지했다. 새 학습·추론·튜닝은 수행하지 않았다.',
      '', '실제 signed 오차는 다음 항등식으로 분리했다. MAE끼리 빼거나 MAE를 더해 기여도를 계산하지 않았다.',
      '', '`미래오차 = (현재추정−현재GT) + [(예측미래−현재추정)−(미래GT−현재GT)]`',
      '',f'전체55방법·2740창에서 항등식 최대 수치오차는 {identity_max:.9g}다. MSE는 두 제곱항과 교차항도 따로 저장했다.',
      '', '| split | n | 현재 MAE↓ | 현재 CCC2↑ | signed bias V | signed bias A | GT-current-hold 미래 MAE↓ |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for r in histories:
        lines.append(f"| {r['split']} | {r['n']} | {r['current_MAE']:.6f} | {r['current_CCC_mean2']:.6f} | {r['current_signed_bias_axes'][0]:.6f} | {r['current_signed_bias_axes'][1]:.6f} | {r['GT_current_hold_oracle']['MAE']:.6f} |")
    lines+=['','GT-current-hold는 실제 현재 정답을 미래에도 유지하는 채점용 oracle이다. 배포 입력이나 학습 없는 영상 baseline이 아니며, 현재 인식이 완벽하다는 가정 아래 자연 상태 변화가 만드는 오차를 보여준다.',
       '', '| split | 방법 | 현재오차 MAE | 변화량오차 MAE | 미래오차 MAE |', '|---|---|---:|---:|---:|']
    for r in decompositions:
        if r['split'] in ['dev','official_test','additional'] and r['method'] in ['hold','localpool_ridge',selected]:
            lines.append(f"| {r['split']} | {r['method']} | {r['current_MAE']:.6f} | {r['change_error_MAE']:.6f} | {r['future_error_MAE']:.6f} |")
    lines+=['','이 세 MAE 열은 가산 관계가 아니다. 현재오차와 변화량오차가 서로 상쇄하거나 증폭할 수 있으므로 signed 평균·교차항 없이 단순 차감으로 원인을 정하지 않는다.',
       '', 'dev265는 head 학습 원본과 분리했지만 공개 backbone이 이미 학습한 첫98영상 앞70%에 속한다. 따라서 dev의 현재인식 성능이 낙관적일 가능성이 있다. 뒤30%로 갈 때 현재인식 오차와 상태분포도 달라지므로 dev/test 격차를 미래 head 하나의 문제로 단정할 수 없다. 이 관찰만으로 backbone 노출의 인과효과를 확정하지 않는다.',
       '', '추가26대상은13원본·434source-time마다2개 GT가 있으나 full-scene 입력은 동일하다. 개인별 구별이 없는 현재 인식의 한계가 남는다. 대상별 표와 공통예측의 GT-pair MAE 하한은 target_scores.json 및 additional_pair_diagnostic.json에 별도 보존했다.',
       '', '선택 GRU의 공식438 MAE 감소와 CCC 하락이라는 결론은 그대로다. 이 분석으로 다른 후보를 선택하거나 하이퍼파라미터를 바꾸지 않았다.']
    (ROOT/'FAILURE_ANALYSIS.md').write_text('\n'.join(lines)+'\n');print('signed identity all55x2740 PASS; frozen selection unchanged; max_error='+str(identity_max),flush=True)

if __name__=='__main__':main()
