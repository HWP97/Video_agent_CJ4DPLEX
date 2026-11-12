# b2 고정 기준

기존 VEATIC 현재 인식 후 hold다. 모델 코드·기존 vendor·런타임이 이 폴더에 있으며 가중치와 관측 자료는 `data/b2`에 있다. 계열 루트에서 `python3 main.py b2 baseline replay` 또는 `CUDA_VISIBLE_DEVICES=1 python3 main.py b2 baseline predict --recognize`를 실행한다. 상세 검증과 제한은 `reports/b2/README.md`에 있다.
