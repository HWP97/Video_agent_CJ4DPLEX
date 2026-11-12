"""현재 fold에서 parallel/masked/AR/MTP를 학습하고 관측만으로 추론한다."""
from adapter_fast import main

if __name__ == '__main__':
    main('masked', ('parallel', 'masked', 'ar', 'mtp'))
