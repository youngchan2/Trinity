### 실행 방법
`convert_eval_rms.py`
- `--n`: case number
- `--m`: model type (falcon / llama)
- `--t`: eval type (vanilla / prenorm / keyformer / qknorm / roco)
- `--o`: 0 코드 생성 1 코드 실행 (py파일 이미 생성되어 있는 경우만 가능) 2 코드 생성 및 실행 (default option: 0)
- `--pre`: 입력하면 prenorm 있는 경우, 입력하지 않으면 prenorm 없는 경우
- 각 실행 결과마다 계산 결과와 시간 함께 출력됨

`evaluation`
- 디렉토리 안에 각 tyep별 디렉토리가 있고 여기에 test case와 변환된 triton code 파일 생성됨
- 논문 eval에 넣었던 gpu 5개에 대한 best case에 대한 txt 파일은 모두 저장해뒀음
- 추가 다른 case로 실험 원하는 경우 `convert_eval_rms.py`를 참고하거나 기존 파일 양식을 보고 파일 이름을 정해주면 됨