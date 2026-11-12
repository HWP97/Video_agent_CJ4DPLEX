# Video Forecasting Agent — 재현용 컨테이너
# Build: docker build -t video-forecasting-agent .
# Run (CPU):  docker run --rm -it -v /path/to/external_data:/data video-forecasting-agent
# Run (GPU):  docker run --rm -it --gpus all -v /path/to/external_data:/data video-forecasting-agent

FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git ca-certificates fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/ ./requirements/
RUN pip install --no-cache-dir -r requirements/local.txt \
    && pip install --no-cache-dir -r requirements/demo.txt

COPY . .

# 평가자가 모델·관측 입력·평가 GT를 외부 저장소에서 받아 /data 에 마운트하면
# src/{api,local}_based/data/ 심볼릭 링크 또는 복사로 연결한다.
# 세부 경로 매핑은 docs/RESOURCES.md 를 따른다.

CMD ["python", "main.py", "--help"]
