FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /solution
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
ENTRYPOINT ["python"]
