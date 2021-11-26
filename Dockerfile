FROM tensorflow/tensorflow:latest-gpu
WORKDIR /app
COPY . .
RUN pip install --trusted-host pypi.python.org -r requirements.txt
CMD ["python", "cnn_patho.py"]

#docker run -v /Users/<path>:/app