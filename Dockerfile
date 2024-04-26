FROM python:3.7
ENV PYTHONBUFFERED 1
ADD . /app 
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt 
RUN pip install -r requirements.txt
CMD python app.py
FROM python:3.7
ENV PYTHONBUFFERED 1
ADD . /app
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
 RUN pip install -r requirements.txt
CMD ["streamlit","run","stream.py",'flask','--host=0.0.0.0']
