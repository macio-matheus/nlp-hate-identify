FROM python:3.6.4

RUN mkdir /app
WORKDIR /app

# INSTALL DEPENDENCIES
COPY requirements.py /app
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.py

EXPOSE 6543
