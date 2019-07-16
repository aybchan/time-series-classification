FROM jupyter/datascience-notebook 

COPY jupyter_notebook_config.py $HOME/.jupyter/jupyter_notebook_config.py

LABEL description="Jupyterlab"

USER root
RUN apt-get update
RUN apt install -y libmysqlclient-dev
USER jovyan

ADD requirements.txt .
RUN pip install -U -r requirements.txt

WORKDIR $HOME/project/
