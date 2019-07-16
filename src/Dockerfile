FROM jupyter/datascience-notebook 

COPY jupyter_notebook_config.py $HOME/.jupyter/jupyter_notebook_config.py

LABEL description="Jupyterlab"

USER root
USER jovyan

ADD requirements.txt .
RUN pip install -U -r requirements.txt

WORKDIR $HOME/project/
