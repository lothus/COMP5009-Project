FROM tensorflow/tensorflow:latest-gpu-jupyter
#To work on GPU
RUN apt-get install nvidia-modprobe
#Set language parameters for the container
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
#Changing directory
WORKDIR /code
#Get all the requirements for PIP
COPY Code/requirements.txt .
#Install the dependencies
RUN pip install -r requirements.txt
#update the path
ENV PATH=/root/.local:$PATH

RUN jupyter notebook --generate-config

ENTRYPOINT [ "jupyter","notebook", "--port=8888","--no-browser","--ip=0.0.0.0", "--notebook-dir=/code/", "--allow-root" ]
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> $HOME/.jupyter/jupyter_notebook_config.py
CMD []