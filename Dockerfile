FROM jupyter/scipy-notebook:2023-05-01
LABEL maintainer="LAI base image for Jupyter notebooks"

# [Optional] If your requirements rarely change, uncomment this section to add them to the image.
#
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt 
# RUN jupyter labextension install jupyterlab-plotly

# [Optional] Uncomment this section to install additional packages.
#
# RUN apt-get update \
#     && export DEBIAN_FRONTEND=noninteractive \
#    && apt-get -y install --no-install-recommends <your-package-list-here>

# Use jupyterlab 
ENV JUPYTER_ENABLE_LAB=yes

# Here we will mount the data to be processed
# VOLUME /data

# By default VS does not start jupyter lab but only the needed kernels
# To start jupyterlab run start-notebook.sh from the vs terminal