############################################################
# Dockerfile exposing a statistical model as a 
# production ready REST Service
############################################################

# Set the base image
FROM debian:bullseye-slim

# File Author / Maintainer
LABEL image.author="daniela.kolarova@gmail.com"

RUN apt-get update && apt-get install -y apache2 \
    libapache2-mod-wsgi-py3 \
    python3 \
    python3-pip \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# Copy over and install the requirements
COPY ./app/requirements.txt /var/www/varmax-as-a-service/app/requirements.txt
RUN pip install -r /var/www/varmax-as-a-service/app/requirements.txt

# Copy over the apache configuration file and enable the site
COPY ./apache-flask.conf /etc/apache2/sites-available/apache-flask.conf
# Copy over the wsgi file, run.py, the app and the saved models
COPY ./ /var/www/varmax-as-a-service/

RUN a2dissite 000-default.conf
RUN a2ensite apache-flask.conf
RUN a2enmod headers

# LINK apache config to docker logs.
RUN ln -sf /proc/self/fd/1 /var/log/apache2/access.log && \
    ln -sf /proc/self/fd/1 /var/log/apache2/error.log

EXPOSE 80

WORKDIR /var/www/varmax-as-a-service

CMD  /usr/sbin/apache2ctl -D FOREGROUND
