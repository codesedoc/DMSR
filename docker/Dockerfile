# syntax = docker/dockerfile:1
FROM nlpx_env:0.3
WORKDIR /explore

ARG USER_NAME
RUN  if ! grep ${USER_NAME} /etc/passwd; then echo "Invalid user name ${USER_NAME}!" && exit 1;  fi

# DEBIAN_FRONTEND=noninteractive
USER $USER_NAME
COPY . .
USER root
RUN chown --recursive ${USER_NAME}:${USER_NAME} \
                /home/${USER_NAME}  \
                /explore
USER $USER_NAME

CMD [ "bash", "sh/docker_run.sh" ]

