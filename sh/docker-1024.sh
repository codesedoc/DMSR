#!/bin/bash

set -e

function setup_bind_volume_host_path() {
  local user_name=$DOCKER_USER_NAME
  local group_name=$DOCKER_USER_NAME
#  cd $(sudo -u "$user_name" bash -c 'echo $HOME')
#  if [ $(basename $(pwd)) != "$user_name" ]; then
#    echo "Current Path is $(pwd) if not at the home of $user_name!"
#    return 0
#  fi
  sudo -u "$user_name" bash -c 'mkdir -p ~/docker/bind/nlpx/tmp'
  sudo -u "$user_name" bash -c 'mkdir -p ~/docker/bind/nlpx/cache'
  sudo -u "$user_name" bash -c 'mkdir -p ~/docker/bind/nlpx/output'
  sudo -u "$user_name" bash -c 'mkdir -p ~/docker/bind/nlpx/storage'
  sudo -u "$user_name" bash -c 'mkdir -p ~/.cache'
  sudo -u "$user_name" bash -c 'mkdir -p ~/ray_results'

#  sudo chown -R "$user_name:$group_name" ~$user_name/docker
#  sudo chown -R "$user_name:$group_name" ~$user_name/.cache
#  sudo chown -R "$user_name:$group_name" ~$user_name/ray_results

  echo "Finish Setup Path for [ ${user_name} ]"
}

function verify_id(){
  local id_number=$1
  local regex='^[0-9]+$'
  if ! [[ $id_number =~ $regex ]] ; then
    echo "error: Your id ($id_number) Not a number" >&2
    return 1
  fi
  if [ $id_number -lt 2000 ] || [ $id_number -gt 3000 ]; then
    echo "error: Your id ($id_number) is illegal, it should be in the range of [2000, 3000)"
    return 1
  fi
}

function set_docker_user() {
  if id -nu $DOCKER_USER_ID; then
    if [ "$DOCKER_USER_NAME" == "$(id -nu $DOCKER_USER_ID)" ]; then
      echo "User ($DOCKER_USER_NAME) has already been setup!"
    else
      echo "User id is already used by user $(id -nu $DOCKER_USER_ID), but your user name is ($DOCKER_USER_NAME)"
      return 1
    fi
  else
    sudo adduser --gecos --quiet $DOCKER_USER_NAME --uid $DOCKER_USER_ID
    sudo chfn -f "$FULL_NAME" -r "$ROOM_NUMBER" -h "$HOME_PHONE" -w "$WORK_PHONE"  $DOCKER_USER_NAME
  fi

  if id -ng $DOCKER_GROUP_ID; then
    if [ "$DOCKER_USER_NAME" == "$(id -nu $DOCKER_GROUP_ID)" ]; then
      echo "Group ($DOCKER_USER_NAME) has already been setup!"
    else
      echo "Group id is already used by group $(id -nu $DOCKER_GROUP_ID), but your group name is ($DOCKER_USER_NAME)"
      return 1
    fi
  else
    sudo addgroup --quiet $DOCKER_USER_NAME --gid $DOCKER_GROUP_ID
  fi

  if ! groups $DOCKER_USER_NAME | grep -q $DOCKER_USER_NAME; then
    echo "Add user ($DOCKER_USER_NAME) to group ($DOCKER_USER_NAME)"
    sudo adduser --quiet $DOCKER_USER_NAME $DOCKER_USER_NAME
  fi
}

function main() {
  set_variables && \
  set_docker_user && \
  setup_bind_volume_host_path
}

function set_variables() {
  if [ -z "$DOCKER_USER_ID" ]; then DOCKER_USER_ID=2001; fi
  if [ -z "$DOCKER_GROUP_ID" ]; then DOCKER_GROUP_ID=2001; fi
  if [ -z "$DOCKER_USER_NAME" ]; then DOCKER_USER_NAME='docker-1024'; fi
  verify_id $DOCKER_USER_ID
  verify_id $DOCKER_GROUP_ID

  FULL_NAME="NLPx"
  ROOM_NUMBER=""
  HOME_PHONE=""
  WORK_PHONE=""

}
main