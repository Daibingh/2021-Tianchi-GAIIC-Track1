#!/bin/bash
if [ -z $1 ]
then
   echo "s1:version"
   exit
fi

# if [ -z $DOCKER_REGISTRY ]
# then
#    echo "not find $DOCKER_REGISTRY"
#    exit
# fi

VERSION=$1

##1.create docker regsit
sudo docker build -t registry.cn-shenzhen.aliyuncs.com/sduhdb/mednlp:$VERSION .
