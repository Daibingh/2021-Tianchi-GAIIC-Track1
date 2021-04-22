sudo docker container ls --all | grep "registry.cn-shenzhen.aliyuncs.com/sduhdb/mednlp" | awk '{print $1 }' | xargs sudo docker container rm
