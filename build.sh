# Train
train=true
if [ "$train" = true ] ; then
    ./train.sh
fi

# Score
eval=true
if [ "$eval" = true ] ; then
    ./sample.sh && python score.py
fi

# Build
build=false
if [ "$build" = true ] ; then
    output=$(docker build .)
    sha256=$(docker image ls | awk 'NR==2{print $3}')
    echo $sha256
    docker save $sha256 | gzip > ./project/AIris_submit.tar.gz
    zip project.zip ./project/AIris_submit.tar.gz
fi


# Test
test=false
if [ "$test" = true ] ; then
    tar_file=./project/AIris_submit.tar.gz
    device=0
    sha256=`docker load --input $tar_file | grep -Po "sha256:(\w+)" | sed 's/sha256:\(.*\)/\1/g'`
    echo $sha256

    docker run -it --gpus "device=${device}" --rm -v /home/mio/project/diffusion-comp/code/eval_prompts_advance:/workspace/eval_prompts_advance -v /home/mio/project/diffusion-comp/code/train_data:/workspace/train_data -v /home/mio/project/diffusion-comp/code/models:/workspace/models -v /home/mio/.cache:/root/.cache -v /home/mio/.insightface:/root/.insightface -v /home/mio/project/diffusion-comp/code/indocker_shell.sh:/workspace/indocker_shell.sh $sha256
fi

