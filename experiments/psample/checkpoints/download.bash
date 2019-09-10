#!/bin/bash
CUR_BASH_DIR_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

SERVER_PREFIX="zhen@a18.millennium.berkeley.edu:video-acc/"

if [ "$#" -eq 0 ];
then
    echo "download all checkpoints..."
    # if the source server delete something, local side will delete too
    rsync -avz --exclude "*.bash" \
        "${SERVER_PREFIX}/pretrained_models/psample/" \
        ./ \
        --delete
else
    for path in "$@"
    do
        # make dir
        mkdir -p "${CUR_BASH_DIR_PATH}/${path}"
        # use rsync to download, the trailing `/` is necessary to avoid
        # duplicating the folder
        echo "download ${path}..."
        rsync -avz --exclude "*.bash" \
            "${SERVER_PREFIX}/pretrained_models/psample/${path}/" \
            ${path} \
            --delete
    done
fi