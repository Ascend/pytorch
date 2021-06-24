
CUR_DIR=$(dirname $(readlink -f $0))
ROOT_DIR=$CUR_DIR/..
PT_DIR=$ROOT_DIR/pytorch
PT_PKG=pytorch-v1.5.0.tar.gz

function main()
{
    cd $ROOT_DIR
    # get ori pytorch
    if [ -f $ROOT_DIR/$PT_PKG ];then
        echo "detect $PT_PKG exist, skip download"
    else
        wget https://ascend-ptadapter.obs.cn-north-4.myhuaweicloud.com/pytorch-v1.5.0/$PT_PKG --no-check-certificate
    fi

    if [ $? != 0 ]; then
        echo "Failed to wget source code of pytorch, check network."
        exit 1
    fi

    # mkdir pytorch
    if [ -d $PT_DIR ];then
        echo "$PT_DIR exists, if nothing to backup, please remove it"
        exit 1
    fi

    mkdir $PT_DIR

    # unpack
    tar -xf $PT_PKG

    echo "download and unpack $PT_PKG success"

    # patch
    cp $ROOT_DIR/patch/npu.patch $PT_DIR
    cd $PT_DIR
    patch -p1 < npu.patch
    cp -r $ROOT_DIR/src/* $PT_DIR
    
    if [[ $1 = "gen" ]];then
        exit 0
    fi
    # build
    bash build.sh
    cp $PT_DIR/dist/torch-* $ROOT_DIR/dist
}

main $@

