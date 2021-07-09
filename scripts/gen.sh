
CUR_DIR=$(dirname $(readlink -f $0))
ROOT_DIR=$CUR_DIR/..
PT_DIR=$ROOT_DIR/pytorch

function main()
{
    cd $ROOT_DIR
    # patch
    cp $ROOT_DIR/patch/npu.patch $PT_DIR
    cd $PT_DIR
    patch -p1 < npu.patch
    cp -r $ROOT_DIR/src/* $PT_DIR
}

main $@

