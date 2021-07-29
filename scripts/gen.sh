
CUR_DIR=$(dirname $(readlink -f $0))
ROOT_DIR=$CUR_DIR/..
PT_DIR=$ROOT_DIR/pytorch

function main()
{
    cd $ROOT_DIR
    # patch
    cp $ROOT_DIR/patch/npu.patch $PT_DIR
    cd $PT_DIR
    dos2unix docs/make.bat
    dos2unix scripts/appveyor/install.bat
    dos2unix scripts/appveyor/install_cuda.bat
    dos2unix scripts/build_windows.bat
    dos2unix scripts/proto.ps1
    dos2unix torch/distributions/von_mises.py
    dos2unix torch/nn/modules/transformer.pyi.in
    patch -p1 < npu.patch
    cp -r $ROOT_DIR/src/* $PT_DIR
}

main $@

