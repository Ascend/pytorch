OFFICIAL_PYTORCH=$1
PATCH_PYTORCH=$2
OUT_PATCH_FILE=$3

diff -Nur --exclude=.git* --exclude=OWNERS \
    --exclude=access_control_test.py \
    --exclude=build.sh --exclude=third_party \
    --exclude=README* -Nur ${OFFICIAL_PYTORCH}  ${PATCH_PYTORCH} > ${OUT_PATCH_FILE}