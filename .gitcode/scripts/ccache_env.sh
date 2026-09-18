echo "===================== ccache 环境变量 ====================="
export CCACHE_DIR="/LOCAL/PTA_CCACHE"
export CCACHE_REMOTE_STORAGE=file:///PTA_CCACHE
export CCACHE_REMOTE_ONLY=true

export CCACHE_NOHASHDIR=1
# export CCACHE_COMPILERCHECK=content
export CCACHE_SLOPPINESS=include_file_ctime,include_file_mtime,time_macros
export CCACHE_NOINODECACHE=1

export CCACHE_IGNORECONFIG=1
export CCACHE_DIRECT_MODE=yes
export CCACHE_COMPRESS=1
export CCACHE_COMPRESSLEVEL=0
# export CCACHE_COMPRESSLEVEL=3

# 临时解决codegen生成代码内容随机问题，26.1分支拉出后去掉
# export PYTHONHASHSEED=0
# 临时解决submodule并行fetch问题
git config --global submodule.fetchJobs 16
export NINJA_STATUS="[%r:%s/%t %es] "

ccache -z -p -V

# 清理缓存，2%概率执行。限制200G，一般不用这么大，但因为PTA共用头文件的问题太严重，cache增长的太快，希望一个PR隔2天再跑cache还是能保留住。
CCACHE_VERSION=$(ccache --version | head -n1 | awk '{print $3}')
if [ "$(printf '%s\n' "$CCACHE_VERSION" "4.4" | sort -V | head -n1)" = "4.4" ]; then
    if [ $((RANDOM % 100)) -lt 2 ]; then
        nice -n 19 ionice -c 2 -n 7 ccache --trim-dir /PTA_CCACHE --trim-max-size 200G &
    fi
fi