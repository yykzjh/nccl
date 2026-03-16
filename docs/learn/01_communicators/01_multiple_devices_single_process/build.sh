cmake \
    -B build \
    -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DNCCL_HOME=/home/yiyin.zjh/nccl/build \
&& \
cmake \
    --build build \
    --config Release \
&& \
ln -sf build/compile_commands.json compile_commands.json
