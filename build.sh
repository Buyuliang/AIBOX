#!/bin/bash

# 记录脚本开始时间
start_time=$(date +%s)

# 记录当前目录
current_dir=$(pwd)
echo "Current directory: $current_dir"

# 检查是否存在 build 目录，如果不存在则创建
if [ ! -d "$current_dir/build" ]; then
    echo "No build directory found. Creating build directory..."
    mkdir build
else
    echo "Build directory exists."
fi

# 进入 build 目录
cd build

# 执行 cmake .. 命令并检查是否成功
echo "Running cmake .."
if ! cmake ..; then
    echo "CMake failed. Exiting."
    exit 1
fi

# 设置默认线程数为 1
threads=1

# 解析命令行参数
for arg in "$@"; do
    case $arg in
        -j)
            threads="$2"
            shift # 跳过值
            shift # 跳过参数
            ;;
        -j*)
            threads="${arg:2}" # 从参数中提取线程数
            shift
            ;;
    esac
done

# 如果线程数不是数字，则设置为 1
if ! [[ "$threads" =~ ^[0-9]+$ ]]; then
    echo "Invalid thread count provided. Using default of 1 thread."
    threads=1
fi

# 执行 make 命令并检查是否成功
echo "Running make with $threads threads"
if ! make -j "$threads"; then
    echo "Make failed. Exiting."
    exit 1
fi

# 执行 make install 命令并检查是否成功
echo "Running make install"
if make install; then
    echo "Build and installation completed successfully."
else
    echo "Make install failed. Exiting."
    exit 1
fi

# 记录脚本结束时间并计算总执行时间
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Total script execution time: $execution_time seconds."
