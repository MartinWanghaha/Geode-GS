#!/bin/bash

# 简化版脚本，直接使用 bash 运行目标脚本，无需它们拥有执行权限

echo "正在查找并执行所有 'run_*.sh' 脚本..."
echo ""

count=0

for script in ./run_*.sh; do
    if [ -f "$script" ]; then
        echo "--- Executing: $script ---"
        # 直接使用 bash 来运行脚本
        bash "$script"
        echo "--- Finished: $script ---"
        echo ""
        count=$((count+1))
    fi
done

if [ "$count" -eq 0 ]; then
    echo "未找到任何匹配 'run_*.sh' 的脚本文件。"
else
    echo "任务完成，共执行了 $count 个脚本。"
fi