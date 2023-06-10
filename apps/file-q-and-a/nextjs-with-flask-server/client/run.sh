#!/bin/bash
export NODE_ENV=development

function dev(){
    echo "start..."
    nohup pnpm dev > output_nodejs.log 2>&1 &
    echo "start successful"
    return 0
}

# 添加启动命令
function start(){
    echo "start..."
    pnpm build
    nohup pnpm start > output_nodejs.log 2>&1 &
    echo "start successful"
    return 0
}

# 添加停止命令
function stop(){
    echo "stop..."
    lsof -i:3000 |awk '{print "kill -9 " $2}'|sh
    echo "stop successful"
    return 0
}
case $1 in
"dev")
    dev
    ;;
"start")
    start
    ;;
"stop")
    stop
    ;;
"restart")
    stop && start
    ;;
*)
    echo "请输入: start, stop, restart"
    ;;
esac