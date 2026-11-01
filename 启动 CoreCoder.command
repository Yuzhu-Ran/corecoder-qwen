#!/bin/zsh

CORECODER_PROJECT_DIR="${0:A:h}"
CORECODER_PYTHON="$CORECODER_PROJECT_DIR/.venv/bin/python"

corecoder_launch_error() {
    print -r -- "$1"
    read -r "CORECODER_LAUNCH_REPLY?按回车结束..."
    exit 1
}

[[ -d "$CORECODER_PROJECT_DIR" ]] || corecoder_launch_error "找不到项目目录：$CORECODER_PROJECT_DIR"
[[ -x "$CORECODER_PYTHON" ]] || corecoder_launch_error "找不到项目的 Python 环境，请先完成依赖安装。"
[[ -f "$CORECODER_PROJECT_DIR/.env" ]] || corecoder_launch_error "找不到 .env，请先配置模型和 API Key。"

cd "$CORECODER_PROJECT_DIR" || corecoder_launch_error "无法进入项目目录。"
export PATH="$CORECODER_PROJECT_DIR/.venv/bin:$PATH"

print -r -- "正在启动 CoreCoder..."
print -r -- "项目：$CORECODER_PROJECT_DIR"
print -r -- "输入 quit 退出；Ctrl+C 取消当前任务。"
print

"$CORECODER_PYTHON" -m corecoder
CORECODER_LAUNCH_EXIT=$?

if (( CORECODER_LAUNCH_EXIT != 0 )); then
    print
    print -r -- "启动或运行遇到问题，请查看上方提示。"
    read -r "CORECODER_LAUNCH_REPLY?按回车结束..."
fi

exit "$CORECODER_LAUNCH_EXIT"
