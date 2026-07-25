#!/usr/bin/env bash

set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$PROJECT_DIR/frontend"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-tec_stack}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
BACKEND_PID=""
FRONTEND_PID=""

usage() {
  cat <<'EOF'
用法：
  ./start.sh          启动 FastAPI 和 React 开发服务器
  ./start.sh --check  只检查运行环境，不启动服务
  ./start.sh --help   显示帮助

可选环境变量：
  CONDA_ENV_NAME  Conda 环境名，默认 tec_stack
  BACKEND_HOST    后端监听地址，默认 127.0.0.1
  BACKEND_PORT    后端端口，默认 8000
  FRONTEND_HOST   前端监听地址，默认 127.0.0.1
  FRONTEND_PORT   前端端口，默认 5173
EOF
}

fail() {
  printf '启动失败：%s\n' "$1" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

port_is_busy() {
  local port="$1"
  command_exists lsof && lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

resolve_python() {
  local python_path=""

  if [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV_NAME" ] && command_exists python; then
    python_path="$(command -v python)"
  elif command_exists conda; then
    python_path="$(conda run -n "$CONDA_ENV_NAME" python -c 'import sys; print(sys.executable)' 2>/dev/null | tail -n 1)"
  fi

  [ -n "$python_path" ] || fail "找不到 Conda 环境 '$CONDA_ENV_NAME'。请先创建环境或设置 CONDA_ENV_NAME。"
  [ -x "$python_path" ] || fail "Python 不可执行：$python_path"
  "$python_path" -c 'import uvicorn' >/dev/null 2>&1 || fail "环境 '$CONDA_ENV_NAME' 中未安装 uvicorn。"

  PYTHON_BIN="$python_path"
}

prepare_frontend() {
  command_exists npm || fail "找不到 npm，请先安装 Node.js。"
  [ -f "$FRONTEND_DIR/package.json" ] || fail "找不到 frontend/package.json。"

  if [ ! -x "$FRONTEND_DIR/node_modules/.bin/vite" ]; then
    printf '首次启动，正在安装前端依赖……\n'
    (cd "$FRONTEND_DIR" && npm install) || fail "前端依赖安装失败。"
  fi
}

cleanup() {
  trap - EXIT INT TERM
  printf '\n正在关闭服务……\n'

  if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi

  [ -z "$FRONTEND_PID" ] || wait "$FRONTEND_PID" >/dev/null 2>&1 || true
  [ -z "$BACKEND_PID" ] || wait "$BACKEND_PID" >/dev/null 2>&1 || true
  printf '服务已关闭。\n'
}

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
  --check|"")
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

resolve_python
prepare_frontend

printf '环境检查通过：\n'
printf '  Python: %s\n' "$PYTHON_BIN"
printf '  Backend: http://%s:%s\n' "$BACKEND_HOST" "$BACKEND_PORT"
printf '  Frontend: http://%s:%s\n' "$FRONTEND_HOST" "$FRONTEND_PORT"

if [ "${1:-}" = "--check" ]; then
  exit 0
fi

port_is_busy "$BACKEND_PORT" && fail "端口 $BACKEND_PORT 已被占用。"
port_is_busy "$FRONTEND_PORT" && fail "端口 $FRONTEND_PORT 已被占用。"

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$PROJECT_DIR" || fail "无法进入项目目录。"
"$PYTHON_BIN" -m uvicorn app.main:app \
  --host "$BACKEND_HOST" \
  --port "$BACKEND_PORT" \
  --reload &
BACKEND_PID=$!

(
  cd "$FRONTEND_DIR" || exit 1
  export VITE_BACKEND_TARGET="http://$BACKEND_HOST:$BACKEND_PORT"
  exec ./node_modules/.bin/vite \
    --host "$FRONTEND_HOST" \
    --port "$FRONTEND_PORT" \
    --strictPort
) &
FRONTEND_PID=$!

printf '\n服务正在启动，按 Ctrl+C 可同时关闭前后端。\n'
printf '打开浏览器访问：http://%s:%s\n\n' "$FRONTEND_HOST" "$FRONTEND_PORT"

while kill -0 "$BACKEND_PID" >/dev/null 2>&1 && kill -0 "$FRONTEND_PID" >/dev/null 2>&1; do
  sleep 1
done

if ! kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
  printf 'FastAPI 进程已退出，请查看上方日志。\n' >&2
else
  printf 'Vite 进程已退出，请查看上方日志。\n' >&2
fi

exit 1
