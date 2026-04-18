#!/usr/bin/env sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/venv}"

print_help() {
    cat <<EOF
Usage:
  ./setup.sh
  . ./setup.sh

What it does:
  1. Creates ./venv if it does not exist
  2. Activates the virtual environment for this shell
  3. Upgrades pip tooling
  4. Installs the package in editable mode with all optional extras (pip install -e ".[all]")

Notes:
  - If you run './setup.sh', the virtualenv is created and populated, but the
    activation only applies inside the script. Activate afterwards with:
      . venv/bin/activate
  - If you source '. ./setup.sh', the virtualenv stays active in your current shell.

Optional environment variables:
  PYTHON_BIN   Python executable to use for creating the virtualenv
  VENV_DIR     Virtualenv path (default: ./venv)
EOF
}

case "${1:-}" in
    -h|--help)
        print_help
        exit 0
        ;;
esac

choose_python() {
    if [ -n "${PYTHON_BIN:-}" ]; then
        printf '%s\n' "$PYTHON_BIN"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        printf '%s\n' "python3"
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        printf '%s\n' "python"
        return 0
    fi
    return 1
}

is_sourced() {
    case "${ZSH_EVAL_CONTEXT:-}" in
        *:file) return 0 ;;
    esac
    if [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE:-$0}" != "$0" ]; then
        return 0
    fi
    return 1
}

PYTHON_CMD=$(choose_python) || {
    echo "No Python interpreter found. Install python3 and rerun setup." >&2
    exit 1
}

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
else
    echo "Using existing virtualenv at $VENV_DIR"
fi

# shellcheck disable=SC1090
. "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "$SCRIPT_DIR[all]"

echo
echo "Virtualenv is ready."
echo "Python: $(python -c 'import sys; print(sys.executable)')"

if is_sourced; then
    echo "The virtualenv is active in your current shell."
else
    echo "Activate it in your shell with:"
    echo "  . \"$VENV_DIR/bin/activate\""
fi
