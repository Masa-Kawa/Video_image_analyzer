#!/usr/bin/env bash
#
# Git フックを有効化する。ネイティブフックはリポジトリに自動共有されないため、
# クローン後に各自が一度だけ実行する。
#
#   bash scripts/install-hooks.sh
#
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
src="$root/scripts/git-hooks/pre-commit"
dst="$root/.git/hooks/pre-commit"

chmod +x "$src"
# .git/hooks/pre-commit から見た相対パスで管理下スクリプトへシンボリックリンク
ln -sf ../../scripts/git-hooks/pre-commit "$dst"

echo "✓ pre-commit フックを有効化しました: $dst -> scripts/git-hooks/pre-commit"
echo "  （ステージした .py に ruff lint を実行します。回避は git commit --no-verify）"
