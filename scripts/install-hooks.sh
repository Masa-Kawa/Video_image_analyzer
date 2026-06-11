#!/usr/bin/env bash
#
# Git フックを有効化する。ネイティブフックはリポジトリに自動共有されないため、
# クローン後に各自が一度だけ実行する。
#
#   bash scripts/install-hooks.sh
#
set -euo pipefail

root="$(git rev-parse --show-toplevel)"

install_hook() {
  local name="$1"
  local src="$root/scripts/git-hooks/$name"
  local dst="$root/.git/hooks/$name"
  chmod +x "$src"
  # .git/hooks/<name> から見た相対パスで管理下スクリプトへシンボリックリンク
  ln -sf "../../scripts/git-hooks/$name" "$dst"
  echo "✓ $name フックを有効化: $dst -> scripts/git-hooks/$name"
}

chmod +x "$root/scripts/safety-check.sh"
install_hook pre-commit
install_hook pre-push

echo ""
echo "  pre-commit : ステージした .py に ruff lint（回避: git commit --no-verify）"
echo "  pre-push   : 送信前に危険ファイル/データを検査（回避: git push --no-verify）"
