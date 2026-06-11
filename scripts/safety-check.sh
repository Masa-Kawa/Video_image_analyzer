#!/usr/bin/env bash
#
# push 前の安全検査: 危険なファイル/データの混入を検出する。
#
#   bash scripts/safety-check.sh            # 未pushのコミットを検査
#   bash scripts/safety-check.sh main..HEAD # 範囲を指定して検査
#
# pre-push フックから自動で呼ばれるほか、PR を出す前に手動でも実行できる。
# BLOCK が1件でもあれば終了コード1（push中断）。WARN は表示のみ。
# どうしても通す場合は: git push --no-verify
#
set -uo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

# 検査対象コミット範囲（rev-list へそのまま渡す。既定=どのリモートにも無い未pushコミット）
if [ "$#" -eq 0 ]; then
  set -- HEAD --not --remotes
fi

mapfile -t commits < <(git rev-list "$@" 2>/dev/null)
if [ "${#commits[@]}" -eq 0 ]; then
  echo "✓ 安全検査: push対象の新規コミットはありません"
  exit 0
fi

# 対象コミットが追加/変更したファイル一覧
mapfile -t files < <(git show --pretty="" --name-only --diff-filter=ACM "${commits[@]}" 2>/dev/null | sort -u)

block=0
warn=0
LARGE_MB=5
LARGE_BYTES=$((LARGE_MB * 1024 * 1024))

say_block() { echo "  ⛔ BLOCK: $1"; block=$((block + 1)); }
say_warn()  { echo "  ⚠ WARN : $1"; warn=$((warn + 1)); }

echo "🔎 安全検査: ${#commits[@]} コミット / ${#files[@]} ファイルを点検…"
echo ""

# 1) 追跡されているが .gitignore 対象のファイル（本来コミットされないはずのもの）
mapfile -t tracked_ignored < <(git ls-files -i -c --exclude-standard 2>/dev/null)
if [ "${#tracked_ignored[@]}" -gt 0 ]; then
  echo "[gitignore対象なのに追跡中]"
  for f in "${tracked_ignored[@]}"; do
    say_block "$f （.gitignore対象。git rm --cached で外す）"
  done
  echo ""
fi

# 2) 各ファイルの種類・サイズ・中身を点検
for f in "${files[@]}"; do
  [ -f "$f" ] || continue   # 範囲内で後に削除されたものは除外

  base="$(basename "$f")"
  lower="${base,,}"

  # 2a) 危険な拡張子（実データ・大容量バイナリ。GitHubに載せない）
  case "$lower" in
    *.mp4|*.avi|*.mov|*.mkv|*.webm|*.m4v)
      say_block "$f （動画ファイル。実データはコミットしない）" ;;
    *.pth|*.pt|*.onnx|*.npy|*.h5|*.ckpt|*.safetensors|*.bin)
      say_block "$f （モデル/重みファイル。再配布しない）" ;;
    *.csv|*.srt|*.jsonl)
      say_block "$f （解析データ/出力。患者情報を含む恐れ。本来gitignore対象）" ;;
    *.zip|*.tar|*.gz|*.tgz|*.7z|*.rar)
      say_block "$f （アーカイブ。中身が不明なため送らない）" ;;
    *.pem|*.key|*.p12|*.pfx|*.keystore|*.jks)
      say_block "$f （鍵/証明書ファイル）" ;;
    *.env|.env|.env.*)
      say_block "$f （環境変数ファイル。秘密情報の恐れ）" ;;
  esac

  # 2b) 危険なファイル名
  case "$lower" in
    .env|.env.*|id_rsa|id_dsa|id_ecdsa|id_ed25519|*secret*|*credential*|*password*|*.secret)
      say_block "$f （秘密情報を示す名前）" ;;
  esac

  # 2c) サイズ超過
  size=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$size" -gt "$LARGE_BYTES" ]; then
    mb=$((size / 1024 / 1024))
    say_block "$f （${mb}MB > ${LARGE_MB}MB。大容量ファイルは送らない）"
  fi

  # 2d) 中身の高確度シークレット（テキストのみ。-I でバイナリは除外）
  #     秘密鍵・各種APIトークンは誤検知が少ないため BLOCK 扱い。
  if grep -IlqE -- '-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36,}|AIza[0-9A-Za-z_-]{35}|xox[baprs]-[A-Za-z0-9-]{10,}|sk-[A-Za-z0-9]{20,}' "$f" 2>/dev/null; then
    say_block "$f （APIキー/秘密鍵らしき文字列を検出）"
  fi

  # 2e) 中身の低確度シグナル（誤検知あり。WARN）
  if grep -IqE -- '/home/[a-z_][a-z0-9_-]*/' "$f" 2>/dev/null; then
    say_warn "$f （個人の絶対パス /home/<user>/ が含まれる）"
  fi
  if grep -IqiE -- '(password|passwd|secret|api[_-]?key|token)[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"']{6,}' "$f" 2>/dev/null; then
    say_warn "$f （パスワード/キーらしき代入。確認推奨）"
  fi
done

echo ""
if [ "$block" -gt 0 ]; then
  echo "✗ 安全検査: BLOCK ${block}件 / WARN ${warn}件 — push を中止します。"
  echo "  上記を取り除いてから再実行してください（誤検知なら git push --no-verify で回避可能）。"
  exit 1
fi

if [ "$warn" -gt 0 ]; then
  echo "△ 安全検査: WARN ${warn}件（push は許可）。内容に問題ないか確認してください。"
else
  echo "✓ 安全検査: 問題は見つかりませんでした。"
fi
exit 0
