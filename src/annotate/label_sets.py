"""
ラベルレジストリ（術式ごとのフェーズ語彙）。

フェーズ名は術式によって異なるため、包括的な単一リストではなく
「術式 → 順序付きフェーズ配列」のレジストリ方式を採る。
新しい術式は LABEL_SETS にエントリを追加するだけで対応できる
（サーバ/フロントのコード変更は不要）。

MVPは胆嚢摘出術（cholecystectomy）。既存の Cholec80 フェーズ定義
（src.cholec_phase.CHOLEC80_PHASES）を再利用し、語彙の二重定義を避ける。
"""

from typing import Dict, List

from src.cholec_phase import CHOLEC80_PHASES

# 術式名 → 順序付きフェーズ名リスト
# キーは API / CLI の --procedure に渡す識別子。
LABEL_SETS: Dict[str, List[str]] = {
    "cholecystectomy": list(CHOLEC80_PHASES),
}

DEFAULT_PROCEDURE = "cholecystectomy"


def get_label_set(procedure: str) -> List[str]:
    """
    指定術式のフェーズ語彙（順序付き）を返す。

    Args:
        procedure: 術式識別子（例: "cholecystectomy"）

    Returns:
        フェーズ名のリスト

    Raises:
        KeyError: 未登録の術式が指定された場合
    """
    if procedure not in LABEL_SETS:
        available = ", ".join(sorted(LABEL_SETS))
        raise KeyError(
            f"未登録の術式: {procedure!r}（利用可能: {available}）"
        )
    return list(LABEL_SETS[procedure])


def available_procedures() -> List[str]:
    """登録済みの術式識別子の一覧を返す。"""
    return sorted(LABEL_SETS)
