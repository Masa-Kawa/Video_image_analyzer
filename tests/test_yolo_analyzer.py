"""
yolo_analyzer のクラスIDマッピングとフェーズ推定のユニットテスト

ultralytics YOLOv8 は標準 COCO80（0始まり）でクラスIDを返すため、
SURGICAL_INSTRUMENTS が標準 COCO に整合していること、および knife/scissors の
ID が infer_phase で正しく機能することを検証する。
"""

import unittest

from src.yolo.yolo_analyzer import (
    SURGICAL_INSTRUMENTS,
    INSTRUMENT_COCONames,
    infer_phase,
)


class TestClassIdMapping(unittest.TestCase):
    """標準 COCO80 インデックスとの整合性。"""

    def test_knife_and_scissors_ids(self):
        # 標準 COCO80: knife=43, scissors=76
        self.assertEqual(SURGICAL_INSTRUMENTS[43], "knife")
        self.assertEqual(SURGICAL_INSTRUMENTS[76], "scissors")

    def test_surrounding_ids_aligned(self):
        self.assertEqual(SURGICAL_INSTRUMENTS[42], "fork")
        self.assertEqual(SURGICAL_INSTRUMENTS[44], "spoon")
        self.assertEqual(SURGICAL_INSTRUMENTS[75], "vase")
        self.assertEqual(SURGICAL_INSTRUMENTS[0], "person")

    def test_no_out_of_range_id(self):
        # COCO80 の最大IDは 79（旧コードの 80 は off-by-one の症状）
        self.assertEqual(max(SURGICAL_INSTRUMENTS), 79)
        self.assertNotIn(80, SURGICAL_INSTRUMENTS)

    def test_cocoNames_consistent_with_main_map(self):
        # 補助マップが主マップと矛盾しないこと
        for cid, name in INSTRUMENT_COCONames.items():
            self.assertEqual(SURGICAL_INSTRUMENTS[cid], name)


class TestInferPhase(unittest.TestCase):
    """knife/scissors の名称が正しく解決された前提でのフェーズ推定。"""

    def test_knife_to_dissection(self):
        # SURGICAL_INSTRUMENTS[43] == "knife" なので dissection になる
        phase, primary = infer_phase([SURGICAL_INSTRUMENTS[43], "person"])
        self.assertEqual(phase, "dissection")
        self.assertIn("knife", primary)

    def test_scissors_to_cutting(self):
        phase, primary = infer_phase([SURGICAL_INSTRUMENTS[76]])
        self.assertEqual(phase, "cutting")
        self.assertIn("scissors", primary)

    def test_person_only_manipulation(self):
        phase, primary = infer_phase(["person"])
        self.assertEqual(phase, "manipulation")

    def test_empty_neutral(self):
        phase, primary = infer_phase([])
        self.assertEqual(phase, "neutral")


if __name__ == "__main__":
    unittest.main()
