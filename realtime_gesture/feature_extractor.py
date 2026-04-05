import numpy as np

class FeatureExtractor:
    def extract_features(self, hand_landmarks):
        # รับค่า hand_landmarks จาก MediaPipe (21 จุด) เเละ คืนค่าเป็น List ของพิกัด 63 ค่าที่ผ่านการ Normalize 
        # แตกข้อมูล x, y, z ออกมาเป็น List
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        zs = [lm.z for lm in hand_landmarks.landmark]

        # ทำ Relative Position (จุดที่ 0 เป็นจุดอ้างอิง)
        base_x, base_y = xs[0], ys[0]
        rel_xs = [x - base_x for x in xs]
        rel_ys = [y - base_y for y in ys]
        rel_zs = zs  # z  relative เพราะเป็นค่าความลึกจากกล้อง

        # Scale Normalization 
        # หาค่า Max Abs ของ XY และ Z เพื่อย่อสเกล
        max_xy = max(max([abs(x) for x in rel_xs]), max([abs(y) for y in rel_ys]))
        max_z = max([abs(z) for z in rel_zs])

        # ป้องกันการหารด้วย 0
        if max_xy > 0:
            rel_xs = [x / max_xy for x in rel_xs]
            rel_ys = [y / max_xy for y in rel_ys]
            
        if max_z > 0:
            rel_zs = [z / max_z for z in rel_zs]

        # รวมค่าเป็น Flat List (63 values)
        combined_features = []
        for x, y, z in zip(rel_xs, rel_ys, rel_zs):
            combined_features.extend([x, y, z])
        return combined_features