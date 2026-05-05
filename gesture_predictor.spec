# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

# ดึงไฟล์ Data ทั้งหมดของ mediapipe แบบอัตโนมัติ
mediapipe_datas = collect_data_files('mediapipe')

a = Analysis(
    ['realtime_gesture/gesture_predictor.py'],
    pathex=[],
    binaries=[],
    # รวมโฟลเดอร์ของเรา และไฟล์ของ mediapipe เข้าด้วยกัน
    datas=[
        ('models', 'models'), 
        ('config', 'config')
    ] + mediapipe_datas, 
    # บังคับหิ้วไลบรารีเบื้องหลังที่ชอบหลุดหาย
    hiddenimports=[
        'sklearn', 
        'sklearn.neural_network', 
        'sklearn.preprocessing',
        'sklearn.utils._typedefs',
        'sklearn.neighbors._partition_nodes',
        'mediapipe'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='gesture_predictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # เปิดเป็น True ไว้ก่อนเพื่อดู Error จอดำตอนเทสต์ใน Unity
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='gesture_predictor',
)