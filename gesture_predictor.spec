Baabae
baabae1tap
ไม่ระบุ

นี่คือจุดเริ่มต้นของช่อง #debug 
Baabae — 5/5/2569 21:42
private void ReceiveData()
{
    client = new UdpClient(port);
    // เพิ่มบรรทัดนี้: ขยาย Buffer ให้รับข้อมูลขนาดใหญ่ได้ (เช่น 1MB)
    client.Client.ReceiveBufferSize = 1024 * 1024; 

    while (true)
    {
        // ... โค้ดเดิมของคุณ ...
    }
}
// ใน GestureReceiver.cs ตรง ReceiveData
byte[] data = client.Receive(ref anyIP);
Debug.Log("ได้รับข้อมูลขนาด: " + data.Length + " bytes"); // เช็กขนาด Packet

string jsonString = Encoding.UTF8.GetString(data).Trim();
DataPacket decoded = JsonUtility.FromJson<DataPacket>(jsonString);

if (decoded == null) {
    Debug.LogError("JSON Decode พัง! ลองเช็ก Format หรือขนาดข้อมูล");
}
eartttpy 1 — 5/5/2569 21:48

Baabae — 5/5/2569 21:49

eartttpy 1 — 5/5/2569 22:00

Baabae — 5/5/2569 22:02
import sklearn.neural_network
eartttpy 1 — 5/5/2569 22:06
ภาพ
ภาพ
✧ ─=≡Σ((( つ•̀ω•́)つ — 5/5/2569 22:49
pyinstaller gesture_predictor.spec -y
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

# ดึงไฟล์ Data ทั้งหมดของ mediapipe แบบอัตโนมัติ
mediapipe_datas = collect_data_files('mediapipe')

gesture_predictor.spec
2 KB
ภาพ
eartttpy — 5/5/2569 23:08
UDP Receive Error: LoadImage can only be called from the main thread.
Constructors and field initializers will be executed from the loading thread when loading a scene.
Don't use this function in the constructor or field initializers, instead move initialization code to the Awake or Start function.
UnityEngine.Debug:LogError (object)
GestureReceiver:ReceiveData () (at Assets/main/code/GestureReceiver.cs:99)
System.Threading.ThreadHelper:ThreadStart ()
Baabae — 6/5/2569 18:57
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

gesture_recieve.cs
4 KB
Baabae — 6/5/2569 19:58
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

gesture.cs
5 KB
✧ ─=≡Σ((( つ•̀ω•́)つ — 6/5/2569 20:04
pyinstaller gesture_predictor.spec -y
Baabae — 6/5/2569 20:15
อันนี้
﻿
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