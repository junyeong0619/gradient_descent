# PyInstaller spec for GD_Simulator (Windows)
#
# 빌드 방법 (Windows PowerShell or CMD):
#   1) python -m venv .venv  &&  .venv\Scripts\activate
#   2) pip install numpy matplotlib pyinstaller
#   3) pyinstaller --clean build_windows.spec
#   → dist\GD_Simulator.exe 생성
#
# 산출물 한 개 파일(.exe)만 배포하려면 --onefile 동작이 기본이고,
# 빠른 실행을 원하면 EXE(...) 의 console/onedir 옵션을 조정하면 됨.

block_cipher = None

a = Analysis(
    ['simulation.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # matplotlib 의 GUI 백엔드 — 명시적으로 포함시키지 않으면 누락될 수 있음
        'matplotlib.backends.backend_tkagg',
        # 3D 플로팅
        'mpl_toolkits.mplot3d',
        # 파일 다이얼로그
        'tkinter',
        'tkinter.filedialog',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='GD_Simulator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                     # GUI 앱: 검은 콘솔창 숨김
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
