# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Run with Isaac Sim's python.sh, in a new headless process and empty Stage."""

from pathlib import Path
import sys
import tempfile
import time
import traceback

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "hide_ui": False, "width": 64, "height": 64,
                     "multi_gpu": False, "sync_loads": True})
exit_code = 0
player = None
try:
    import omni.kit.app
    import omni.ui as ui
    import omni.usd
    from pxr import UsdGeom

    tests_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(tests_dir))
    from test_playback import EXTENSION, SceneOutputRecorder, scene, person
    from meta_sejong.scene_player.extension import ScenePlayerExtension
    import json

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    before = stage.GetRootLayer().ExportToString()

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "smoke.jsonl"
        with SceneOutputRecorder(path) as recorder:
            for value in (scene(0, [person()]), scene(0.1, [person(pose=False)]), scene(0.2)):
                recorder.write_payload(json.dumps(value).encode())
        player = ScenePlayerExtension()
        player.on_startup("meta_sejong.scene_player.smoke")
        player._path.set_value(str(path))
        player._load()
        deadline = time.monotonic() + 30
        while player._loading is not None and time.monotonic() < deadline:
            app.update()
        assert player._player is not None, player._status.text
        root_path = player._renderer.root_path
        assert stage.GetPrimAtPath(root_path).IsValid()
        assert player._people_count == 1
        player._seek(1)
        assert player._player.index == 1
        player._frame.set_value(2)
        assert player._player.index == 2 and player._people_count == 0
        player._seek(0)
        player._play()
        deadline = time.monotonic() + 10
        while player._player.playing and time.monotonic() < deadline:
            app.update()
        assert player._player.index == 2 and not player._player.playing
        player._unload()
        assert not stage.GetPrimAtPath(root_path).IsValid()
        assert stage.GetRootLayer().ExportToString() == before
        player.on_shutdown()
        player = None
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.add_path(str(EXTENSION.parent))
        assert manager.set_extension_enabled_immediate("meta_sejong.scene_player", True)
        assert ui.Workspace.get_window("Meta Sejong Scene Player") is not None
        manager.set_extension_enabled_immediate("meta_sejong.scene_player", False)
        app.update()
        print("SCENE_PLAYER_ISAAC_SMOKE_OK", flush=True)
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    if player is not None:
        player.on_shutdown()
    app.close()
raise SystemExit(exit_code)
