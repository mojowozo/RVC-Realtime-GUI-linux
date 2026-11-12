import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from engine import core
from tools.model_registry import IndexFileError, attach_index, discover_models
from tests.test_model_library import ModelFolders


class AttachIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "models" / "Voice"
        self.model.mkdir(parents=True)
        (self.model / "Voice.pth").write_bytes(b"pth")
        self.downloads = self.root / "downloads"
        self.downloads.mkdir()
        self.trashed = []

    def index(self, name, data=b"faiss index"):
        path = self.downloads / name
        path.write_bytes(data)
        return path

    def trash(self, path):
        self.trashed.append(Path(path).name)
        os.remove(path)

    def test_copies_the_index_into_the_model(self):
        target, replaced = attach_index(self.model, self.index("added_Voice.index"), trash=self.trash)
        self.assertEqual(target, self.model / "added_Voice.index")
        self.assertEqual(target.read_bytes(), b"faiss index")
        self.assertEqual(replaced, [])
        (entry,) = discover_models(self.root / "models")
        self.assertEqual(entry.index_path.name, "added_Voice.index")
        self.assertEqual(sorted(p.name for p in self.model.iterdir()), ["Voice.pth", "added_Voice.index"])

    def test_replaces_old_indexes_including_one_with_the_same_name(self):
        (self.model / "added_Voice.index").write_bytes(b"old")
        (self.model / "added_Other.index").write_bytes(b"older")
        target, replaced = attach_index(self.model, self.index("added_Voice.index", b"new"), trash=self.trash)
        self.assertEqual(sorted(replaced), ["added_Other.index", "added_Voice.index"])
        self.assertEqual(sorted(self.trashed), ["added_Other.index", "added_Voice.index"])
        self.assertEqual(target.read_bytes(), b"new")
        self.assertEqual(sorted(p.name for p in self.model.glob("*.index")), ["added_Voice.index"])

    def test_rejects_the_trained_index_and_other_files(self):
        for name, code in (("trained_IVF256_Flat_Voice.index", "index_trained"), ("Voice.pth", "index_not_index")):
            with self.subTest(name=name):
                with self.assertRaises(IndexFileError) as raised:
                    attach_index(self.model, self.index(name), trash=self.trash)
                self.assertEqual(raised.exception.code, code)
        with self.assertRaises(IndexFileError) as raised:
            attach_index(self.model, self.index("added_Empty.index", b""), trash=self.trash)
        self.assertEqual(raised.exception.code, "index_missing_file")
        self.assertEqual(list(self.model.glob("*.index")), [])

    def test_a_failed_copy_leaves_the_model_untouched(self):
        (self.model / "added_Voice.index").write_bytes(b"old")
        with mock.patch("tools.model_registry.shutil.copyfile", side_effect=OSError("disk full")):
            with self.assertRaises(IndexFileError):
                attach_index(self.model, self.index("added_New.index"), trash=self.trash)
        self.assertEqual(self.trashed, [])
        self.assertEqual(sorted(p.name for p in self.model.iterdir()), ["Voice.pth", "added_Voice.index"])


class EngineAddIndexTest(ModelFolders):
    def test_attaches_reloads_and_updates_the_loaded_model(self):
        engine = self.engine(selected="Bass")
        source = Path(self.temp.name) / "added_Bass.index"
        source.write_bytes(b"faiss")
        loaded = []
        engine.rvc = SimpleNamespace(change_index=loaded.append)
        with mock.patch("tools.model_registry.move_to_trash"):
            result = engine.add_index("Bass", str(source))
        self.assertEqual(result, {"index_file": "added_Bass.index", "replaced": []})
        self.assertEqual(engine.models_by_name["Bass"].index_path.name, "added_Bass.index")
        self.assertEqual(loaded, [str((self.root / "Bass" / "added_Bass.index").resolve())])
        self.assertIn("index_added", [data["code"] for event, data in engine.events if event == "status"])

    def test_other_models_leave_the_loaded_one_alone(self):
        engine = self.engine(selected="Bass")
        source = Path(self.temp.name) / "added_Alto.index"
        source.write_bytes(b"faiss")
        loaded = []
        engine.rvc = SimpleNamespace(change_index=loaded.append)
        engine.add_index("Alto", str(source))
        self.assertEqual(loaded, [])

    def test_errors_carry_their_code(self):
        engine = self.engine()
        bad = Path(self.temp.name) / "trained_Alto.index"
        bad.write_bytes(b"x")
        with self.assertRaises(core.EngineError) as raised:
            engine.add_index("Alto", str(bad))
        self.assertEqual(raised.exception.code, "index_trained")
        with self.assertRaises(core.EngineError) as raised:
            engine.add_index("Nobody", str(bad))
        self.assertEqual(raised.exception.code, "no_model")


if __name__ == "__main__":
    unittest.main()
