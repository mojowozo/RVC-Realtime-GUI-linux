import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelEntry:
    name: str
    directory: Path
    model_path: Path
    index_path: Path | None
    #: Size of the model file in bytes and its modification time (Unix).
    size_bytes: int = 0
    modified: float = 0.0


def discover_models(models_root: str | Path) -> list[ModelEntry]:
    """Discover one RVC model per direct child directory.

    A directory is listed when it contains at least one ``.pth`` file (or,
    failing that, a ``.nextvc.json`` voice description).
    When multiple files exist, names are sorted case-insensitively and the
    first file is selected.  An ``added_*.index`` file is preferred over
    other index files.
    """
    root = Path(models_root)
    if not root.is_dir():
        return []

    entries = []
    for directory in sorted(
        # Hidden folders are skipped, including in-progress imports (.import-*).
        (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name.casefold(),
    ):
        model_files = sorted(
            directory.glob("*.pth"), key=lambda path: path.name.casefold()
        )
        if not model_files:
            # A voice of the streaming NextVC backend (infer/nextvc_backend.py).
            model_files = sorted(
                directory.glob("*.nextvc.json"), key=lambda path: path.name.casefold()
            )
        if not model_files:
            continue

        index_files = sorted(
            directory.glob("*.index"),
            key=lambda path: (
                not path.name.casefold().startswith("added_"),
                path.name.casefold(),
            ),
        )
        try:
            stat = model_files[0].stat()
        except OSError:
            continue
        entries.append(
            ModelEntry(
                name=directory.name,
                directory=directory.resolve(),
                model_path=model_files[0].resolve(),
                index_path=index_files[0].resolve() if index_files else None,
                size_bytes=stat.st_size,
                modified=stat.st_mtime,
            )
        )
    return entries


#: Characters a model folder name cannot contain (also invalid on NTFS/FAT).
INVALID_NAME_CHARACTERS = '/\\:*?"<>|'


def model_name_problem(name):
    """Why ``name`` cannot be a model folder name, or ``None`` if it can."""
    if not name or not name.strip():
        return "empty"
    if name != name.strip() or name.endswith("."):
        return "edges"
    if name.startswith("."):
        return "hidden"
    if any(character in INVALID_NAME_CHARACTERS or ord(character) < 32 for character in name):
        return "characters"
    if len(name.encode("utf-8")) > 200:
        return "too_long"
    return None


def rename_model_folder(directory, new_name):
    """Rename a model folder within its parent; returns the new path.

    Case-only renames (``voice`` -> ``Voice``) go through a temporary name,
    which case-insensitive filesystems such as NTFS need.
    """
    directory = Path(directory)
    target = directory.parent / new_name
    same_folder = target.exists() and os.path.samefile(target, directory)
    if target.exists() and not same_folder:
        raise FileExistsError(new_name)
    if same_folder and directory.name != new_name:
        temporary = directory.parent / f".rename-{os.getpid()}-{directory.name}"
        os.rename(directory, temporary)
        directory = temporary
    os.rename(directory, target)
    return target


class TrashError(Exception):
    pass


class IndexFileError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def attach_index(model_directory, index_file, trash=None):
    """Copy ``index_file`` into a model folder; return its new path.

    The copy goes to a hidden temporary name first and is renamed into place
    only when complete.  Index files already in the folder are passed to
    ``trash`` (default: the desktop Trash), so the model has exactly one.
    """
    source = Path(index_file)
    if source.suffix.lower() != ".index":
        raise IndexFileError("index_not_index", f"Not an .index file: {source.name}")
    if source.name.lower().startswith("trained_"):
        # RVC trainers write trained_*.index (not usable for retrieval)
        # beside added_*.index (the one to use).
        raise IndexFileError(
            "index_trained",
            f"{source.name} is the trained_ index; use the added_ index from the same folder.",
        )
    if not source.is_file() or source.stat().st_size == 0:
        raise IndexFileError("index_missing_file", f"Index file not found: {source}")
    directory = Path(model_directory)
    target = directory / source.name
    temporary = directory / f".{source.name}.part"
    try:
        shutil.copyfile(source, temporary)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise IndexFileError("index_copy_failed", str(error)) from error
    # Every index already there goes, including one with the same name.
    old = sorted(path for path in directory.glob("*.index") if not path.name.startswith("."))
    try:
        for path in old:
            (trash or move_to_trash)(path)
    except (OSError, subprocess.SubprocessError, TrashError) as error:
        temporary.unlink(missing_ok=True)
        raise IndexFileError("index_copy_failed", f"Could not move the old index to the Trash: {error}") from error
    os.replace(temporary, target)
    return target, [path.name for path in old]


def move_to_trash(path):
    """Move a model folder to the desktop Trash (recoverable), via ``gio``."""
    gio = shutil.which("gio")
    if gio is None:
        raise TrashError("gio was not found, so the folder cannot be moved to the Trash")
    completed = subprocess.run(
        [gio, "trash", "--", os.fspath(path)], capture_output=True, text=True, timeout=60
    )
    if completed.returncode != 0:
        raise TrashError(completed.stderr.strip() or f"gio trash failed ({completed.returncode})")
