"""Build a static GitHub Pages site. Copies weights; never trains."""
import argparse
from importlib.metadata import distribution
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import chess

ROOT = Path(__file__).resolve().parent


def build(output, model):
    if not model.is_file():
        raise FileNotFoundError(model)
    output.mkdir(parents=True, exist_ok=True)
    html = (ROOT / "chess_web.html").read_text()
    marker = '<meta name="chess-runtime" content="server">'
    if html.count(marker) != 1:
        raise ValueError("Missing or duplicate browser runtime marker")
    (output / "index.html").write_text(html.replace(marker, marker.replace("server", "browser")))
    for name in ("chess_worker.js", "chess_position.py", "neko.js", "favicon.svg"):
        shutil.copyfile(ROOT / name, output / name)
    shutil.copyfile(ROOT.parent / "chesslm.py", output / "chesslm.py")
    shutil.copyfile(model, output / "model.npz")
    # Bundle the installed pure-Python package; no PyPI access is needed in the browser.
    package = Path(chess.__file__).parent
    license_file = next(f for f in distribution("chess").files if str(f).endswith("licenses/LICENSE.txt"))
    with ZipFile(output / "chess.zip", "w", ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*.py")):
            archive.write(path, "chess/" + path.relative_to(package).as_posix())
        archive.write(distribution("chess").locate_file(license_file), "chess/LICENSE.txt")
    (output / ".nojekyll").touch()
    print(f"Static site written to {output}. No training performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "_site")
    parser.add_argument("--model", type=Path, default=ROOT.parent / "model.npz")
    args = parser.parse_args()
    build(args.output.resolve(), args.model.resolve())


if __name__ == "__main__":
    main()
