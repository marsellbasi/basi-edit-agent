import argparse, time, os
from pathlib import Path
import yaml
from subprocess import run, CalledProcessError

IMG_EXTS = {'.jpg','.jpeg','.png','.tif','.tiff'}

def list_new_images(inbox: Path, done_dir: Path):
    files = [p for p in inbox.rglob("*") if p.suffix.lower() in IMG_EXTS]
    new = [p for p in files if not (done_dir / (p.stem + ".done")).exists()]
    return sorted(new)

def mark_done(p: Path, done_dir: Path):
    done_dir.mkdir(parents=True, exist_ok=True)
    (done_dir / (p.stem + ".done")).write_text("ok")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--sleep", type=int, default=10)
    args = ap.parse_args()

    with open(args.config,'r') as f: cfg = yaml.safe_load(f)
    output_root = Path(cfg["output_root"]).expanduser()
    agent_root  = output_root / "agent"
    inbox  = agent_root / "inbox"
    outbox = agent_root / "outbox"
    done   = agent_root / "done"
    model_dir = Path(args.model_dir) if args.model_dir else (output_root / "models" / "color_v0")
    for d in [inbox,outbox,done]: d.mkdir(parents=True, exist_ok=True)

    print("[BASI Watch] Inbox:", inbox)
    print("[BASI Watch] Outbox:", outbox)
    print("[BASI Watch] Model:", model_dir)
    print("[BASI Watch] Press Ctrl+C to stop.")

    while True:
        try:
            imgs = list_new_images(inbox, done)
            if imgs:
                print(f"[BASI Watch] New images: {len(imgs)}")
            for src in imgs:
                cmd = [
                    os.sys.executable, str((Path(__file__).parent / "apply_color_model.py").resolve()),
                    "--config", str(Path(args.config).resolve()),
                    "--model_dir", str(model_dir.resolve()),
                    "--input_glob", str(src.resolve()),
                    "--out_dir", str(outbox.resolve())
                ]
                try:
                    run(cmd, check=True)
                    mark_done(src, done)
                except CalledProcessError as e:
                    print("[BASI Watch] ERROR:", e)
            time.sleep(args.sleep)
        except KeyboardInterrupt:
            print("\n[BASI Watch] Stopped.")
            break
        except Exception as e:
            print("[BASI Watch] ERROR:", e)
            time.sleep(args.sleep)

if __name__ == "__main__":
    main()
