import os, hashlib
import numpy as np
import pandas as pd
from PIL import Image

TARGET_IMAGE_SIZE = 128

PREFIX_MAP = {
    "SAT-Race-2010-CNF": "Application SAT+UNSAT/SAT Race 2010",
    "SATCompetition2007": "SATCompetition2007/industrial",
    "SATCompetition2009": "SATCompetition2009/application",
    "SATCompetition2011": "SATCompetition2011/application",
    "SC2012_Application_Debugged": "SC2012_Application_Debugged/satrace-unselected",
}

def _resolve_path(instances_root: str, instance_id: str):
    if not isinstance(instance_id, str) or not instance_id.strip():
        return None

    p1 = os.path.join(instances_root, instance_id)
    if os.path.exists(p1):
        return p1

    parts = instance_id.split("/", 1)
    if len(parts) == 2 and parts[0] in PREFIX_MAP:
        p2 = os.path.join(instances_root, PREFIX_MAP[parts[0]], parts[1])
        if os.path.exists(p2):
            return p2

    base = os.path.basename(instance_id)
    for root, _, files in os.walk(instances_root):
        if base in files:
            return os.path.join(root, base)

    return None

def convert_raw_text_to_image_matrix(raw_text_path: str, target_size: int = TARGET_IMAGE_SIZE):
    with open(raw_text_path, "rb") as f:
        data = f.read()

    vector = np.frombuffer(data, dtype=np.uint8)
    N = vector.size
    if N == 0:
        return np.zeros((target_size, target_size), dtype=np.float32)

    lado = int(np.floor(np.sqrt(N)))
    usable = lado * lado
    if usable <= 0:
        return np.zeros((target_size, target_size), dtype=np.float32)

    img0 = vector[:usable].reshape(lado, lado).astype(np.uint8)
    img = Image.fromarray(img0, mode="L").resize((target_size, target_size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    std = float(arr.std())
    if std > 0:
        arr = (arr - float(arr.mean())) / std
    return arr

def generate_all_images(csv_path: str, instances_root: str):
    df = pd.read_csv(csv_path)
    out_dir = os.path.dirname(csv_path) or "."
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    raw_paths = []
    image_paths = []
    ok, miss = 0, 0

    for _, row in df.iterrows():
        raw = row.get("Raw_Text_Path")
        if not (isinstance(raw, str) and os.path.exists(raw)):
            raw = _resolve_path(instances_root, row.get("Instance_Id", ""))

        if raw and os.path.exists(raw):
            raw_paths.append(raw)
            try:
                arr = convert_raw_text_to_image_matrix(raw, TARGET_IMAGE_SIZE)
                iid = str(row.get("Instance_Id", row.get("Instance_Name", "")))
                h = hashlib.md5(iid.encode("utf-8", errors="ignore")).hexdigest()[:12]
                base = os.path.splitext(os.path.basename(iid))[0]
                npy_path = os.path.join(img_dir, f"{base}__{h}.npy")
                np.save(npy_path, arr.astype(np.float32))
                image_paths.append(npy_path)
                ok += 1
            except Exception as e:
                print(f"⚠️  Fallo al convertir {raw}: {e}")
                image_paths.append("")
                miss += 1
        else:
            raw_paths.append("" if raw is None else raw)
            image_paths.append("")
            miss += 1

    df["Raw_Text_Path"] = raw_paths
    df["Image_Npy_Path"] = image_paths
    df.to_csv(csv_path, index=False)
    print(f"✅ Image conversion complete. NPY paths added to CSV. ({ok} ok, {miss} sin imagen)")
