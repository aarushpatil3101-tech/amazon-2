"""Live demo: predict the number of items in a bin photo and reconcile it with the recorded count.

  python demo.py --run runs/r34_cls --image path/to/bin.jpg --recorded 3
  python demo.py --run runs/r34_cls --csv recorded.csv --images data/images     # id,recorded_qty -> report
  python demo.py --run runs/r34_cls --gradio                                    # web UI (share link in Colab)
"""
import argparse
import os

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

from abid.common import load_json
from train_cnn import MEAN, STD, build_model, native_to_classprobs, native_to_pred, to_native


def load_run(run_dir, device):
    meta = load_json(os.path.join(run_dir, "metrics.json"))
    model = build_model(meta["arch"], meta["head"], pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    model.eval()
    tf = T.Compose([T.Resize((meta["img"], meta["img"])), T.ToTensor(), T.Normalize(MEAN, STD)])
    return model, tf, meta["head"]


@torch.no_grad()
def predict_image(model, tf, head, img, device):
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)
    nat = (to_native(model(x), head) + to_native(model(torch.flip(x, dims=[3])), head)) / 2
    probs = native_to_classprobs(nat, head)[0].cpu()
    return int(native_to_pred(nat, head).item()), probs


def verdict(pred, probs, recorded):
    if recorded is None:
        return f"Predicted items: {pred} (confidence {probs[pred]:.0%})"
    if pred == recorded:
        return f"MATCH - recorded {recorded}, model sees {pred} (P(recorded)={probs[recorded]:.0%})"
    return f"VARIANCE - recorded {recorded} but model sees {pred} (P(recorded)={probs[recorded]:.0%}) -> send for recount"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--image")
    ap.add_argument("--recorded", type=int)
    ap.add_argument("--csv")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--gradio", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, head = load_run(a.run, device)

    if a.image:
        pred, probs = predict_image(model, tf, head, Image.open(a.image), device)
        print(verdict(pred, probs, a.recorded))
        print("class probabilities:", {i: round(float(p), 3) for i, p in enumerate(probs)})

    if a.csv:
        df = pd.read_csv(a.csv)
        rows = []
        for _, r in df.iterrows():
            pred, probs = predict_image(model, tf, head, Image.open(os.path.join(a.images, f"{int(r.id)}.jpg")), device)
            rows.append({"id": int(r.id), "recorded": int(r.recorded_qty), "predicted": pred,
                         "p_recorded": round(float(probs[int(r.recorded_qty)]), 3),
                         "status": "MATCH" if pred == int(r.recorded_qty) else "VARIANCE"})
        out = pd.DataFrame(rows)
        out.to_csv("reconciliation_report.csv", index=False)
        print(out.status.value_counts().to_string())
        print("saved reconciliation_report.csv")

    if a.gradio:
        import gradio as gr

        def fn(img, recorded):
            if img is None:
                return "Upload a bin photo.", None
            rec = None if recorded is None or recorded < 0 else int(recorded)
            pred, probs = predict_image(model, tf, head, img, device)
            return verdict(pred, probs, rec), {str(i): float(p) for i, p in enumerate(probs)}

        gr.Interface(fn, [gr.Image(type="pil", label="Bin photo"),
                          gr.Number(value=-1, precision=0, label="Recorded count in inventory system (-1 = unknown)")],
                     [gr.Textbox(label="Reconciliation result"), gr.Label(label="P(number of items)")],
                     title="Amazon Inventory Reconciliation Using AI",
                     description="Counts the items in a fulfilment-centre bin and flags a mismatch with the recorded inventory.",
                     flagging_mode="never").launch(share=True)


if __name__ == "__main__":
    main()
