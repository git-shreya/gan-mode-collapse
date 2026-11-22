"""
Fashion Design Idea Generator: Gradio demo (Phase 4).

Uses ONLY the improved GAN (spectral norm + hinge loss).
User flow:
  1. Pick a category and how many designs  ->  Generate
  2. Click any design to pick it           ->  "More like this" gives close variations
  3. Pick two designs                       ->  "Mix my last two picks" blends them step by step
  4. Download everything on screen as one mood-board image

    python app.py                       # local
    python app.py --share               # public link (use this in Colab)
"""
import argparse
import os
import tempfile

import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.models import load_generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISPLAY_SIZE = 192  # images are generated at 64x64 and upscaled only for viewing
CONF_THRESHOLD = 0.6

G = None
CLF = None


# ------------------------------------------------------------------ helpers
def to_pil(x):
    """(3,64,64) tensor in [-1,1] -> upscaled PIL image."""
    arr = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)


@torch.no_grad()
def render(z):
    return [to_pil(img) for img in G(torch.as_tensor(z, dtype=torch.float32, device=DEVICE))]


def slerp(a, b, t):
    """Spherical interpolation, works better than a straight line for Gaussian latents."""
    a_n, b_n = a / np.linalg.norm(a), b / np.linalg.norm(b)
    omega = np.arccos(np.clip(np.dot(a_n, b_n), -1, 1))
    if omega < 1e-6:
        return (1 - t) * a + t * b
    return (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / np.sin(omega)


def empty_state():
    return {"z_main": None, "z_var": None, "z_mix": None, "picks": []}


# ------------------------------------------------------------------ actions
@torch.no_grad()
def generate(category, n, seed, state):
    n = int(n)
    rng = np.random.default_rng(int(seed) if seed not in (None, "") else None)
    msg = ""
    if category == "All categories" or CLF is None:
        z = rng.standard_normal((n, G.nz)).astype(np.float32)
    else:
        target = CLF.class_names.index(category)
        found = []
        for _ in range(30):  # sample in batches, keep what the classifier is confident about
            zb = rng.standard_normal((256, G.nz)).astype(np.float32)
            pred, conf = CLF.predict(G(torch.from_numpy(zb).to(DEVICE)))
            keep = ((pred == target) & (conf >= CONF_THRESHOLD)).numpy()
            found.extend(zb[keep])
            if len(found) >= n:
                break
        if not found:
            return [], state, f"Couldn't find confident **{category}** designs. Try another category or 'All categories'."
        z = np.stack(found[:n])
        if len(z) < n:
            msg = f"Found {len(z)} confident {category} designs (asked for {n})."
    state = {**state, "z_main": z, "z_var": None, "z_mix": None, "picks": []}
    return render(z), state, msg or f"Generated {len(z)} new designs. Click any design to pick it."


def pick(source):
    def _pick(state, evt: gr.SelectData):
        z = state.get(source)
        if z is None:
            return state, [], "Generate some designs first."
        picks = (state["picks"] + [z[evt.index]])[-2:]  # keep the last two picks
        state = {**state, "picks": picks}
        hint = "Now click 'More like this', or pick one more design to mix." if len(picks) == 1 \
            else "Two designs picked. Try 'Mix my last two picks'."
        return state, render(np.stack(picks)), hint
    return _pick


def more_like_this(strength, state):
    if not state["picks"]:
        return [], state, "Pick a design first by clicking on it."
    base = state["picks"][-1]
    rng = np.random.default_rng()
    z = base[None, :] + float(strength) * rng.standard_normal((8, G.nz)).astype(np.float32)
    state = {**state, "z_var": z.astype(np.float32)}
    return render(z), state, "Here are 8 close variations of your last pick. You can pick from these too."


def mix(state):
    if len(state["picks"]) < 2:
        return [], state, "Pick two designs first (click one, then another)."
    a, b = state["picks"]
    z = np.stack([slerp(a, b, t) for t in np.linspace(0, 1, 8)]).astype(np.float32)
    state = {**state, "z_mix": z}
    return render(z), state, "Left = first pick, right = second pick. The middle ones are new blends."


def mood_board(state):
    blocks = [state[k] for k in ("z_main", "z_var", "z_mix") if state.get(k) is not None]
    if not blocks:
        return None, "Nothing to download yet."
    imgs = render(np.concatenate(blocks))
    cols = 8
    rows = int(np.ceil(len(imgs) / cols))
    pad = 6
    board = Image.new("RGB", (cols * (DISPLAY_SIZE + pad) + pad, rows * (DISPLAY_SIZE + pad) + pad), "white")
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        board.paste(im, (pad + c * (DISPLAY_SIZE + pad), pad + r * (DISPLAY_SIZE + pad)))
    path = os.path.join(tempfile.gettempdir(), "mood_board.png")
    board.save(path)
    return path, f"Mood board with {len(imgs)} designs is ready to download."


# ------------------------------------------------------------------ UI
def build_ui():
    categories = ["All categories"] + (list(CLF.class_names) if CLF is not None else [])
    with gr.Blocks(title="Fashion Design Idea Generator") as demo:
        state = gr.State(empty_state())
        gr.Markdown(
            "# Fashion Design Idea Generator\n"
            "Get new product design ideas in one click. Every image is new and created by a GAN "
            "trained on Myntra catalogue photos, using spectral normalization to keep the designs varied.\n\n"
            "*Images are 64x64 thumbnails meant for ideas on colour, pattern and shape, "
            "not for use as product photos.*")
        with gr.Row():
            category = gr.Dropdown(categories, value="All categories", label="1. Category")
            n = gr.Slider(8, 32, value=16, step=8, label="How many designs")
            seed = gr.Number(value=None, label="Seed (optional, to get the same batch again)", precision=0)
        gen_btn = gr.Button("Generate designs", variant="primary")
        status = gr.Markdown()
        main_gallery = gr.Gallery(label="2. Generated designs (click to pick)", columns=8, height="auto",
                                  allow_preview=False)

        with gr.Row():
            picks_gallery = gr.Gallery(label="Your picks (last two)", columns=2, height=220, allow_preview=False)
            with gr.Column():
                strength = gr.Slider(0.1, 0.8, value=0.35, step=0.05,
                                     label="Variation strength (low = very similar, high = more different)")
                more_btn = gr.Button("3. More like this")
                mix_btn = gr.Button("4. Mix my last two picks")
        var_gallery = gr.Gallery(label="Variations of your pick (click to pick)", columns=8, height="auto",
                                 allow_preview=False)
        mix_gallery = gr.Gallery(label="Blend: first pick to second pick", columns=8, height="auto",
                                 allow_preview=False)
        with gr.Row():
            dl_btn = gr.Button("5. Create mood board")
            dl_file = gr.File(label="Download")

        gen_btn.click(generate, [category, n, seed, state], [main_gallery, state, status])
        main_gallery.select(pick("z_main"), [state], [state, picks_gallery, status])
        var_gallery.select(pick("z_var"), [state], [state, picks_gallery, status])
        mix_gallery.select(pick("z_mix"), [state], [state, picks_gallery, status])
        more_btn.click(more_like_this, [strength, state], [var_gallery, state, status])
        mix_btn.click(mix, [state], [mix_gallery, state, status])
        dl_btn.click(mood_board, [state], [dl_file, status])
    return demo


def main():
    global G, CLF
    p = argparse.ArgumentParser()
    p.add_argument("--generator", default="checkpoints/G_improved.pt")
    p.add_argument("--classifier", default="checkpoints/classifier.pt")
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    G = load_generator(a.generator, DEVICE)
    if os.path.exists(a.classifier):
        from src.classifier import CategoryClassifier
        CLF = CategoryClassifier(a.classifier, DEVICE)
    else:
        print("Classifier not found: category filter disabled.")
    build_ui().launch(share=a.share, server_port=a.port)


if __name__ == "__main__":
    main()
