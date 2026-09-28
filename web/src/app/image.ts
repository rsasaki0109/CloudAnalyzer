/** Saving the view as a PNG with its overlays. */

import { rampColor } from "./distance";
import { $, download, errorText, setStatus } from "./dom";
import { profileCanvas, profilePlot } from "./profile";
import { viewer } from "./state";

/** Draw an HTML overlay box (background, border, radius) onto the image. */
function drawBox(ctx: CanvasRenderingContext2D, el: HTMLElement, origin: DOMRect): void {
  const style = getComputedStyle(el);
  const r = el.getBoundingClientRect();
  const [x, y, w, h] = [r.left - origin.left, r.top - origin.top, r.width, r.height];
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, Number.parseFloat(style.borderTopLeftRadius) || 0);
  ctx.fillStyle = style.backgroundColor;
  ctx.fill();
  const border = Number.parseFloat(style.borderTopWidth) || 0;
  if (border > 0) {
    ctx.lineWidth = border;
    ctx.strokeStyle = style.borderTopColor;
    ctx.stroke();
  }
}

/** Draw an element's text where it is laid out. */
function drawText(ctx: CanvasRenderingContext2D, el: HTMLElement, origin: DOMRect): void {
  const text = el.textContent?.trim();
  if (!text || el.hidden) return;
  const style = getComputedStyle(el);
  const r = el.getBoundingClientRect();
  ctx.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
  ctx.fillStyle = style.color;
  ctx.textBaseline = "middle";
  const padding = Number.parseFloat(style.paddingLeft) || 0;
  ctx.fillText(text, r.left - origin.left + padding, r.top - origin.top + r.height / 2);
}

/** The view as a PNG: the 3D render with labels, the colorbar and the profile plot on top. */
export async function viewImage(): Promise<Blob> {
  const shot = viewer.snapshot();
  const origin = $("viewport").getBoundingClientRect();
  const scale = shot.width / origin.width;
  const ctx = shot.getContext("2d")!;
  ctx.scale(scale, scale);
  for (const el of $("labels").querySelectorAll<HTMLElement>("span")) {
    if (el.hidden) continue;
    drawBox(ctx, el, origin);
    drawText(ctx, el, origin);
  }
  const colorbar = $("colorbar");
  if (!colorbar.hidden) {
    drawBox(ctx, colorbar, origin);
    drawText(ctx, $("colorbar-title"), origin);
    const ramp = $("colorbar-ramp").getBoundingClientRect();
    for (let i = 0; i < ramp.height; i++) {
      ctx.fillStyle = rampColor(1 - i / Math.max(1, ramp.height - 1));
      ctx.fillRect(ramp.left - origin.left, ramp.top - origin.top + i, ramp.width, 1.5);
    }
    for (const id of ["colorbar-max", "colorbar-mid", "colorbar-min"]) drawText(ctx, $(id), origin);
  }
  if (!profilePlot.hidden) {
    drawBox(ctx, profilePlot, origin);
    for (const el of profilePlot.querySelectorAll<HTMLElement>(".profile-legend span, #profile-readout")) {
      drawText(ctx, el, origin);
    }
    const r = profileCanvas.getBoundingClientRect();
    ctx.drawImage(profileCanvas, r.left - origin.left, r.top - origin.top, r.width, r.height);
  }
  return new Promise((resolve, reject) =>
    shot.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("could not encode the image"))), "image/png"),
  );
}

$<HTMLButtonElement>("save-image").onclick = async () => {
  try {
    const blob = await viewImage();
    download(blob, "cloudanalyzer-view.png");
    setStatus(`Saved cloudanalyzer-view.png (${(blob.size / 1e6).toFixed(1)} MB)`);
  } catch (err) {
    setStatus(`Image failed: ${errorText(err)}`, true);
  }
};
