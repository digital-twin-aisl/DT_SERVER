// SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
// SPDX-License-Identifier: LGPL-2.1-or-later
// Screen-space aid only: no unprojection, snapping or landmark mutations.
export function pixelFromPointer(event, box, width, height) {
  if (!(box.width > 0 && box.height > 0 && width > 0 && height > 0)) return null;
  const x = (event.clientX - box.left) / box.width * width;
  const y = (event.clientY - box.top) / box.height * height;
  return Number.isFinite(x) && Number.isFinite(y) && x >= 0 && y >= 0 && x < width && y < height ? [x, y] : null;
}

export function placePixelGuide(element, pixel, width, height, visible = true) {
  element.hidden = !visible || !pixel;
  if (element.hidden) return;
  element.style.left = `${pixel[0] / width * 100}%`;
  element.style.top = `${pixel[1] / height * 100}%`;
}

export function renderPixelGuides(container, pixels, width, height, onRemove, visible = true) {
  container.hidden = !visible || pixels.length === 0;
  container.replaceChildren();
  if (container.hidden) return;
  pixels.forEach((pixel, index) => {
    const dot = container.ownerDocument.createElement('button');
    dot.type = 'button';
    dot.className = 'pixel-guide';
    dot.title = `Remove guide ${index + 1} (X ${pixel[0].toFixed(1)}, Y ${pixel[1].toFixed(1)})`;
    dot.setAttribute('aria-label', dot.title);
    placePixelGuide(dot, pixel, width, height);
    dot.onclick = event => { event.stopPropagation(); onRemove(index); };
    container.append(dot);
  });
}
