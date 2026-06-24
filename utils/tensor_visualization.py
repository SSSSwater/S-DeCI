import math
from pathlib import Path

import numpy as np


def _to_numpy(item):
    if hasattr(item, "detach") and callable(item.detach):
        item = item.detach()
        if hasattr(item, "cpu") and callable(item.cpu):
            item = item.cpu()
        if hasattr(item, "numpy") and callable(item.numpy):
            return item.numpy()
    return np.asarray(item)


def _to_heatmap_array(item, index=0):
    arr = np.asarray(_to_numpy(item))
    original_shape = arr.shape
    original_ndim = arr.ndim

    if arr.ndim == 0:
        raise ValueError("Cannot visualize a scalar tensor; expected 1D, 2D, or 3D input.")
    if arr.ndim > 3:
        raise ValueError(f"Cannot visualize {arr.ndim}D input; expected 1D, 2D, or 3D input.")

    if arr.ndim == 1:
        return arr.reshape(1, -1), original_shape, original_ndim
    if arr.ndim == 2:
        return arr, original_shape, original_ndim

    if arr.shape[0] == 0:
        raise ValueError("Cannot visualize a 3D tensor with an empty batch dimension.")
    if index < 0 or index >= arr.shape[0]:
        raise ValueError(f"batch_index {index} is out of range for batch size {arr.shape[0]}.")
    return arr[index], original_shape, original_ndim


def _subplot_shape(count):
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    return rows, cols


def _subtitle(shape, ndim, batch_index):
    text = f"shape={tuple(shape)}"
    if ndim == 3:
        text += f" | showing Batch{batch_index}"
    return text


def visualize_tensors(
    *items,
    titles=None,
    cmap="viridis",
    figsize=None,
    colorbar=True,
    save_path=None,
    show=False,
    batch_index=0,
    squeeze=False,
):
    """Visualize one or more 1D/2D/3D tensors as heatmaps.

    For 3D tensors, only ``batch_index`` is shown. The default is Batch0.
    Returns the matplotlib ``(fig, axes)`` pair for further customization.
    """
    if not items:
        raise ValueError("visualize_tensors requires at least one tensor or matrix.")

    if titles is not None and len(titles) != len(items):
        raise ValueError(
            f"titles length ({len(titles)}) must match the number of inputs ({len(items)})."
        )

    plot_items = []
    for item in items:
        arr = _to_numpy(item)
        if squeeze:
            arr = np.squeeze(arr)
        plot_items.append(_to_heatmap_array(arr, index=batch_index))

    import matplotlib.pyplot as plt

    rows, cols = _subplot_shape(len(plot_items))
    if figsize is None:
        figsize = (4 * cols, 3 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    flat_axes = axes.ravel()

    for i, (ax, (arr, original_shape, original_ndim)) in enumerate(zip(flat_axes, plot_items)):
        image = ax.imshow(arr, aspect="auto", cmap=cmap)
        heading = titles[i] if titles is not None else f"Tensor {i + 1}"
        ax.set_title(f"{heading}\n{_subtitle(original_shape, original_ndim, batch_index)}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        if colorbar:
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax in flat_axes[len(plot_items) :]:
        ax.axis("off")

    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        if path.parent and str(path.parent) != ".":
            path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes
