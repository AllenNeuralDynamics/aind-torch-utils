#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def _rolling_mean(values: List[float], window: int) -> List[float]:
    """
    Simple rolling mean (same length output) using a centered window.

    Pads edges by shrinking the window near boundaries, a common pattern for
    quick visualization to avoid the phase shift that occurs with causal
    (one-sided) windows.

    Parameters
    ----------
    values : List[float]
        The list of numerical values to process.
    window : int
        The size of the rolling window. If <= 1, the original list is returned.

    Returns
    -------
    List[float]
        The list of smoothed values.
    """
    if window <= 1 or not values:
        return values
    n = len(values)
    out = []
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        segment = values[start:end]
        out.append(sum(segment) / len(segment))
    return out


def plot_queues(
    data: Dict[str, Any],
    out_path: str,
    show_mean: bool,
    smooth: int,
) -> None:
    """
    Generates and saves plots for queue sizes over time.

    A separate plot is created for each queue found in the data.

    Parameters
    ----------
    data : Dict[str, Any]
        A dictionary containing the queue monitoring data, expected to have a
        "queues" key.
    out_path : str
        The base path for saving the output plot images. The queue name will be
        appended.
    show_mean : bool
        If True, a horizontal line indicating the mean value is drawn on the plot.
    smooth : int
        The window size for a rolling mean to smooth the data. If > 1, both
        raw and smoothed data are shown.
    """
    # One plot per queue
    for qname, qdata in data.get("queues", {}).items():
        samples = qdata.get("samples", [])
        if not samples:
            continue

        times = [s["t_rel_s"] for s in samples]
        values = [s["size"] for s in samples]
        max_size = qdata.get("max_size", 0)

        plt.figure()
        y_raw = values
        if smooth > 1:
            y_smooth = _rolling_mean(y_raw, smooth)
            plt.plot(times, y_raw, alpha=0.35, linewidth=1, label="raw")
            plt.plot(
                times,
                y_smooth,
                linewidth=1.8,
                label=f"rolling_mean(w={smooth})",
            )
        else:
            plt.plot(times, y_raw, linewidth=1.2, label="value")

        title = f"Queue size over time: {qname}"
        if max_size > 0:
            title += f" (max={max_size})"
            plt.axhline(
                max_size,
                color="r",
                linestyle="--",
                linewidth=1,
                label=f"max_size ({max_size})",
            )
        if show_mean and y_raw:
            mean_val = sum(y_raw) / len(y_raw)
            plt.axhline(
                mean_val,
                color="k",
                linestyle=":",
                linewidth=1.2,
                label=f"mean={mean_val:.2f}",
            )
        plt.legend()

        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Items in queue")
        plt.grid(True)
        plt.ylim(bottom=0)

        if out_path:
            stem = f"{out_path}_{qname}.png"
            plt.savefig(stem, dpi=150, bbox_inches="tight")


def plot_system(
    data: Dict[str, Any], out_path: str, show_mean: bool, smooth: int
) -> None:
    """
    Generates and saves plots for system metrics (CPU, RAM, Net I/O, Disk I/O).

    Parameters
    ----------
    data : Dict[str, Any]
        A dictionary containing the system monitoring data.
    out_path : str
        The base path for saving the output plot images. The metric name will
        be appended (e.g., "_cpu.png").
    show_mean : bool
        If True, a horizontal line indicating the mean value is drawn on the plot.
    smooth : int
        The window size for a rolling mean to smooth the data. If > 1, both
        raw and smoothed data are shown.
    """
    times = data.get("t_rel_s", [])
    if not times:
        return

    # Plot CPU
    cpu_percent = data.get("cpu_percent", [])
    if cpu_percent:
        plt.figure()
        cpu_percent[0] = 0.0  # first sample is always bogus
        y_raw = cpu_percent
        y_smooth = _rolling_mean(y_raw, smooth)
        if smooth > 1:
            plt.plot(times, y_raw, alpha=0.35, linewidth=1, label="raw")
            plt.plot(
                times,
                y_smooth,
                linewidth=1.8,
                label=f"rolling_mean(w={smooth})",
            )
        else:
            plt.plot(times, y_raw, linewidth=1.2, label="CPU %")
        plt.title("CPU Usage Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("CPU (%)")
        plt.grid(True)
        plt.ylim(bottom=0, top=max(cpu_percent) * 1.1)
        if show_mean and y_raw:
            mean_val = sum(y_raw) / len(y_raw)
            plt.axhline(
                mean_val,
                color="k",
                linestyle=":",
                linewidth=1.2,
                label=f"mean={mean_val:.2f}",
            )
        plt.legend()
        if out_path:
            plt.savefig(f"{out_path}_cpu.png", dpi=150, bbox_inches="tight")

    # Plot RAM
    ram_rss_mb = data.get("ram_rss_mb", [])
    if ram_rss_mb:
        plt.figure()
        y_raw = ram_rss_mb
        y_smooth = _rolling_mean(y_raw, smooth)
        if smooth > 1:
            plt.plot(times, y_raw, alpha=0.35, linewidth=1, label="raw")
            plt.plot(
                times,
                y_smooth,
                linewidth=1.8,
                label=f"rolling_mean(w={smooth})",
            )
        else:
            plt.plot(times, y_raw, linewidth=1.2, label="RAM")
        plt.title("RAM Usage Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("RAM (RSS, MB)")
        plt.grid(True)
        plt.ylim(bottom=0)
        if show_mean and y_raw:
            mean_val = sum(y_raw) / len(y_raw)
            plt.axhline(
                mean_val,
                color="k",
                linestyle=":",
                linewidth=1.2,
                label=f"mean={mean_val:.2f}",
            )
        plt.legend()
        if out_path:
            plt.savefig(f"{out_path}_ram.png", dpi=150, bbox_inches="tight")

    # Plot Net IO
    net_io = data.get("net_io_mbytes_sec", {})
    if net_io:
        plt.figure()
        for key, label in [("sent", "Sent"), ("recv", "Received")]:
            if key in net_io:
                y_raw = net_io[key]
                y_smooth = _rolling_mean(y_raw, smooth)
                if smooth > 1:
                    plt.plot(
                        times,
                        y_raw,
                        alpha=0.25,
                        linewidth=1,
                        label=f"{label} raw",
                    )
                    plt.plot(
                        times,
                        y_smooth,
                        linewidth=1.8,
                        label=f"{label} rolling_mean(w={smooth})",
                    )
                else:
                    plt.plot(times, y_raw, linewidth=1.2, label=label)
                if show_mean and y_raw:
                    mean_val = sum(y_raw) / len(y_raw)
                    plt.axhline(
                        mean_val,
                        color="k",
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.5,
                        label=f"{label} mean={mean_val:.2f}",
                    )
        plt.title("Network IO Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("MB/s")
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.legend()
        if out_path:
            plt.savefig(f"{out_path}_net_io.png", dpi=150, bbox_inches="tight")

    # Plot Disk IO
    disk_io = data.get("disk_io_mbytes_sec", {})
    if disk_io:
        plt.figure()
        for key, label in [("read", "Read"), ("write", "Write")]:
            if key in disk_io:
                y_raw = disk_io[key]
                y_smooth = _rolling_mean(y_raw, smooth)
                plt.plot(times, y_raw, linewidth=1.2, label=label)
                if show_mean and y_raw:
                    mean_val = sum(y_raw) / len(y_raw)
                    plt.axhline(
                        mean_val,
                        color="k",
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.5,
                        label=f"{label} mean={mean_val:.2f}",
                    )
        plt.title("Disk IO Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("MB/s")
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.legend()
        if out_path:
            plt.savefig(f"{out_path}_disk_io.png", dpi=150, bbox_inches="tight")


def main() -> None:
    """
    Main function to parse arguments, load monitoring data, and generate plots.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--show", action="store_true", help="Show plot window")
    ap.add_argument("--out", type=str, default=None, help="Save plot to file (prefix)")
    ap.add_argument(
        "--mean",
        action="store_true",
        help="Overlay global mean line for each series",
    )
    ap.add_argument(
        "--smooth",
        type=int,
        default=10,
        metavar="N",
        help="Rolling mean window (in samples) to overlay (raw shown faint). 0/1 disables.",
    )
    args = ap.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        data = json.load(f)

    if "queue_monitor" in data:
        plot_queues(data["queue_monitor"], args.out, args.mean, args.smooth)

    if "system_monitor" in data:
        plot_system(data["system_monitor"], args.out, args.mean, args.smooth)

    if args.show or not args.out:
        plt.show()


if __name__ == "__main__":
    main()
