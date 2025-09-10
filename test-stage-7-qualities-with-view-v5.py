from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

from pythonosc.udp_client import SimpleUDPClient
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TWO_PI: float = 2.0 * math.pi


@dataclass(frozen=True)
class Config:
    # OSC
    ip: str = "127.0.0.1"
    port: int = 10000

    # Lines / motion
    lines: int = 16
    spin_period: float = 2.0       # seconds per full spin of each line
    orbit_period: float = 5.0      # seconds per full orbit of centers

    # Pulse (radial in/out, synchronized)
    pulse_period: float = 1.0      # seconds per full pulse in/out
    pulse_amp: float = 0.3         # fractional amplitude (0..1) of base radius

    # Length modulation (60..160 by default)
    length_min: float = 10.0
    length_max: float = 180.0
    length_period: float = 2.0     # seconds per full length cycle

    # Opacity modulation (0.2..1.0 default, can be constant if period <= 0)
    opacity_min: float = 0.0
    opacity_max: float = 1.0
    opacity_period: float = 5.0    # 0 => constant (use opacity_min==opacity_max)

    # Geometry
    radius: float = 350.0 / 2      # base orbit radius (UU)
    base_length: float = 120.0     # legacy default if you disable modulation
    center_x: float = 0.0
    center_y: float = 0.0

    # Timing
    fps: float = 60.0              # send & draw rate

    # Visual
    show_centers: bool = True
    axis_margin: float = 250.0     # padding around orbit radius in the view


@dataclass
class LaserLine:
    """UE-like struct: X, Y = center; Length; Rotation (degrees 0..360); Opacity 0..1."""
    x: float
    y: float
    length: float
    rotation_deg: float
    opacity: float

    def endpoints(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        half = 0.5 * self.length
        rad = math.radians(self.rotation_deg)
        dx, dy = math.cos(rad), math.sin(rad)
        x1 = self.x - half * dx
        y1 = self.y - half * dy
        x2 = self.x + half * dx
        y2 = self.y + half * dy
        return (x1, y1), (x2, y2)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Send /laser/line (N*5 floats: x y length rot_deg opacity) + live viz.")
    p.add_argument("--ip", default=Config.ip, help="OSC target IP")
    p.add_argument("--port", type=int, default=Config.port, help="OSC target port")
    p.add_argument("--lines", type=int, default=Config.lines, help="number of lines (default 16)")
    p.add_argument("--spin-period", type=float, default=Config.spin_period, help="seconds per full line spin")
    p.add_argument("--orbit-period", type=float, default=Config.orbit_period, help="seconds per full orbit of centers")
    p.add_argument("--pulse-period", type=float, default=Config.pulse_period, help="seconds per radial pulse")
    p.add_argument("--pulse-amp", type=float, default=Config.pulse_amp, help="fractional radius amplitude (0..1)")
    p.add_argument("--length-min", type=float, default=Config.length_min, help="minimum line length")
    p.add_argument("--length-max", type=float, default=Config.length_max, help="maximum line length")
    p.add_argument("--length-period", type=float, default=Config.length_period, help="seconds per length cycle")
    p.add_argument("--opacity-min", type=float, default=Config.opacity_min, help="minimum opacity (0..1)")
    p.add_argument("--opacity-max", type=float, default=Config.opacity_max, help="maximum opacity (0..1)")
    p.add_argument("--opacity-period", type=float, default=Config.opacity_period, help="seconds per opacity cycle (0=constant)")
    p.add_argument("--radius", type=float, default=Config.radius, help="base orbit radius (UU)")
    p.add_argument("--length", type=float, default=Config.base_length, help="legacy static length (unused if modulating)")
    p.add_argument("--fps", type=float, default=Config.fps, help="send/draw rate")
    p.add_argument("--center-x", type=float, default=Config.center_x)
    p.add_argument("--center-y", type=float, default=Config.center_y)
    p.add_argument("--no-centers", action="store_true", help="hide center dots")
    ns = p.parse_args()

    # Clamp pulse amp to avoid negative radius
    pulse_amp = max(0.0, min(0.99, ns.pulse_amp))

    # Sanity for length range
    lmin = float(min(ns.length_min, ns.length_max))
    lmax = float(max(ns.length_min, ns.length_max))

    # Sanity for opacity range
    omin = max(0.0, min(1.0, float(min(ns.opacity_min, ns.opacity_max))))
    omax = max(0.0, min(1.0, float(max(ns.opacity_min, ns.opacity_max))))
    oper = max(0.0, float(ns.opacity_period))

    return Config(
        ip=ns.ip,
        port=ns.port,
        lines=ns.lines,
        spin_period=ns.spin_period,
        orbit_period=ns.orbit_period,
        pulse_period=ns.pulse_period,
        pulse_amp=pulse_amp,
        length_min=lmin,
        length_max=lmax,
        length_period=ns.length_period,
        opacity_min=omin,
        opacity_max=omax,
        opacity_period=oper,
        radius=ns.radius,
        base_length=ns.length,
        fps=ns.fps,
        center_x=ns.center_x,
        center_y=ns.center_y,
        show_centers=not ns.no_centers,
    )


def make_phases(n: int) -> Tuple[List[float], List[float]]:
    spin_phase = [i * TWO_PI / n for i in range(n)]
    orbit_phase = [i * TWO_PI / n for i in range(n)]
    return spin_phase, orbit_phase


def length_at_time(t: float, cfg: Config) -> float:
    """Sinusoid between [length_min, length_max] over length_period seconds."""
    if cfg.length_period <= 0:
        return cfg.base_length
    mid = 0.5 * (cfg.length_min + cfg.length_max)
    amp = 0.5 * (cfg.length_max - cfg.length_min)
    return mid + amp * math.sin(TWO_PI * (t / cfg.length_period))


def opacity_at_time(t: float, cfg: Config) -> float:
    """Sinusoid between [opacity_min, opacity_max] over opacity_period seconds (or constant if period<=0)."""
    if cfg.opacity_period <= 0:
        # constant: use the midpoint of min/max (both 1.0 by default)
        return 0.5 * (cfg.opacity_min + cfg.opacity_max)
    mid = 0.5 * (cfg.opacity_min + cfg.opacity_max)
    amp = 0.5 * (cfg.opacity_max - cfg.opacity_min)
    return max(0.0, min(1.0, mid + amp * math.sin(TWO_PI * (t / cfg.opacity_period))))


def build_lines(t: float, cfg: Config, spin_phase: List[float], orbit_phase: List[float]) -> List[LaserLine]:
    """Compute line structs for the current time, with synchronized radial pulse + length/opacity modulation."""
    spin_ang  = TWO_PI * (t / cfg.spin_period)
    orbit_ang = TWO_PI * (t / cfg.orbit_period)

    # Pulse radius around base: R(t) = R0 * (1 + amp*sin(2π t / T))
    pulse = math.sin(TWO_PI * (t / cfg.pulse_period))
    radius_now = max(0.0, cfg.radius * (1.0 + cfg.pulse_amp * pulse))

    len_now = max(0.0, length_at_time(t, cfg))
    op_now  = opacity_at_time(t, cfg)

    out: List[LaserLine] = []
    for i in range(cfg.lines):
        a = orbit_ang + orbit_phase[i]
        x = cfg.center_x + radius_now * math.cos(a)
        y = cfg.center_y + radius_now * math.sin(a)
        r = (spin_ang + spin_phase[i]) % TWO_PI
        rot_deg = (r * 180.0 / math.pi) % 360.0
        out.append(LaserLine(x=x, y=y, length=len_now, rotation_deg=rot_deg, opacity=op_now))
    return out


def flatten_payload(lines: List[LaserLine]) -> List[float]:
    """Flatten to [x y length rot_deg opacity01] * N for OSC."""
    payload: List[float] = []
    for ln in lines:
        payload.extend([float(ln.x), float(ln.y), float(ln.length), float(ln.rotation_deg), float(ln.opacity)])
    return payload


def format_message_preview(payload: List[float], max_chars: int = 220) -> str:
    head = "/laser/line "
    nums = " ".join(f"{v:.1f}" for v in payload)
    s = head + nums
    if len(s) > max_chars:
        # Approximate: 5 chars per number incl. space when formatted as .1f
        omitted = max(0, len(payload) - int((max_chars - len(head)) / 5))
        s = head + nums[: max_chars - len(head) - 15] + f"... (+{omitted} vals)"
    return s


def setup_plot(cfg: Config) -> Tuple[plt.Figure, plt.Axes, List[Line2D], Optional[Line2D], plt.Text, plt.Text]:
    """Create a view: lines + optional centers + header/footer text."""
    plt.ion()
    fig, ax = plt.subplots(num="Laser Debug")
    plt.subplots_adjust(top=0.88, bottom=0.12)

    ax.set_aspect("equal", adjustable="box")
    lim = cfg.radius * (1.0 + cfg.pulse_amp) + cfg.axis_margin
    ax.set_xlim(cfg.center_x - lim, cfg.center_x + lim)
    ax.set_ylim(cfg.center_y - lim, cfg.center_y + lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Laser Lines: orbit=5s, spin=2s, pulse=radial in/out, length=sin, opacity=sin/const")

    ax.grid(True, alpha=0.2)

    values_per_line = 5  # x, y, length, rotation_deg, opacity01
    header_text = fig.text(
        0.01, 0.97,
        f"OSC → {cfg.ip}:{cfg.port}  addr=/laser/line  values={cfg.lines*values_per_line}  "
        f"[radius={cfg.radius:.0f}, pulse={cfg.pulse_amp:.2f}@{cfg.pulse_period:.1f}s]  "
        f"[len={cfg.length_min:.0f}..{cfg.length_max:.0f}@{cfg.length_period:.1f}s]  "
        f"[opacity={cfg.opacity_min:.2f}..{cfg.opacity_max:.2f}@{cfg.opacity_period:.1f}s]",
        ha="left", va="top", fontsize=9, family="monospace"
    )

    footer_text = fig.text(
        0.01, 0.03,
        "", ha="left", va="bottom", fontsize=8, family="monospace"
    )

    line_artists: List[Line2D] = []
    for _ in range(cfg.lines):
        (ln,) = ax.plot([0, 0], [0, 0], lw=2, alpha=1.0)
        line_artists.append(ln)

    centers_artist: Optional[Line2D] = None
    if cfg.show_centers:
        (centers_artist,) = ax.plot([], [], "o", ms=4, alpha=0.7)

    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, line_artists, centers_artist, header_text, footer_text


def update_plot(lines: List[LaserLine], line_artists: List[Line2D], centers_artist: Optional[Line2D]) -> None:
    for ln, artist in zip(lines, line_artists):
        (x1, y1), (x2, y2) = ln.endpoints()
        artist.set_data([x1, x2], [y1, y2])
        artist.set_alpha(max(0.0, min(1.0, ln.opacity)))

    if centers_artist is not None:
        centers_artist.set_data([ln.x for ln in lines], [ln.y for ln in lines])


def main() -> None:
    cfg = parse_args()
    client = SimpleUDPClient(cfg.ip, cfg.port)

    spin_phase, orbit_phase = make_phases(cfg.lines)
    fig, ax, line_artists, centers_artist, header_text, footer_text = setup_plot(cfg)

    t0 = time.perf_counter()
    dt = 1.0 / max(1.0, cfg.fps)
    next_tick = t0

    try:
        while plt.fignum_exists(fig.number):
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)
                continue

            t = now - t0
            lines = build_lines(t, cfg, spin_phase, orbit_phase)

            # OSC payload + send: [x y length rot_deg opacity]*N
            payload = flatten_payload(lines)
            client.send_message("/laser/line", payload)

            # Update debug view
            update_plot(lines, line_artists, centers_artist)
            footer_text.set_text(format_message_preview(payload))
            fig.canvas.draw()
            fig.canvas.flush_events()

            next_tick += dt
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.close(fig)


if __name__ == "__main__":
    main()
