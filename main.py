import re
import time
import argparse
from typing import List, Tuple
from pythonosc.udp_client import SimpleUDPClient

# default replay rate
DEFAULT_FPS: float = 14.0

def parse_log_line(line: str) -> Tuple[str, List[float]]:
    """
    Extract the OSC address and float payload from one log line.
    Example line:
      RECEIVE | ... ADDRESS(/mocap/joint/lin_vel) FLOAT(0.01) FLOAT(-0.02) ...
    """
    # pull out ADDRESS(...)
    addr_match = re.search(r"ADDRESS\(([^\)]+)\)", line)
    if not addr_match:
        raise ValueError(f"Cannot find ADDRESS in line: {line!r}")
    osc_addr = addr_match.group(1)

    # pull all floats
    floats = [float(x) for x in re.findall(r"FLOAT\((-?[0-9\.eE+-]+)\)", line)]
    return osc_addr, floats

def load_entries(log_file: str) -> List[Tuple[str, List[float]]]:
    """
    Read the log file and return a list of (address, payload) tuples.
    Skips any non-RECEIVE lines.
    """
    entries: List[Tuple[str, List[float]]] = []
    with open(log_file, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("RECEIVE"):
                continue
            try:
                addr, payload = parse_log_line(line)
                entries.append((addr, payload))
            except ValueError:
                # skip malformed
                continue
    if not entries:
        raise RuntimeError(f"No valid RECEIVE entries found in {log_file}")
    return entries

def replay(
        entries: List[Tuple[str, List[float]]],
        client: SimpleUDPClient,
        fps: float,
) -> None:
    period = 1.0 / fps
    try:
        while True:
            for addr, payload in entries:
                t0 = time.time()
                client.send_message(addr, payload)
                dt = time.time() - t0
                sleep = period - dt
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        print("🛑 Replay stopped by user.")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay OSC mocap messages from a log file, looping at a fixed FPS."
    )
    parser.add_argument(
        "--log-file", "-l",
        default="log.txt",
        help="Path to your log file (default: %(default)s)",
    )
    parser.add_argument(
        "--send-address",
        default="127.0.0.1",
        help="OSC target IP/address (default: %(default)s)",
    )
    parser.add_argument(
        "--send-port",
        type=int,
        default=9000,
        help="OSC target port (default: %(default)s)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Replay rate in frames per second (default: %(default)s)",
    )

    args = parser.parse_args()
    entries = load_entries(args.log_file)
    client = SimpleUDPClient(args.send_address, args.send_port)

    print(
        f"▶️  Replaying {len(entries)} messages from {args.log_file} "
        f"→ {args.send_address}:{args.send_port} @ {args.fps} FPS"
    )
    replay(entries, client, args.fps)

if __name__ == "__main__":
    main()
