import re
import time
import argparse
from typing import List, Tuple
from pythonosc.udp_client import SimpleUDPClient
import numpy as np

# default replay rate
DEFAULT_FPS: float = 14.0

def replay(
        client: SimpleUDPClient,
        fps: float,
) -> None:
    period = 1.0 / fps
    try:
        while True:
            t0 = time.time()
            tilts = np.random.random(16 * 2)
            client.send_message("/light/offset", tilts)
            dt = time.time() - t0
            sleep = period - dt
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("🛑 Test stopped by user.")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test /light/tilts OSC messages, looping at a fixed FPS."
    )
    parser.add_argument(
        "--send-address",
        default="127.0.0.1",
        help="OSC target IP/address (default: %(default)s)",
    )
    parser.add_argument(
        "--send-port",
        type=int,
        default=57120,
        help="OSC target port (default: %(default)s)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Replay rate in frames per second (default: %(default)s)",
    )

    args = parser.parse_args()
    client = SimpleUDPClient(args.send_address, args.send_port)

    print(
        f"▶️  Sending /light/offset "
        f"→ {args.send_address}:{args.send_port} @ {args.fps} FPS"
    )
    replay(client, args.fps)

if __name__ == "__main__":
    main()
