"""Start the server on your Wi-Fi network and print the address to open on your phone."""
import os
import socket

import uvicorn


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))  # no packets are sent; this just picks the outbound interface
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"\n  On this computer:  http://localhost:{port}")
    print(f"  On your phone:     http://{lan_ip()}:{port}   (same Wi-Fi network)\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
