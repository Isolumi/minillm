import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLLM single-GPU Responses server")
    parser.add_argument("--host", default=os.getenv("MINILLM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MINILLM_PORT", "8123")))
    args = parser.parse_args()
    uvicorn.run(
        "minillm.server:app",
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
