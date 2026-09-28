import uvicorn


def main() -> None:
    uvicorn.run(
        "minillm.server:app",
        host="127.0.0.1",
        port=8123,
    )


if __name__ == "__main__":
    main()