# --- Build Stage ---
FROM rust:1.90 AS builder
WORKDIR /src
COPY . .
RUN cargo build --release

# --- Runtime Stage ---
FROM debian:trixie-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/server .

EXPOSE 8082
ENTRYPOINT ["./server"]
CMD ["--port", "8082", "--address", "0.0.0.0"]
