# ---- Build stage ----
FROM ubuntu:22.04 AS build

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN mkdir -p build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && cmake --build . --target NeuroCoreXWeb -- -j$(nproc)

# ---- Runtime stage ----
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what's needed to run: the compiled binary, the web UI,
# and an empty data/ directory for persistence within the container.
COPY --from=build /app/build/NeuroCoreXWeb /app/NeuroCoreXWeb
COPY --from=build /app/web /app/web
RUN mkdir -p /app/data/snapshots

# Render provides the port to bind via the PORT environment variable.
ENV PORT=8080
EXPOSE 8080

CMD ["/app/NeuroCoreXWeb"]
