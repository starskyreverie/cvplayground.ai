#!/bin/sh
# Build the project initially.
wasm-pack build

# Start the live-server to serve the pkg directory.
live-server pkg/ &

# Watch for changes in the src directory, rebuild the project when changes are detected.
cargo watch -w src/ -s "wasm-pack build"
