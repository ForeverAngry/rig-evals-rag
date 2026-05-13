# rig-evals-rag task runner.
#
# Install just: https://github.com/casey/just
#   brew install just
#
# Run `just` with no args to see the recipe list.

# Show all recipes.
default:
    @just --list

# Build all targets with default features.
build:
    cargo build --all-targets

# Format check + clippy + tests across CI and local feature gates.
check: fmt clippy test msrv doc

fmt:
    cargo fmt --all -- --check

clippy:
    cargo clippy --all-targets -- -D warnings
    cargo clippy --no-default-features --all-targets -- -D warnings
    cargo clippy --no-default-features --features ragas --all-targets -- -D warnings
    cargo clippy --all-features --all-targets -- -D warnings

test:
    cargo test --all-features --all-targets
    cargo test --no-default-features --features ragas

msrv:
    cargo +1.89 build --all-targets

doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
