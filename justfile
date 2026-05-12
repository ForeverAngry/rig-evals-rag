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

# Format check + clippy + tests — the same gates CI runs.
check: fmt clippy test

fmt:
    cargo fmt --all -- --check

clippy:
    cargo clippy --all-targets -- -D warnings
    cargo clippy --no-default-features --all-targets -- -D warnings

test:
    cargo test --all-features

doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
