bench:
  cargo bench

test:
  echo "Run tests.."
  cargo nextest run --release

format-check:
  echo "Run format check.."
  cargo fmt --check

format:
  cargo fmt

lint:
  echo "Run clippy.."
  cargo clippy

test-all: format-check lint test

book-build:
  mdbook build odesign-book

book-build-tar: book-build
  tar --exclude='./odesign-book/src/' -cf ./deploy.tar ./captain-definition ./odesign-book/*

book-watch:
  mdbook watch odesign-book --open

book-publish: book-build-tar
  caprover deploy -t ./deploy.tar -a odesign

run-all-examples:
  #!/bin/sh
  for dir in ./odesign-examples/examples/*/
  do
    dir=${dir%*/}
    example=${dir##*/}
    cargo run --example $example --release
  done

