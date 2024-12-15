build like so:

wasm-pack build --out-dir ../simulator --scope tpaw

// You have to remove the extern "C" from header first.
~/.cargo/bin/bindgen src/cuda/cuda.h -o src/bindings.rs