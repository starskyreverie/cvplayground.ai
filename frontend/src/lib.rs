use wasm_bindgen::prelude::*;

// This function is marked as `pub` to make it callable from outside the Rust module,
// and `#[wasm_bindgen]` allows it to be called from JavaScript.
#[wasm_bindgen]
pub fn greet() {
    alert("Hello, wasm-pack!");
}

// The `alert` function needs to be defined in this scope. It's a function that
// will be provided by the JavaScript environment where the WASM module runs.
#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}
