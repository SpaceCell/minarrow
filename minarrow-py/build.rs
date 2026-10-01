fn main() {
    pyo3_build_config::use_pyo3_cfgs();
    // The source root, published to dependent build scripts as `DEP_MINARROW_PY_ROOT`.
    println!(
        "cargo:root={}",
        std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR")
    );
}
