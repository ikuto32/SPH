use std::env;
use std::path::PathBuf;
use std::process::Command;

fn nvcc_in_path() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn nvcc_in_cuda_home() -> bool {
    let home = match env::var_os("CUDA_HOME").or_else(|| env::var_os("CUDA_PATH")) {
        Some(val) => val,
        None => return false,
    };
    let mut candidate = PathBuf::from(home);
    candidate.push("bin");
    candidate.push("nvcc");
    Command::new(candidate)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(cuda_available)");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let has_nvcc = nvcc_in_path() || nvcc_in_cuda_home();
    if has_nvcc {
        println!("cargo:rustc-cfg=cuda_available");
    } else {
        println!("cargo:warning=CUDA feature requested, but nvcc toolchain was not detected. Falling back to CPU-only stubs.");
    }
}
