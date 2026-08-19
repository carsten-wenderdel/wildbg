use std::path::PathBuf;
use std::process::Command;

fn git_head_path() -> Option<PathBuf> {
    let output = Command::new("git")
        .args(["rev-parse", "--git-path", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let head_path = String::from_utf8(output.stdout).ok()?;
    let head_path = head_path.trim();
    if head_path.is_empty() {
        return None;
    }
    PathBuf::from(head_path).canonicalize().ok()
}

fn main() {
    if let Some(head_path) = git_head_path() {
        println!("cargo:rerun-if-changed={}", head_path.display());
    }

    let output = Command::new("git").args(["rev-parse", "HEAD"]).output();

    let git_hash = match output {
        Ok(o) if o.status.success() => String::from_utf8(o.stdout)
            .unwrap_or_default()
            .trim()
            .to_string(),
        _ => "unknown".to_string(),
    };

    println!("cargo:rustc-env=WILDBG_GIT_HASH={git_hash}");
}
