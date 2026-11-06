//! Records which build of the library this is: the commit, the compiler and the profile. The
//! result is read by `src/build_info.rs`; see its documentation for what it is used for.
//!
//! Everything here is best effort. A missing `git`, a source tree outside a repository or an
//! unreadable compiler produces an empty field, never a build failure.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Runs `program args` in `dir` and returns its trimmed standard output if it succeeded.
fn run(program: &str, args: &[&str], dir: &Path) -> Option<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (!text.is_empty()).then_some(text)
}

/// The commit recorded by `cargo package` in a published crate, which has no `.git`.
fn packaged_commit(manifest: &Path) -> Option<String> {
    let text = fs::read_to_string(manifest.join(".cargo_vcs_info.json")).ok()?;
    let after_key = text.split("\"sha1\"").nth(1)?;
    let value = after_key.split('"').nth(1)?;
    (value.len() == 40 && value.chars().all(|c| c.is_ascii_hexdigit())).then(|| value.to_string())
}

/// The commit of the working tree, if `manifest` is tracked by a git repository. The check on
/// `Cargo.toml` keeps a build from the cargo registry from reporting the commit of some
/// unrelated repository that happens to contain the registry (a home directory under git).
fn git_commit(manifest: &Path) -> Option<String> {
    run(
        "git",
        &["ls-files", "--error-unmatch", "Cargo.toml"],
        manifest,
    )?;
    let commit = run("git", &["rev-parse", "HEAD"], manifest)?;
    // A new commit moves HEAD, or the branch it points to, or the packed refs.
    let git_dir = run(
        "git",
        &["rev-parse", "--path-format=absolute", "--git-dir"],
        manifest,
    )
    .map(PathBuf::from);
    let common_dir = run(
        "git",
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
        manifest,
    )
    .map(PathBuf::from);
    if let Some(git_dir) = git_dir {
        let head = git_dir.join("HEAD");
        println!("cargo:rerun-if-changed={}", head.display());
        if let (Ok(text), Some(common)) = (fs::read_to_string(&head), common_dir.as_ref())
            && let Some(reference) = text.strip_prefix("ref:")
        {
            let branch = common.join(reference.trim());
            if branch.exists() {
                println!("cargo:rerun-if-changed={}", branch.display());
            }
        }
    }
    if let Some(common) = common_dir {
        let packed = common.join("packed-refs");
        if packed.exists() {
            println!("cargo:rerun-if-changed={}", packed.display());
        }
    }
    Some(commit)
}

/// The first line of `rustc --version`, for the compiler that builds the crate.
fn rustc_version() -> Option<String> {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let text = run(&rustc, &["--version"], Path::new("."))?;
    text.lines().next().map(str::to_string)
}

/// The cargo features that are on (but `default`, which only names a set), sorted, read from the `CARGO_FEATURE_*` variables.
fn features() -> Vec<String> {
    let mut list: Vec<String> = env::vars()
        .filter_map(|(key, _)| key.strip_prefix("CARGO_FEATURE_").map(str::to_string))
        .map(|name| name.to_lowercase().replace('_', "-"))
        .filter(|name| name != "default")
        .collect();
    list.sort();
    list
}

/// A string as a Rust literal.
fn literal(text: &str) -> String {
    format!("{text:?}")
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let commit = packaged_commit(&manifest).or_else(|| git_commit(&manifest));
    let var = |name: &str| env::var(name).unwrap_or_default();
    let features: Vec<String> = features().iter().map(|f| literal(f)).collect();
    let source = format!(
        "/// The commit the crate was built from, if known.\n\
         pub(crate) const COMMIT: Option<&str> = {};\n\
         /// The output of `rustc --version`.\n\
         pub(crate) const RUSTC: &str = {};\n\
         /// The compilation target.\n\
         pub(crate) const TARGET: &str = {};\n\
         /// The cargo profile.\n\
         pub(crate) const PROFILE: &str = {};\n\
         /// The optimisation level of the profile.\n\
         pub(crate) const OPT_LEVEL: &str = {};\n\
         /// The cargo features that were on, sorted.\n\
         pub(crate) const FEATURES: &[&str] = &[{}];\n",
        match &commit {
            Some(c) => format!("Some({})", literal(c)),
            None => "None".to_string(),
        },
        literal(&rustc_version().unwrap_or_default()),
        literal(&var("TARGET")),
        literal(&var("PROFILE")),
        literal(&var("OPT_LEVEL")),
        features.join(", "),
    );
    let out = PathBuf::from(env::var("OUT_DIR").expect("set by cargo")).join("build_info.rs");
    fs::write(&out, source).unwrap_or_else(|e| panic!("cannot write {}: {e}", out.display()));
}
