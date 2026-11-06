//! Which build of the library is running: its version, the commit it was built from, and the
//! compiler and settings that built it.
//!
//! # Meaning
//!
//! A number measured with `aicad` (a runtime, a memory footprint, a count of refinement steps)
//! only means something together with the build that produced it. The version in the source tree
//! says little: the code that runs may be an installed copy, an older build, or a different
//! commit of the same version. [`version`] reports what was compiled *into* the running code.
//!
//! # How it works
//!
//! `build.rs` records the commit, the compiler and the build settings when the crate is compiled
//! and this module exposes them as a [`BuildInfo`]:
//!
//! - `version` is the crate version of `Cargo.toml`.
//! - `commit` is the full hash of the commit the working tree was at, or `None` when it cannot
//!   be known (a source archive without `.git`, or no `git` on the machine). A crate downloaded
//!   from crates.io carries the commit it was published from.
//! - `rustc` is the output of `rustc --version`, `target` the compilation target, `profile` and
//!   `opt_level` the cargo profile and its optimisation level, and `features` the cargo features
//!   that were on.
//!
//! The build info is also written, once, when a data log or a verbose log is started: it is the
//! first line of every `data_log!` file (event `build_info`, see [`crate::diagnostics`]) and a
//! `log::info!` record, so the output of a run says which build produced it.
//!
//! # Design notes
//!
//! - **The working tree is not checked for uncommitted changes.** Performance numbers are only
//!   comparable between commits, so benchmarks are to be launched from a clean copy of a commit;
//!   it is the responsibility of whoever launches them. A modified tree reports the commit it was
//!   based on.
//! - **The commit is refreshed when `HEAD` moves**, so an incremental build after `git commit`
//!   or `git checkout` reports the new commit. Editing files does not rebuild `build.rs`.
//! - **No build dependency.** `build.rs` runs `git` and `rustc` directly rather than pulling a
//!   crate for it; every field degrades to empty or `None` instead of failing the build.
//!
//! # Example
//!
//! ```
//! let info = aicad::version();
//! assert_eq!(info.version, env!("CARGO_PKG_VERSION"));
//! println!("{info}");
//! ```

#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

use std::fmt;

include!(concat!(env!("OUT_DIR"), "/build_info.rs"));

/// The build of the library that is running. See the [module documentation](self).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildInfo {
    /// The crate version.
    pub version: &'static str,
    /// The full hash of the commit the library was built from, if known.
    pub commit: Option<&'static str>,
    /// The compiler, as printed by `rustc --version`. Empty if it could not be run.
    pub rustc: &'static str,
    /// The compilation target, for example `x86_64-unknown-linux-gnu`.
    pub target: &'static str,
    /// The cargo profile: `debug` or `release`.
    pub profile: &'static str,
    /// The optimisation level of the profile: `0`, `1`, `2`, `3`, `s` or `z`.
    pub opt_level: &'static str,
    /// The cargo features that were on, sorted.
    pub features: &'static [&'static str],
}

/// The build of the library that is running.
pub fn version() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION"),
        commit: COMMIT,
        rustc: RUSTC,
        target: TARGET,
        profile: PROFILE,
        opt_level: OPT_LEVEL,
        features: FEATURES,
    }
}

impl BuildInfo {
    /// The fields as a JSON object, with the features as a list and an unknown commit as `null`.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "version": self.version,
            "commit": self.commit,
            "rustc": self.rustc,
            "target": self.target,
            "profile": self.profile,
            "opt_level": self.opt_level,
            "features": self.features,
        })
    }
}

impl fmt::Display for BuildInfo {
    /// `1.0.0 (commit 0123abc, rustc 1.95.0 ..., release opt-level 3, x86_64-..., features: ...)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let commit = self.commit.map_or("unknown", |c| &c[..c.len().min(12)]);
        let rustc = if self.rustc.is_empty() {
            "unknown"
        } else {
            self.rustc
        };
        write!(
            f,
            "{} (commit {commit}, {rustc}, {} opt-level {}, {}, features: [{}])",
            self.version,
            self.profile,
            self.opt_level,
            self.target,
            self.features.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_version_is_the_one_of_the_manifest() {
        assert_eq!(version().version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn the_commit_is_a_full_hash_or_unknown() {
        if let Some(commit) = version().commit {
            assert_eq!(commit.len(), 40, "{commit}");
            assert!(commit.chars().all(|c| c.is_ascii_hexdigit()), "{commit}");
        }
    }

    #[test]
    fn the_compiler_and_the_settings_are_recorded() {
        let info = version();
        assert!(info.rustc.starts_with("rustc "), "{}", info.rustc);
        assert!(!info.target.is_empty());
        assert!(
            ["debug", "release"].contains(&info.profile),
            "{}",
            info.profile
        );
        assert!(!info.opt_level.is_empty());
    }

    #[test]
    fn the_features_are_sorted_and_match_the_enabled_ones() {
        let info = version();
        assert!(info.features.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(
            info.features.contains(&"learning"),
            cfg!(feature = "learning")
        );
        assert_eq!(info.features.contains(&"python"), cfg!(feature = "python"));
    }

    #[test]
    fn the_json_and_the_display_carry_every_field() {
        let info = version();
        let json = info.to_json();
        assert_eq!(json["version"], info.version);
        assert_eq!(json["rustc"], info.rustc);
        assert_eq!(
            json["features"].as_array().unwrap().len(),
            info.features.len()
        );
        assert_eq!(json["commit"].is_null(), info.commit.is_none());
        let text = info.to_string();
        assert!(
            text.starts_with(info.version) && text.contains(info.target),
            "{text}"
        );
    }
}
