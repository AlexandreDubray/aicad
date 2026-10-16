//! A switchable sink for structured events, written as JSON lines for offline analysis.
//!
//! # Purpose
//!
//! Timings and statistics of a long run (training epochs, MDD compilation, ...) are recorded
//! with [`data_log!`], one JSON object per line, and read back afterwards (`pd.read_json(path,
//! lines=True)`). This is separate from the `log` crate, which carries text for a human reading a
//! terminal. While no sink is enabled a call site costs one relaxed atomic load: its arguments
//! are not evaluated and nothing is allocated, so call sites can stay in hot-ish code.
//!
//! The contract:
//!
//! - There is one process-wide sink. [`enable`] opens it (truncating the file) and fails if one
//!   is already open, so a log is never silently replaced; [`disable`] flushes and closes it.
//! - Every event is flushed as it is recorded. A crash loses at most the line being written, and
//!   a forgotten [`disable`] loses nothing (a `static` never runs its destructor).
//! - A diagnostics failure never aborts the run: if writing fails (disk full, file removed), one
//!   warning is logged and the sink is closed, so recording stops instead of failing repeatedly.
//! - A value that cannot be serialised (a non-finite float, say) is recorded as `null`.
//!
//! # Events
//!
//! Each line has an `"event"` field naming its kind, plus the fields given at the call site.
//! The scripts in `cpaior27` read these by name, so changing one is a change to that interface.
//!
//! - `train_epoch` (`learning::train`): `epoch`, the phase timings `t_batch`, `t_forward`,
//!   `t_monitor`, `t_loss`, `t_backward`, `epoch_rt_s` and `epoch_loss`.
//! - `valid_epoch` (`learning::train`): `epoch`, `t_batch`, `t_forward`, `t_score`, `total_s`
//!   and `score`.
//! - `mdd_dataset_build` (`pyaicad::learn`): `split` (`"train"` or `"valid"`), `num_problems`
//!   and `elapsed_s`.
//!
//! Nothing is recorded unless the caller opens a sink: training, solving and compiling never
//! start one themselves. A script that wants the timings of a training run calls
//! `pyaicad.diagnostics.enable_data_log` before training and `disable_data_log` after it,
//! before running inference, so the file holds exactly what it asked for.
//!
//! # Design notes
//!
//! - **Fields are named after their identifier.** `data_log!("e", n = 3)` writes `"n": 3`, so a
//!   call site reads like the line it produces. A field named `event`, or given twice, would
//!   overwrite another one; the macro panics with the field's name instead (only when a sink is
//!   enabled, so the mistake shows up the first time a log is recorded).
//! - **One global sink.** Threading a handle through the training code would touch every
//!   signature for a facility that is off by default; the price is the single-sink rule above.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

struct Sink {
    writer: BufWriter<File>,
    path: PathBuf,
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static SINK: Mutex<Option<Sink>> = Mutex::new(None);

/// The sink, usable even if a thread panicked while holding it: the worst case is one truncated
/// line, and a diagnostics facility must not turn that into a second panic.
fn lock_sink() -> MutexGuard<'static, Option<Sink>> {
    SINK.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Starts recording to `path`, truncating it. Fails, without touching either file, if a sink is
/// already open (see the module documentation) or if `path` cannot be created.
pub(crate) fn enable(path: &Path) -> std::io::Result<()> {
    let mut sink = lock_sink();
    if let Some(open) = sink.as_ref() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!(
                "cannot record to {}: a data log is already being written to {}; disable it first",
                path.display(),
                open.path.display()
            ),
        ));
    }
    let file = File::create(path)?;
    *sink = Some(Sink {
        writer: BufWriter::new(file),
        path: path.to_path_buf(),
    });
    ENABLED.store(true, Ordering::Relaxed);
    Ok(())
}

/// Flushes and closes the sink; does nothing if none is open.
pub(crate) fn disable() {
    ENABLED.store(false, Ordering::Relaxed);
    if let Some(mut open) = lock_sink().take()
        && let Err(e) = open.writer.flush()
    {
        log::warn!("failed to flush the data log {}: {e}", open.path.display());
    }
}

/// Writes `value` as one line and flushes it. On a write error, logs one warning and closes the
/// sink (see the module documentation).
pub(crate) fn record(value: serde_json::Value) {
    let mut sink = lock_sink();
    let Some(open) = sink.as_mut() else {
        return;
    };
    // Serialising a `Value` cannot fail: its keys are strings and it holds no non-finite float
    // (`serde_json::to_value` already turned those into `null`).
    let line = value.to_string();
    let result = writeln!(open.writer, "{line}").and_then(|()| open.writer.flush());
    if let Err(e) = result {
        log::warn!(
            "writing the data log {} failed ({e}); recording stops",
            open.path.display()
        );
        ENABLED.store(false, Ordering::Relaxed);
        *sink = None;
    }
}

/// Records one event when a sink is enabled, and does nothing (one atomic load, arguments not
/// evaluated) otherwise. `event` names the kind; each `key = value` becomes a field of the same
/// JSON object. See the module documentation for the field rules.
///
/// ```ignore
/// data_log!("mdd_entropy", problem_idx = idx, row = row, var = v, entropy = h);
/// ```
macro_rules! data_log {
    ($event:expr, $($key:ident = $value:expr),+ $(,)?) => {
        if $crate::diagnostics::is_enabled() {
            let mut map = serde_json::Map::new();
            map.insert("event".to_string(), serde_json::Value::from($event));
            $(
                let previous = map.insert(
                    stringify!($key).to_string(),
                    serde_json::to_value(&$value).unwrap_or(serde_json::Value::Null),
                );
                assert!(
                    previous.is_none(),
                    "data_log!(\"{}\"): the field `{}` is given twice or clashes with the reserved field `event`",
                    $event,
                    stringify!($key),
                );
            )+
            $crate::diagnostics::record(serde_json::Value::Object(map));
        }
    };
}

pub(crate) use data_log;

#[cfg(test)]
mod tests {
    use super::*;

    // The sink is process-wide, so the tests of this module must not overlap.
    static SERIAL: Mutex<()> = Mutex::new(());

    /// Holds the test lock, hands out a fresh path, and closes the sink even if the test panics.
    struct Scenario {
        _lock: MutexGuard<'static, ()>,
        path: PathBuf,
    }

    impl Scenario {
        fn new(name: &str) -> Self {
            let lock = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
            disable();
            let path = std::env::temp_dir()
                .join(format!("aicad_diag_{}_{name}.jsonl", std::process::id()));
            Scenario { _lock: lock, path }
        }

        fn lines(&self) -> Vec<serde_json::Value> {
            std::fs::read_to_string(&self.path)
                .unwrap()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect()
        }
    }

    impl Drop for Scenario {
        fn drop(&mut self) {
            disable();
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn events_are_written_one_per_line_with_their_fields() {
        let s = Scenario::new("round_trip");
        enable(&s.path).unwrap();
        data_log!("first", epoch = 3usize, loss = 0.5f64, split = "train");
        data_log!("second", values = vec![1, 2, 3]);
        disable();
        let lines = s.lines();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["event"], "first");
        assert_eq!(lines[0]["epoch"], 3);
        assert_eq!(lines[0]["loss"], 0.5);
        assert_eq!(lines[0]["split"], "train");
        assert_eq!(lines[1]["values"], serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn non_finite_floats_are_recorded_as_null() {
        let s = Scenario::new("non_finite");
        enable(&s.path).unwrap();
        data_log!("loss", value = f64::NAN);
        disable();
        assert!(s.lines()[0]["value"].is_null());
    }

    #[test]
    fn a_disabled_log_does_not_evaluate_its_arguments() {
        let _s = Scenario::new("disabled");
        let mut evaluated = false;
        data_log!(
            "never",
            flag = {
                evaluated = true;
                1
            }
        );
        assert!(!evaluated);
    }

    #[test]
    fn enabling_twice_fails_and_leaves_the_first_log_intact() {
        let s = Scenario::new("twice");
        let other = s.path.with_extension("other");
        enable(&s.path).unwrap();
        let err = enable(&other).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert!(err.to_string().contains("already being written"), "{err}");
        assert!(!other.exists(), "the refused log must not be created");
        data_log!("still_first", n = 1);
        disable();
        assert_eq!(s.lines().len(), 1);
        // Closing makes it possible to open another one.
        enable(&other).unwrap();
        disable();
        let _ = std::fs::remove_file(&other);
    }

    #[test]
    fn enabling_an_unwritable_path_fails_and_stays_disabled() {
        let s = Scenario::new("unwritable");
        let missing_dir = s.path.join("no").join("such").join("dir.jsonl");
        assert!(enable(&missing_dir).is_err());
        assert!(!is_enabled());
    }

    #[test]
    #[should_panic(expected = "clashes with the reserved field `event`")]
    fn a_field_named_event_is_refused() {
        let s = Scenario::new("reserved");
        enable(&s.path).unwrap();
        data_log!("kind", event = 1);
    }

    #[test]
    #[should_panic(expected = "is given twice")]
    fn a_repeated_field_is_refused() {
        let s = Scenario::new("repeated");
        enable(&s.path).unwrap();
        data_log!("kind", n = 1, n = 2);
    }

    // /dev/full accepts the open and fails every write with "no space left on device".
    #[cfg(target_os = "linux")]
    #[test]
    fn a_write_error_closes_the_sink_instead_of_failing_the_run() {
        let _s = Scenario::new("full");
        enable(Path::new("/dev/full")).unwrap();
        data_log!("lost", n = 1);
        assert!(!is_enabled(), "the first failed write must close the sink");
        // Later events are ignored without further errors.
        data_log!("also_lost", n = 2);
    }
}
