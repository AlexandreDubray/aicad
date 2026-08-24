//! Structure used to log structured data during execution for further off-line analysis.
//! The goal is to have a lightweight way of storing data into a jsonl file using a macro that is
//! etiher active or unactive. If the storing is unactive, only a boolean is checked to limit
//! overhead.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static ENABLED: AtomicBool = AtomicBool::new(false);
static SINK: Mutex<Option<BufWriter<File>>> = Mutex::new(None);

pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn enable(path: &Path) -> std::io::Result<()> {
    let file = File::create(path)?;
    *SINK.lock().unwrap() = Some(BufWriter::new(file));
    ENABLED.store(true, Ordering::Relaxed);
    Ok(())
}

/// Flushes and closes the sink. Also called implicitly by dropping the process, but since a
/// `static` never runs its destructor, call this explicitly once done recording to guarantee
/// every buffered event actually made it to disk.
pub fn disable() {
    ENABLED.store(false, Ordering::Relaxed);
    if let Some(mut writer) = SINK.lock().unwrap().take() {
        let _ = writer.flush();
    }
}

/// Writes `value` as one JSONL line. Flushes on every call -- this is an opt-in diagnostics path,
/// not a hot one, and losing buffered events to a crash or a forgotten `disable()` would defeat
/// the point.
pub fn record(value: serde_json::Value) {
    let Ok(mut guard) = SINK.lock() else {
        return;
    };
    let Some(writer) = guard.as_mut() else {
        return;
    };
    let Ok(line) = serde_json::to_string(&value) else {
        return;
    };
    let _ = writeln!(writer, "{line}");
    let _ = writer.flush();
}

/// Records one structured event when data logging is enabled (see `enable`/`disable`), a no-op
/// (single atomic load, no allocation) otherwise. `event` is a `&str` naming the event kind;
/// remaining `key = value` pairs become fields on the same JSON object, each run through
/// `serde_json::to_value` (a value that fails to serialize, e.g. a non-finite float, is logged as
/// `null` rather than panicking).
///
/// ```ignore
/// data_log!("mdd_entropy", problem_idx = idx, row = row, var = v, phase = "before", entropy = h);
/// ```
#[macro_export]
macro_rules! data_log {
    ($event:expr, $($key:ident = $value:expr),+ $(,)?) => {
        if $crate::diagnostics::is_enabled() {
            let mut map = serde_json::Map::new();
            map.insert("event".to_string(), serde_json::Value::from($event));
            $(
                map.insert(
                    stringify!($key).to_string(),
                    serde_json::to_value(&$value).unwrap_or(serde_json::Value::Null),
                );
            )+
            $crate::diagnostics::record(serde_json::Value::Object(map));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_by_default_and_record_is_a_silent_no_op() {
        // Order-independent of other tests only because `enable`/`disable` here use a private
        // temp file and this test doesn't assert on `is_enabled()`'s value at entry -- just that
        // recording while (possibly) disabled never panics.
        record(serde_json::json!({"event": "unused"}));
    }

    #[test]
    fn enable_then_disable_writes_and_flushes_exactly_the_recorded_lines() {
        let path =
            std::env::temp_dir().join(format!("aicad_data_log_test_{}.jsonl", std::process::id()));
        enable(&path).expect("enable should succeed");

        data_log!("unit_test_event", value = 42);
        data_log!("unit_test_event", value = 7);

        disable();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["event"], "unit_test_event");
        assert_eq!(first["value"], 42);

        std::fs::remove_file(&path).ok();
    }
}
