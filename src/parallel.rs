// SPDX-License-Identifier: GPL-3.0-or-later
//! Thread setup for the parallel parts (MCMC chains, mode-finding gradients,
//! post-processing): macOS quality-of-service classes so the scheduler keeps
//! the compute threads on the fastest cores.
//!
//! On Apple silicon, threads with the default QoS can be placed on any core
//! cluster; tagging the compute threads `QOS_CLASS_USER_INTERACTIVE` (the
//! highest class) asks the scheduler to prefer the fastest cores, which is
//! what a foreground batch job wants. Elsewhere these functions do nothing.

/// macOS QoS classes (`<sys/qos.h>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qos {
    UserInteractive = 0x21,
    UserInitiated = 0x19,
    Default = 0x15,
    Utility = 0x11,
    Background = 0x09,
}

impl Qos {
    /// `ui`, `initiated`, `default`, `utility`, `bg`.
    pub fn parse(s: &str) -> Option<Qos> {
        Some(match s.to_ascii_lowercase().as_str() {
            "ui" | "user_interactive" | "interactive" => Qos::UserInteractive,
            "initiated" | "user_initiated" => Qos::UserInitiated,
            "default" => Qos::Default,
            "utility" => Qos::Utility,
            "bg" | "background" => Qos::Background,
            _ => return None,
        })
    }
}

/// Sets the calling thread's QoS class (macOS only; a no-op elsewhere).
/// Returns whether the call succeeded.
pub fn set_current_thread_qos(q: Qos) -> bool {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        unsafe { pthread_set_qos_class_self_np(q as u32, 0) == 0 }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = q;
        false
    }
}

/// Raises the calling thread to `QOS_CLASS_USER_INTERACTIVE`.
pub fn set_current_thread_high_qos() -> bool {
    set_current_thread_qos(Qos::UserInteractive)
}

/// Builds rayon's global pool with `threads` workers (None = one per logical
/// CPU), each raised to the highest QoS class, and raises the calling thread
/// too (it runs work as well when it joins). Call once, before any parallel
/// work; later calls (or a pool already built) are ignored.
pub fn init_thread_pool(threads: Option<usize>) {
    set_current_thread_high_qos();
    let mut b = rayon::ThreadPoolBuilder::new().start_handler(|_| {
        set_current_thread_high_qos();
    });
    if let Some(n) = threads {
        b = b.num_threads(n);
    }
    b.build_global().ok();
}
