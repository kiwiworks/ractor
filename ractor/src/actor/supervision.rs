// Copyright (c) Sean Lawlor
//
// This source code is licensed under both the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree.

//! Supervision management logic
//!
//! Supervision is a special notion of "ownership" over actors by a parent (supervisor).
//! Supervisors are responsible for the lifecycle of a child actor such that they get notified
//! when a child actor starts, stops, or panics (when possible). The supervisor can then decide
//! how to handle the event. Should it restart the actor, leave it dead, potentially die itself
//! notifying the supervisor's supervisor? That's up to the implementation of the [super::Actor]

#[cfg(feature = "monitors")]
use std::collections::HashMap;
use std::sync::Mutex;

use indexmap::IndexMap;

use super::actor_cell::ActorCell;
use super::messages::SupervisionEvent;
use crate::ActorId;

/// How a supervisor should respond when a child exits.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum RestartPolicy {
    /// Always restart the child (OTP `permanent`).
    #[default]
    Permanent,
    /// Restart only on abnormal exit — crash or error, not clean stop (OTP `transient`).
    Transient,
    /// Never restart (OTP `temporary`).
    Temporary,
}

/// Per-child configuration in a supervision tree.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ChildSpec {
    /// Maximum time to wait for graceful stop before killing.
    pub shutdown_timeout: crate::concurrency::Duration,
    /// When to restart this child after exit.
    pub restart_policy: RestartPolicy,
}

impl Default for ChildSpec {
    fn default() -> Self {
        Self {
            shutdown_timeout: crate::concurrency::Duration::from_secs(5),
            restart_policy: RestartPolicy::default(),
        }
    }
}

/// A supervision tree
///
/// Children are stored in insertion order ([`IndexMap`]) so that shutdown
/// can proceed in reverse start order, matching OTP supervisor semantics.
#[derive(Default, Debug)]
pub(crate) struct SupervisionTree {
    children: Mutex<Option<IndexMap<ActorId, (ActorCell, ChildSpec)>>>,
    supervisor: Mutex<Option<ActorCell>>,
    #[cfg(feature = "monitors")]
    monitors: Mutex<Option<HashMap<ActorId, ActorCell>>>,
}

impl SupervisionTree {
    /// Push a child into the tree with default spec (preserves insertion order)
    pub(crate) fn insert_child(&self, child: ActorCell) {
        self.insert_child_with_spec(child, ChildSpec::default());
    }

    /// Push a child into the tree with an explicit [`ChildSpec`]
    pub(crate) fn insert_child_with_spec(&self, child: ActorCell, spec: ChildSpec) {
        let mut guard = self.children.lock().unwrap();
        if let Some(map) = &mut *(guard) {
            map.insert(child.get_id(), (child, spec));
        } else {
            *guard = Some(IndexMap::from_iter([(child.get_id(), (child, spec))]));
        }
    }

    /// Remove a specific actor from the supervision tree (e.g. actor died).
    /// Uses `swap_remove` for O(1) removal.
    pub(crate) fn remove_child(&self, child: ActorId) {
        let mut guard = self.children.lock().unwrap();
        if let Some(map) = &mut *(guard) {
            map.swap_remove(&child);
        }
    }

    /// Retrieve the [`ChildSpec`] for a specific child, if it exists.
    #[allow(dead_code)]
    pub(crate) fn get_child_spec(&self, child: ActorId) -> Option<ChildSpec> {
        let guard = self.children.lock().unwrap();
        guard
            .as_ref()
            .and_then(|map| map.get(&child).map(|(_, spec)| spec.clone()))
    }

    /// Push a parent into the tere
    pub(crate) fn set_supervisor(&self, parent: ActorCell) {
        *(self.supervisor.lock().unwrap()) = Some(parent);
    }

    /// Remove a specific actor from the supervision tree (e.g. actor died)
    pub(crate) fn clear_supervisor(&self) {
        *(self.supervisor.lock().unwrap()) = None;
    }

    /// Try and retrieve the set supervisor
    pub(crate) fn try_get_supervisor(&self) -> Option<ActorCell> {
        self.supervisor.lock().unwrap().clone()
    }

    /// Set a monitor of this supervision tree
    #[cfg(feature = "monitors")]
    pub(crate) fn set_monitor(&self, who: ActorCell) {
        let mut guard = self.monitors.lock().unwrap();
        if let Some(map) = &mut *guard {
            map.insert(who.get_id(), who);
        } else {
            *guard = Some(HashMap::from_iter([(who.get_id(), who)]))
        }
    }

    /// Remove a specific monitor from the supervision tree
    #[cfg(feature = "monitors")]
    pub(crate) fn remove_monitor(&self, who: ActorId) {
        let mut guard = self.monitors.lock().unwrap();
        if let Some(map) = &mut *guard {
            map.remove(&who);
            if map.is_empty() {
                *guard = None;
            }
        }
    }

    /// Gracefully shut down all children with a timeout, then kill any survivors.
    ///
    /// Sends `stop` to each child and waits up to `timeout` for them to exit.
    /// Any child still alive after the timeout is forcefully killed.
    /// Clears the children map and unlinks all children from this supervisor.
    pub(crate) async fn graceful_shutdown_all_children(&self) {
        // Collect children with their specs in reverse insertion order and clear
        // the map under the lock. Release before any async work to keep the
        // future Send.
        let entries: Vec<(ActorCell, ChildSpec)> = {
            let mut guard = self.children.lock().unwrap();
            let entries = if let Some(map) = &mut *guard {
                // Reverse: shut down last-started child first (OTP semantics)
                map.values().rev().cloned().collect()
            } else {
                vec![]
            };
            *guard = None;
            entries
        };

        // Sequentially stop each child in reverse start order, using per-child
        // shutdown timeout from its ChildSpec. Kill on timeout.
        for (cell, spec) in &entries {
            let child_timeout = spec.shutdown_timeout;
            if cell.stop_and_wait(None, Some(child_timeout)).await.is_err() {
                cell.kill();
            }
            if cell.get_status() != super::actor_cell::ActorStatus::Stopped {
                cell.kill();
            }
            cell.clear_supervisor();
        }
    }

    /// Terminate all your supervised children and unlink them
    /// from the supervision tree since the supervisor is shutting down
    /// and can't deal with superivison events anyways
    pub(crate) fn terminate_all_children(&self) {
        let cells: Vec<ActorCell> = {
            let mut guard = self.children.lock().unwrap();
            let cells = if let Some(map) = &mut *guard {
                map.values().map(|(cell, _)| cell.clone()).collect()
            } else {
                vec![]
            };
            *guard = None;
            cells
        };
        for cell in cells {
            cell.terminate();
            cell.clear_supervisor();
        }
    }

    /// Stop all the linked children, but does NOT unlink them (stop flow will do that)
    pub(crate) fn stop_all_children(&self, reason: Option<String>) {
        self.for_each_child(|cell| {
            cell.stop(reason.clone());
        });
    }

    /// Drain all the linked children, but does NOT unlink them
    pub(crate) fn drain_all_children(&self) {
        self.for_each_child(|cell| {
            _ = cell.drain();
        });
    }

    /// Stop all the linked children, but does NOT unlink them (stop flow will do that),
    /// and wait for them to exit (concurrently)
    pub(crate) async fn stop_all_children_and_wait(
        &self,
        reason: Option<String>,
        timeout: Option<crate::concurrency::Duration>,
    ) {
        let cells = self.get_children();
        let mut js = crate::concurrency::JoinSet::new();
        for cell in cells {
            let lreason = reason.clone();
            let ltimeout = timeout;
            js.spawn(async move { cell.stop_and_wait(lreason, ltimeout).await });
        }
        // drain the tasks
        while let Some(res) = js.join_next().await {
            #[cfg(any(
                feature = "async-std",
                all(target_arch = "wasm32", target_os = "unknown")
            ))]
            if res.is_err() {
                panic!("JoinSet join error");
            }
            #[cfg(not(any(
                feature = "async-std",
                all(target_arch = "wasm32", target_os = "unknown")
            )))]
            {
                match res {
                    Err(err) if err.is_panic() => std::panic::resume_unwind(err.into_panic()),
                    Err(err) => panic!("{err}"),
                    _ => {}
                }
            }
        }
    }

    /// Drain all the linked children, but does NOT unlink them
    pub(crate) async fn drain_all_children_and_wait(
        &self,
        timeout: Option<crate::concurrency::Duration>,
    ) {
        let cells = self.get_children();
        let mut js = crate::concurrency::JoinSet::new();
        for cell in cells {
            let ltimeout = timeout;
            js.spawn(async move { cell.drain_and_wait(ltimeout).await });
        }
        // drain the tasks
        while let Some(res) = js.join_next().await {
            #[cfg(any(
                feature = "async-std",
                all(target_arch = "wasm32", target_os = "unknown")
            ))]
            if res.is_err() {
                panic!("JoinSet join error");
            }
            #[cfg(not(any(
                feature = "async-std",
                all(target_arch = "wasm32", target_os = "unknown")
            )))]
            {
                match res {
                    Err(err) if err.is_panic() => std::panic::resume_unwind(err.into_panic()),
                    Err(err) => panic!("{err}"),
                    _ => {}
                }
            }
        }
    }

    /// Determine if the specified actor is a parent of this actor
    pub(crate) fn is_child_of(&self, id: ActorId) -> bool {
        if let Some(parent) = &*(self.supervisor.lock().unwrap()) {
            parent.get_id() == id
        } else {
            false
        }
    }

    /// Return all linked children (cells only, no specs)
    pub(crate) fn get_children(&self) -> Vec<ActorCell> {
        let guard = self.children.lock().unwrap();
        if let Some(map) = &*guard {
            map.values().map(|(cell, _)| cell.clone()).collect()
        } else {
            vec![]
        }
    }

    /// Execute a closure for each child without allocating a Vec.
    pub(crate) fn for_each_child<F>(&self, mut f: F)
    where
        F: FnMut(&ActorCell),
    {
        let guard = self.children.lock().unwrap();
        if let Some(map) = &*guard {
            for (cell, _) in map.values() {
                f(cell);
            }
        }
    }

    /// Send a notification to the supervisor.
    ///
    /// Optimized to collect all targets under a single lock, then send outside the lock
    /// to minimize lock contention.
    pub(crate) fn notify_supervisor(&self, evt: SupervisionEvent) {
        // Collect all notification targets under a single lock acquisition
        #[cfg(feature = "monitors")]
        let monitor_targets: Vec<ActorCell> = {
            let guard = self.monitors.lock().unwrap();
            if let Some(monitors) = &*guard {
                monitors.values().cloned().collect()
            } else {
                Vec::new()
            }
        };

        let supervisor_target = {
            let guard = self.supervisor.lock().unwrap();
            (*guard).clone()
        };

        // Send to all monitors (best-effort, outside the lock)
        #[cfg(feature = "monitors")]
        if !monitor_targets.is_empty() {
            for monitor in monitor_targets.iter() {
                // Clone the event for each monitor (without requiring inner data to be Clone)
                let monitor_evt = evt.clone_no_data();
                if monitor.send_supervisor_evt(monitor_evt).is_err() {
                    // Best-effort delivery - if send fails, remove the monitor
                    let mut guard = self.monitors.lock().unwrap();
                    if let Some(monitors) = &mut *guard {
                        monitors.remove(&monitor.get_id());
                    }
                }
            }
        }

        // Send to supervisor
        if let Some(parent) = supervisor_target {
            _ = parent.send_supervisor_evt(evt);
        }
    }

    /// Retrieve the number of supervised children
    #[cfg(test)]
    pub(crate) fn get_num_children(&self) -> usize {
        let guard = self.children.lock().unwrap();
        if let Some(map) = &*guard {
            map.len()
        } else {
            0
        }
    }

    /// Retrieve the number of supervised children
    #[cfg(test)]
    pub(crate) fn get_num_parents(&self) -> usize {
        usize::from(self.supervisor.lock().unwrap().is_some())
    }
}
