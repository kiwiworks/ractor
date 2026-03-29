// Copyright (c) Sean Lawlor
//
// This source code is licensed under both the MIT license found in the
// LICENSE-MIT file in the root directory of this source tree.

//! Built-in OTP-style supervisor actor.
//!
//! Provides automatic child restart with configurable strategies and intensity limits.
//!
//! # Example
//!
//! ```ignore
//! use ractor::actor::supervisor::*;
//! use ractor::{ChildSpec, RestartPolicy};
//!
//! let (sup_ref, _) = Supervisor::builder()
//!     .strategy(SupervisionStrategy::OneForOne)
//!     .restart_intensity(RestartIntensity { max_restarts: 3, period: Duration::from_secs(10) })
//!     .child(ChildSpec::default(), |sup_cell| Box::pin(async move {
//!         let (actor_ref, _) = MyActor::spawn_linked(None, MyActor, (), sup_cell).await?;
//!         Ok(actor_ref.get_cell())
//!     }))
//!     .spawn(Some("my_supervisor".to_string()))
//!     .await
//!     .expect("supervisor failed to start");
//! ```

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::actor_cell::ActorCell;
use super::actor_ref::ActorRef;
use super::messages::SupervisionEvent;
use super::supervision::{ChildSpec, RestartPolicy};
use crate::concurrency::Duration;
use crate::errors::{ActorProcessingErr, SpawnErr};
use crate::ActorId;

/// How a supervisor restarts children after a failure.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SupervisionStrategy {
    /// Restart only the failed child.
    #[default]
    OneForOne,
    /// Stop all children, then restart all.
    OneForAll,
    /// Stop children started after the failed child, then restart them in order.
    RestForOne,
}

/// Controls how many restarts are allowed before the supervisor itself terminates.
///
/// Matches OTP's `max_restarts` / `max_seconds` concept.
#[derive(Debug, Clone)]
pub struct RestartIntensity {
    /// Maximum number of restarts allowed within `period`.
    pub max_restarts: u32,
    /// Time window in which `max_restarts` is measured.
    pub period: Duration,
}

impl Default for RestartIntensity {
    fn default() -> Self {
        // OTP default: 1 restart per 5 seconds
        Self {
            max_restarts: 1,
            period: Duration::from_secs(5),
        }
    }
}

// ─── Child starter function type ─────────────────────────────────────────────

/// A function that spawns a child actor linked to the given supervisor cell.
///
/// Returns the child's [`ActorCell`] on success.
pub type ChildStarter = Arc<
    dyn Fn(ActorCell) -> Pin<Box<dyn Future<Output = Result<ActorCell, SpawnErr>> + Send>>
        + Send
        + Sync,
>;

/// A registered child entry in the supervisor.
struct ChildEntry {
    spec: ChildSpec,
    starter: ChildStarter,
    /// Current live cell for this child (None if not yet started or dead).
    cell: Option<ActorCell>,
}

impl std::fmt::Debug for ChildEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChildEntry")
            .field("spec", &self.spec)
            .field("cell", &self.cell.as_ref().map(ActorCell::get_id))
            .finish()
    }
}

// ─── Supervisor state ────────────────────────────────────────────────────────

/// Internal state for the [`Supervisor`] actor.
#[derive(Debug)]
pub struct SupervisorState {
    strategy: SupervisionStrategy,
    intensity: RestartIntensity,
    children: Vec<ChildEntry>,
    /// Rolling window of restart timestamps for intensity checking.
    restart_timestamps: VecDeque<crate::concurrency::Instant>,
}

// ─── Supervisor actor ────────────────────────────────────────────────────────

/// A built-in OTP-style supervisor actor.
///
/// Use [`Supervisor::builder()`] to construct and spawn.
pub struct Supervisor {
    strategy: SupervisionStrategy,
    intensity: RestartIntensity,
    child_defs: Vec<(ChildSpec, ChildStarter)>,
}

impl std::fmt::Debug for Supervisor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Supervisor")
            .field("strategy", &self.strategy)
            .field("intensity", &self.intensity)
            .field("num_children", &self.child_defs.len())
            .finish()
    }
}

/// Message type for the supervisor (no user messages).
#[derive(Debug)]
pub enum SupervisorMessage {}

#[cfg(not(feature = "async-trait"))]
impl crate::Actor for Supervisor {
    type Msg = SupervisorMessage;
    type State = SupervisorState;
    type Arguments = ();

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        _args: (),
    ) -> Result<Self::State, ActorProcessingErr> {
        let mut children = Vec::with_capacity(self.child_defs.len());
        let sup_cell = myself.get_cell();

        // Start all children in order
        for (spec, starter) in &self.child_defs {
            let cell = starter(sup_cell.clone())
                .await
                .map_err(|e| -> ActorProcessingErr { Box::new(e) })?;

            // Attach the child spec for graceful shutdown.
            // spawn_linked already linked with default spec; this overwrites it.
            cell.link_with_spec(sup_cell.clone(), spec.clone());

            children.push(ChildEntry {
                spec: spec.clone(),
                starter: Arc::clone(starter),
                cell: Some(cell),
            });
        }

        Ok(SupervisorState {
            strategy: self.strategy,
            intensity: self.intensity.clone(),
            children,
            restart_timestamps: VecDeque::new(),
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        _msg: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        Ok(())
    }

    async fn handle_supervisor_evt(
        &self,
        myself: ActorRef<Self::Msg>,
        evt: SupervisionEvent,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match evt {
            SupervisionEvent::ActorTerminated(cell, _boxed_state, exit_reason) => {
                handle_child_exit(myself, state, cell.get_id(), exit_reason.is_some()).await
            }
            SupervisionEvent::ActorFailed(cell, err) => {
                tracing::error!(
                    child = ?cell.get_id(),
                    error = %err,
                    "supervised child failed"
                );
                handle_child_exit(myself, state, cell.get_id(), false).await
            }
            SupervisionEvent::ActorStarted(_) | SupervisionEvent::ProcessGroupChanged(_) => Ok(()),
            #[cfg(feature = "cluster")]
            SupervisionEvent::PidLifecycleEvent(_) => Ok(()),
        }
    }
}

// Also implement for async-trait feature
#[cfg(feature = "async-trait")]
#[crate::async_trait]
impl crate::Actor for Supervisor {
    type Msg = SupervisorMessage;
    type State = SupervisorState;
    type Arguments = ();

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        _args: (),
    ) -> Result<Self::State, ActorProcessingErr> {
        let mut children = Vec::with_capacity(self.child_defs.len());
        let sup_cell = myself.get_cell();

        for (spec, starter) in &self.child_defs {
            let cell = starter(sup_cell.clone())
                .await
                .map_err(|e| -> ActorProcessingErr { Box::new(e) })?;

            cell.link_with_spec(sup_cell.clone(), spec.clone());

            children.push(ChildEntry {
                spec: spec.clone(),
                starter: Arc::clone(starter),
                cell: Some(cell),
            });
        }

        Ok(SupervisorState {
            strategy: self.strategy,
            intensity: self.intensity.clone(),
            children,
            restart_timestamps: VecDeque::new(),
        })
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        _msg: Self::Msg,
        _state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        Ok(())
    }

    async fn handle_supervisor_evt(
        &self,
        myself: ActorRef<Self::Msg>,
        evt: SupervisionEvent,
        state: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match evt {
            SupervisionEvent::ActorTerminated(cell, _boxed_state, exit_reason) => {
                handle_child_exit(myself, state, cell.get_id(), exit_reason.is_some()).await
            }
            SupervisionEvent::ActorFailed(cell, err) => {
                tracing::error!(
                    child = ?cell.get_id(),
                    error = %err,
                    "supervised child failed"
                );
                handle_child_exit(myself, state, cell.get_id(), false).await
            }
            SupervisionEvent::ActorStarted(_) | SupervisionEvent::ProcessGroupChanged(_) => Ok(()),
            #[cfg(feature = "cluster")]
            SupervisionEvent::PidLifecycleEvent(_) => Ok(()),
        }
    }
}

// ─── Core restart logic ──────────────────────────────────────────────────────

/// Determine whether a child should be restarted based on its [`RestartPolicy`]
/// and the nature of its exit.
fn should_restart(policy: RestartPolicy, normal_exit: bool) -> bool {
    match policy {
        RestartPolicy::Permanent => true,
        RestartPolicy::Transient => !normal_exit,
        RestartPolicy::Temporary => false,
    }
}

/// Check restart intensity. Returns `true` if the supervisor should self-terminate.
fn check_intensity(state: &mut SupervisorState) -> bool {
    let now = crate::concurrency::Instant::now();
    state.restart_timestamps.push_back(now);

    // Remove timestamps outside the window
    while let Some(&front) = state.restart_timestamps.front() {
        if now.duration_since(front) > state.intensity.period {
            state.restart_timestamps.pop_front();
        } else {
            break;
        }
    }

    state.restart_timestamps.len() > state.intensity.max_restarts as usize
}

/// Handle a child exit event. Dispatches to the appropriate strategy.
async fn handle_child_exit(
    myself: ActorRef<SupervisorMessage>,
    state: &mut SupervisorState,
    child_id: ActorId,
    normal_exit: bool,
) -> Result<(), ActorProcessingErr> {
    // Find the child entry
    let idx = state
        .children
        .iter()
        .position(|e| matches!(&e.cell, Some(c) if c.get_id() == child_id));

    let Some(idx) = idx else {
        // Unknown child — ignore (might be a transient actor we already removed)
        return Ok(());
    };

    let policy = state.children[idx].spec.restart_policy;

    if !should_restart(policy, normal_exit) {
        // Mark as dead, don't restart
        state.children[idx].cell = None;
        tracing::debug!(
            child = ?child_id,
            ?policy,
            normal_exit,
            "child exited, not restarting"
        );
        return Ok(());
    }

    // Check restart intensity before restarting
    if check_intensity(state) {
        tracing::error!(
            max_restarts = state.intensity.max_restarts,
            period = ?state.intensity.period,
            "restart intensity exceeded, supervisor terminating"
        );
        myself.stop(Some("restart intensity exceeded".to_string()));
        return Ok(());
    }

    match state.strategy {
        SupervisionStrategy::OneForOne => {
            restart_child(myself, state, idx).await?;
        }
        SupervisionStrategy::OneForAll => {
            restart_all(myself, state).await?;
        }
        SupervisionStrategy::RestForOne => {
            restart_rest(myself, state, idx).await?;
        }
    }

    Ok(())
}

/// Restart a single child by index.
async fn restart_child(
    myself: ActorRef<SupervisorMessage>,
    state: &mut SupervisorState,
    idx: usize,
) -> Result<(), ActorProcessingErr> {
    let entry = &mut state.children[idx];
    let starter = Arc::clone(&entry.starter);
    let spec = entry.spec.clone();
    let sup_cell = myself.get_cell();

    tracing::info!(child_idx = idx, ?spec, "restarting child");

    match starter(sup_cell.clone()).await {
        Ok(new_cell) => {
            new_cell.link_with_spec(sup_cell, spec);
            entry.cell = Some(new_cell);
            Ok(())
        }
        Err(err) => {
            tracing::error!(
                child_idx = idx,
                error = %err,
                "failed to restart child"
            );
            entry.cell = None;
            Err(Box::new(err))
        }
    }
}

/// Stop all children in reverse order, then restart all in forward order.
async fn restart_all(
    myself: ActorRef<SupervisorMessage>,
    state: &mut SupervisorState,
) -> Result<(), ActorProcessingErr> {
    let sup_cell = myself.get_cell();

    // Stop all live children in reverse order
    for entry in state.children.iter_mut().rev() {
        if let Some(cell) = entry.cell.take() {
            let timeout = entry.spec.shutdown_timeout;
            if cell.stop_and_wait(None, Some(timeout)).await.is_err() {
                cell.kill();
            }
        }
    }

    // Restart all in forward order
    for entry in &mut state.children {
        let starter = Arc::clone(&entry.starter);
        let spec = entry.spec.clone();
        match starter(sup_cell.clone()).await {
            Ok(new_cell) => {
                new_cell.link_with_spec(sup_cell.clone(), spec);
                entry.cell = Some(new_cell);
            }
            Err(err) => {
                tracing::error!(error = %err, "failed to restart child in one_for_all");
                entry.cell = None;
                return Err(Box::new(err));
            }
        }
    }

    Ok(())
}

/// Stop children started after the failed child (in reverse order),
/// then restart them (including the failed child) in forward order.
async fn restart_rest(
    myself: ActorRef<SupervisorMessage>,
    state: &mut SupervisorState,
    failed_idx: usize,
) -> Result<(), ActorProcessingErr> {
    let sup_cell = myself.get_cell();

    // Stop children from last down to failed_idx (reverse order)
    for i in (failed_idx..state.children.len()).rev() {
        if let Some(cell) = state.children[i].cell.take() {
            let timeout = state.children[i].spec.shutdown_timeout;
            if cell.stop_and_wait(None, Some(timeout)).await.is_err() {
                cell.kill();
            }
        }
    }

    // Restart from failed_idx forward
    for i in failed_idx..state.children.len() {
        let entry = &mut state.children[i];
        let starter = Arc::clone(&entry.starter);
        let spec = entry.spec.clone();
        match starter(sup_cell.clone()).await {
            Ok(new_cell) => {
                new_cell.link_with_spec(sup_cell.clone(), spec);
                entry.cell = Some(new_cell);
            }
            Err(err) => {
                tracing::error!(
                    child_idx = i,
                    error = %err,
                    "failed to restart child in rest_for_one"
                );
                entry.cell = None;
                return Err(Box::new(err));
            }
        }
    }

    Ok(())
}

// ─── Builder ─────────────────────────────────────────────────────────────────

/// Builder for constructing a [`Supervisor`].
pub struct SupervisorBuilder {
    strategy: SupervisionStrategy,
    intensity: RestartIntensity,
    children: Vec<(ChildSpec, ChildStarter)>,
}

impl std::fmt::Debug for SupervisorBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SupervisorBuilder")
            .field("strategy", &self.strategy)
            .field("intensity", &self.intensity)
            .field("num_children", &self.children.len())
            .finish()
    }
}

impl Supervisor {
    /// Create a new supervisor builder.
    pub fn builder() -> SupervisorBuilder {
        SupervisorBuilder {
            strategy: SupervisionStrategy::default(),
            intensity: RestartIntensity::default(),
            children: Vec::new(),
        }
    }
}

impl SupervisorBuilder {
    /// Set the restart strategy.
    #[must_use]
    pub fn strategy(mut self, strategy: SupervisionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the restart intensity (max restarts within a time period).
    #[must_use]
    pub fn restart_intensity(mut self, intensity: RestartIntensity) -> Self {
        self.intensity = intensity;
        self
    }

    /// Register a child with its spec and starter function.
    ///
    /// The starter function receives the supervisor's [`ActorCell`] and must
    /// spawn the child actor linked to it, returning the child's [`ActorCell`].
    #[must_use]
    pub fn child<F, Fut>(mut self, spec: ChildSpec, starter: F) -> Self
    where
        F: Fn(ActorCell) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ActorCell, SpawnErr>> + Send + 'static,
    {
        self.children
            .push((spec, Arc::new(move |cell| Box::pin(starter(cell)) as _)));
        self
    }

    /// Spawn the supervisor actor.
    ///
    /// # Errors
    /// Returns error if the supervisor or any of its children fail to start.
    pub async fn spawn(
        self,
        name: Option<String>,
    ) -> Result<
        (
            ActorRef<SupervisorMessage>,
            crate::concurrency::JoinHandle<()>,
        ),
        SpawnErr,
    > {
        let supervisor = Supervisor {
            strategy: self.strategy,
            intensity: self.intensity,
            child_defs: self.children,
        };
        crate::Actor::spawn(name, supervisor, ()).await
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;

    /// Actor that crashes on first message and counts starts.
    struct CrashOnMsg {
        start_count: Arc<AtomicU32>,
        /// Shared slot so tests can grab the latest ActorRef to send messages.
        ref_slot: Arc<Mutex<Option<ActorRef<()>>>>,
    }

    #[cfg_attr(feature = "async-trait", crate::async_trait)]
    impl crate::Actor for CrashOnMsg {
        type Msg = ();
        type State = ();
        type Arguments = (Arc<AtomicU32>, Arc<Mutex<Option<ActorRef<()>>>>);

        #[cfg(not(feature = "async-trait"))]
        fn pre_start(
            &self,
            myself: ActorRef<Self::Msg>,
            (counter, slot): Self::Arguments,
        ) -> impl Future<Output = Result<Self::State, ActorProcessingErr>> + Send {
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                *slot.lock().unwrap() = Some(myself);
                Ok(())
            }
        }

        #[cfg(feature = "async-trait")]
        async fn pre_start(
            &self,
            myself: ActorRef<Self::Msg>,
            (counter, slot): Self::Arguments,
        ) -> Result<Self::State, ActorProcessingErr> {
            counter.fetch_add(1, Ordering::SeqCst);
            *slot.lock().unwrap() = Some(myself);
            Ok(())
        }

        #[cfg(not(feature = "async-trait"))]
        fn handle(
            &self,
            _myself: ActorRef<Self::Msg>,
            _msg: Self::Msg,
            _state: &mut Self::State,
        ) -> impl Future<Output = Result<(), ActorProcessingErr>> + Send {
            async { panic!("intentional crash") }
        }

        #[cfg(feature = "async-trait")]
        async fn handle(
            &self,
            _myself: ActorRef<Self::Msg>,
            _msg: Self::Msg,
            _state: &mut Self::State,
        ) -> Result<(), ActorProcessingErr> {
            panic!("intentional crash")
        }
    }

    fn make_child_starter(
        start_count: Arc<AtomicU32>,
        ref_slot: Arc<Mutex<Option<ActorRef<()>>>>,
    ) -> impl Fn(ActorCell) -> Pin<Box<dyn Future<Output = Result<ActorCell, SpawnErr>> + Send>>
           + Send
           + Sync
           + 'static {
        move |sup_cell| {
            let sc = Arc::clone(&start_count);
            let slot = Arc::clone(&ref_slot);
            Box::pin(async move {
                let (actor_ref, _) = crate::Actor::spawn_linked(
                    None,
                    CrashOnMsg {
                        start_count: Arc::clone(&sc),
                        ref_slot: Arc::clone(&slot),
                    },
                    (sc, slot),
                    sup_cell,
                )
                .await?;
                Ok(actor_ref.get_cell())
            })
        }
    }

    #[crate::concurrency::test]
    async fn test_one_for_one_restarts_failed_child() {
        let start_count = Arc::new(AtomicU32::new(0));
        let ref_slot: Arc<Mutex<Option<ActorRef<()>>>> = Arc::new(Mutex::new(None));

        let (sup_ref, sup_handle) = Supervisor::builder()
            .strategy(SupervisionStrategy::OneForOne)
            .restart_intensity(RestartIntensity {
                max_restarts: 5,
                period: Duration::from_secs(10),
            })
            .child(
                ChildSpec::default(),
                make_child_starter(Arc::clone(&start_count), Arc::clone(&ref_slot)),
            )
            .spawn(None)
            .await
            .expect("supervisor should start");

        // Initial start = 1
        crate::concurrency::sleep(Duration::from_millis(100)).await;
        assert_eq!(1, start_count.load(Ordering::SeqCst));

        // Crash the child via its ActorRef
        {
            let slot = ref_slot.lock().unwrap();
            slot.as_ref().unwrap().cast(()).expect("send failed");
        }

        // Wait for restart
        crate::concurrency::sleep(Duration::from_millis(500)).await;
        assert_eq!(2, start_count.load(Ordering::SeqCst));

        sup_ref.stop(None);
        sup_handle.await.unwrap();
    }

    #[crate::concurrency::test]
    async fn test_temporary_child_not_restarted() {
        let start_count = Arc::new(AtomicU32::new(0));
        let ref_slot: Arc<Mutex<Option<ActorRef<()>>>> = Arc::new(Mutex::new(None));

        let spec = ChildSpec {
            restart_policy: RestartPolicy::Temporary,
            ..ChildSpec::default()
        };

        let (sup_ref, sup_handle) = Supervisor::builder()
            .strategy(SupervisionStrategy::OneForOne)
            .child(
                spec,
                make_child_starter(Arc::clone(&start_count), Arc::clone(&ref_slot)),
            )
            .spawn(None)
            .await
            .expect("supervisor should start");

        crate::concurrency::sleep(Duration::from_millis(100)).await;
        assert_eq!(1, start_count.load(Ordering::SeqCst));

        // Crash the child
        {
            let slot = ref_slot.lock().unwrap();
            slot.as_ref().unwrap().cast(()).expect("send failed");
        }

        // Wait — should NOT restart
        crate::concurrency::sleep(Duration::from_millis(500)).await;
        assert_eq!(1, start_count.load(Ordering::SeqCst));

        sup_ref.stop(None);
        sup_handle.await.unwrap();
    }

    #[crate::concurrency::test]
    async fn test_restart_intensity_exceeded_kills_supervisor() {
        let start_count = Arc::new(AtomicU32::new(0));
        let ref_slot: Arc<Mutex<Option<ActorRef<()>>>> = Arc::new(Mutex::new(None));

        let (sup_ref, sup_handle) = Supervisor::builder()
            .strategy(SupervisionStrategy::OneForOne)
            .restart_intensity(RestartIntensity {
                max_restarts: 2,
                period: Duration::from_secs(10),
            })
            .child(
                ChildSpec::default(),
                make_child_starter(Arc::clone(&start_count), Arc::clone(&ref_slot)),
            )
            .spawn(None)
            .await
            .expect("supervisor should start");

        // Crash the child 3 times rapidly (intensity limit is 2)
        for _ in 0..3 {
            crate::concurrency::sleep(Duration::from_millis(200)).await;
            let slot = ref_slot.lock().unwrap();
            if let Some(r) = slot.as_ref() {
                let _ = r.cast(());
            } else {
                break;
            }
        }

        // Wait for supervisor to self-terminate
        crate::concurrency::sleep(Duration::from_millis(500)).await;
        assert_eq!(
            sup_ref.get_cell().get_status(),
            crate::ActorStatus::Stopped,
            "supervisor should have terminated after intensity exceeded"
        );

        let _ = sup_handle.await;
    }

    #[crate::concurrency::test]
    async fn test_one_for_all_restarts_all_children() {
        let count_a = Arc::new(AtomicU32::new(0));
        let count_b = Arc::new(AtomicU32::new(0));
        let ref_slot_a: Arc<Mutex<Option<ActorRef<()>>>> = Arc::new(Mutex::new(None));
        let ref_slot_b: Arc<Mutex<Option<ActorRef<()>>>> = Arc::new(Mutex::new(None));

        let (sup_ref, sup_handle) = Supervisor::builder()
            .strategy(SupervisionStrategy::OneForAll)
            .restart_intensity(RestartIntensity {
                max_restarts: 5,
                period: Duration::from_secs(10),
            })
            .child(
                ChildSpec::default(),
                make_child_starter(Arc::clone(&count_a), Arc::clone(&ref_slot_a)),
            )
            .child(
                ChildSpec::default(),
                make_child_starter(Arc::clone(&count_b), Arc::clone(&ref_slot_b)),
            )
            .spawn(None)
            .await
            .expect("supervisor should start");

        crate::concurrency::sleep(Duration::from_millis(100)).await;
        assert_eq!(1, count_a.load(Ordering::SeqCst));
        assert_eq!(1, count_b.load(Ordering::SeqCst));

        // Crash child A — both should restart
        {
            let slot = ref_slot_a.lock().unwrap();
            slot.as_ref().unwrap().cast(()).expect("send failed");
        }

        crate::concurrency::sleep(Duration::from_millis(500)).await;
        assert_eq!(
            2,
            count_a.load(Ordering::SeqCst),
            "child A should have restarted"
        );
        assert_eq!(
            2,
            count_b.load(Ordering::SeqCst),
            "child B should also have restarted"
        );

        sup_ref.stop(None);
        sup_handle.await.unwrap();
    }

    #[crate::concurrency::test]
    async fn test_rest_for_one_restarts_subsequent_children() {
        let count_a = Arc::new(AtomicU32::new(0));
        let count_b = Arc::new(AtomicU32::new(0));
        let count_c = Arc::new(AtomicU32::new(0));
        let ref_slot_a: Arc<Mutex<Option<ActorRef<()>>>> = Arc::new(Mutex::new(None));
        let ref_slot_b: Arc<Mutex<Option<ActorRef<()>>>> = Arc::new(Mutex::new(None));
        let ref_slot_c: Arc<Mutex<Option<ActorRef<()>>>> = Arc::new(Mutex::new(None));

        let (sup_ref, sup_handle) = Supervisor::builder()
            .strategy(SupervisionStrategy::RestForOne)
            .restart_intensity(RestartIntensity {
                max_restarts: 5,
                period: Duration::from_secs(10),
            })
            .child(
                ChildSpec::default(),
                make_child_starter(Arc::clone(&count_a), Arc::clone(&ref_slot_a)),
            )
            .child(
                ChildSpec::default(),
                make_child_starter(Arc::clone(&count_b), Arc::clone(&ref_slot_b)),
            )
            .child(
                ChildSpec::default(),
                make_child_starter(Arc::clone(&count_c), Arc::clone(&ref_slot_c)),
            )
            .spawn(None)
            .await
            .expect("supervisor should start");

        crate::concurrency::sleep(Duration::from_millis(100)).await;
        assert_eq!(1, count_a.load(Ordering::SeqCst));
        assert_eq!(1, count_b.load(Ordering::SeqCst));
        assert_eq!(1, count_c.load(Ordering::SeqCst));

        // Crash child B — B and C should restart, A should NOT
        {
            let slot = ref_slot_b.lock().unwrap();
            slot.as_ref().unwrap().cast(()).expect("send failed");
        }

        crate::concurrency::sleep(Duration::from_millis(500)).await;
        assert_eq!(
            1,
            count_a.load(Ordering::SeqCst),
            "child A should NOT have restarted"
        );
        assert_eq!(
            2,
            count_b.load(Ordering::SeqCst),
            "child B should have restarted"
        );
        assert_eq!(
            2,
            count_c.load(Ordering::SeqCst),
            "child C should have restarted"
        );

        sup_ref.stop(None);
        sup_handle.await.unwrap();
    }
}
