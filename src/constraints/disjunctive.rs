//! `Disjunctive`: a unary resource. The jobs are processed one at a time, in the order given by
//! the scope, and each must respect its release date and its deadline.
//!
//! # Meaning
//!
//! There are $n$ jobs, numbered $0, \dots, n-1$, each with a release date $r_j$, a deadline $d_j$
//! and a duration $p_j \ge 0$. The scope is $x_1, \dots, x_n$, and $x_k$ is the job processed in
//! the $k$-th position. The constraint holds when:
//!
//! - $(x_1, \dots, x_n)$ is a permutation of the jobs, and
//! - processing the jobs in this order, each as early as possible and without interruption,
//!   every job $j$ starts at $s_j = \max(r_j, c_{prev})$ and ends at $c_j = s_j + p_j \le d_j$,
//!   where $c_{prev}$ is the end of the previous job (the first job starts at its release date).
//!
//! This is the permutation model of a unary resource, and it is the natural one for decision
//! diagrams: a layer is a position of the sequence, and an edge is a job. The start times are not
//! variables: they follow from the order. A deadline is a bound on the *completion* time.
//!
//! Optionally, a set of *precedences* $(i, j)$, written $i \ll j$: job $i$ is processed before
//! job $j$ (not necessarily right before). The constraint then also requires that $i$ comes
//! earlier than $j$ in the sequence.
//!
//! The scope must have exactly one variable per job, and the values of the domains are job
//! numbers. A value that is not a job number $0..n$ is never allowed. A variable may appear only
//! once in the scope, and the three lists must have one entry per variable: [`Disjunctive::new`]
//! panics otherwise, as for a negative duration, and as for a precedence that is not between two
//! different jobs, or that is part of a cycle (a cycle is a mistake of the model, not a
//! constraint to detect). Duplicate precedences are stored once. An empty scope is always
//! satisfied.
//!
//! Optionally, *setup times* $t_{i,j} \ge 0$: when job $j$ is processed right after job $i$, it
//! cannot start before $t_{i,j}$ time units after the end of $i$, so
//! $s_j = \max(r_j, c_i + t_{i,j})$. The first job has no setup. The matrix is given as `n` rows of
//! `n` entries, the diagonal is ignored, and a matrix with no positive entry is the same as none.
//!
//! # How it compiles
//!
//! A node carries two things, computed top-down (the paths from the root) and bottom-up (the
//! paths to the sink):
//!
//! - **The permutation state**, the pair of sets $(A, S)$ of [`AllDifferent`]: the jobs taken on
//!   all the paths and on some path.
//! - **A time bound**, which differs in the two directions. Top-down, the *earliest completion
//!   time* $e$ of the jobs placed so far, over the paths that reach the node. Bottom-up, the
//!   *latest start time* $l$ of the jobs that remain, over the paths that leave the node. Both are
//!   kept as a pair, the best case (smallest $e$, largest $l$) used to filter, and the worst case
//!   (largest $e$, smallest $l$), which is what tells that all the paths agree.
//! - **With setup times, what is known of the job at the open end of the paths**: the last job
//!   placed (top-down) or the first job still to place (bottom-up). The setup that comes next
//!   depends on it. How much of it a property keeps is the choice of a [`SetupRelaxation`], see
//!   below. Nothing is kept when there is no setup time.
//!
//! ## Property update
//!
//! Top-down, a node folds its parents: an edge with the job $j$ from a parent whose earliest
//! completion is $e$ gives
//!
//! $$e \otimes j = \max(e, r_j) + p_j$$
//!
//! for the best case and for the worst case, and the node keeps the smallest best case and the
//! largest worst case over its parents. The root has no job yet, $e = -\infty$.
//!
//! Bottom-up, an edge with the job $j$ towards a child whose latest start is $l$ gives
//!
//! $$l \otimes j = \min(l, d_j) - p_j$$
//!
//! and the node keeps the largest best case and the smallest worst case. The sink has no job to
//! place, $l = +\infty$. Out of scope layers copy the property. The sets $(A, S)$ are updated as
//! in [`AllDifferent`].
//!
//! With setup times, the job $i$ at the open end is read to add the setup between it and $j$:
//! $e \otimes j = \max(e + t_{i,j}, r_j) + p_j$ top-down, and
//! $l \otimes j = \min(l - t_{j,k}, d_j) - p_j$ bottom-up, where $k$ is the job at the open end of
//! the child. The root and the sink have no job, so no setup. The job of the child is $j$ itself:
//! the paths that leave an edge with the job $j$ all end with $j$. What happens when the paths of a
//! node do not agree on $i$ or $k$ is the subject of the next section.
//!
//! ## Setup times: the three relaxations
//!
//! When a node merges paths that end with different jobs, its state cannot give *the* setup that
//! comes next. A [`SetupRelaxation`] says what the state keeps instead, from the cheapest and the
//! weakest to the most expensive and the strongest. All of them are exact on a single path, they
//! all keep every solution, and a node is a single state in all of them only if its paths agree
//! on the job at the open end (and on the time), so the true setup of every edge is used in a
//! fully refined diagram: the choice only matters for the nodes of a relaxed diagram, and for the
//! work that the compilation spends on a node.
//!
//! - [`SetupRelaxation::LastJob`] (the default). One integer: the job, or a marker if the paths
//!   disagree. When they disagree, the best case uses the smallest setup that could come after
//!   (or before) $j$ and the worst case the largest, over all the other jobs; they are computed
//!   once, at construction. Merging, hashing and comparing a state is one integer, and the filter
//!   reads one setup in a table. A node that merges two last jobs forgets which ones.
//! - [`SetupRelaxation::LastSet`]. The set of the jobs, one bit each (plus one for the empty
//!   path). When the paths disagree the best case uses the smallest setup over *the jobs of the
//!   set*, and the worst case the largest. Merging is an `or` of $\lceil (n+1)/64 \rceil$ words,
//!   like the permutation state, and the filter walks the set bits, $O(n)$ in the worst case. It
//!   is never weaker than `LastJob`.
//! - [`SetupRelaxation::ArcTimes`]. One time bound per job at the open end (the arc states
//!   $ect_a$ and $lst_a$ of Cire and van Hoeve, sections 4 and 6.3): the best and the worst
//!   completion time of the paths top-down (the best and the worst latest start bottom-up) that
//!   end with each job. The edge test is
//!   $\max(r_j, \min_{i \neq j} (e_i + t_{i,j})) + p_j > \min(d_j, \max_{k \neq j} (l_k - t_{j,k}))$,
//!   which pairs every last job with its own time and not with the best time of the node. It is
//!   never weaker than `LastSet`. A state holds $4(n+1)$ integers, and an update or a test is
//!   $O(n)$. The scalar bounds are kept as well, as the best and the worst of the slots.
//!
//! A node is a single state, in all the relaxations, only when the paths agree on the job. The
//! recommendation is `LastJob` while the diagrams are exact, since nothing weaker or cheaper is
//! lost; `LastSet` or `ArcTimes` pay off when the width is limited and the setups differ a lot
//! between the jobs that a node merges. Nothing is kept, and no instruction of the setup times is
//! executed, when there are no setup times.
//!
//! ## Node merging
//!
//! Merging two nodes merges the sets as in [`AllDifferent`] (intersection of $A$, union of $S$) and
//! takes, for each time bound, the smallest best case and the largest worst case in the top-down
//! direction (resp. the largest and the smallest in the bottom-up one). The part that stands for
//! the job at the open end is merged as its relaxation says (the marker of disagreement, the union
//! of the sets, the slot-wise bounds). The merged node can only be more permissive than each of
//! them, so this is a relaxation.
//!
//! ## Edge filtering
//!
//! An edge with job $j$, between a parent with earliest completion $e$ and a child with latest
//! start $l$, is removed in three cases:
//!
//! 1. the [`AllDifferent`] test says so: $j$ is on all the paths above or below, or the jobs on
//!    some path above and below form a full Hall set;
//! 2. the jobs that could still be placed are not enough: $|S_{parent} \cup S_{child} \cup \{j\}| < n$,
//!    so some job is on no path through the edge, while every job has to be placed;
//! 3. the time windows cannot be met: the job ends, at best, after the latest moment it may end:
//!
//!    $$\max(e, r_j) + p_j > \min(l, d_j)$$
//!
//!    With setup times, $e$ is replaced by $e + t_{i,j}$ and $l$ by $l - t_{j,k}$, where $i$ and
//!    $k$ are the jobs at the open end of the parent and of the child (the smallest setup that
//!    the relaxation allows if they are not known; with `ArcTimes`, the best pairing of a last
//!    job with its own time).
//!
//! 4. a precedence is broken: some job $i \ll j$ is in no set $S$ of the parent, so no path above
//!    has placed it, or some job $j \ll k$ is in no set $S$ of the child, so no path below will.
//!
//! Rule 3 is exact on a single path, and on an exact node: it is the schedule that is built, and
//! the deadlines of the jobs after the edge are taken into account through $l$. With merged
//! nodes it is a necessary condition only (with setup times, the more so when the jobs at the
//! open ends are not known). Rule 4 is exact as well: on a single path, a job placed
//! before one of its predecessors is caught at the edge that places it. It needs no state of its
//! own, the sets $S$ that the permutation already carries are enough.
//!
//! ## Single state
//!
//! A node is a single state when $A = S$ (all the paths place the same jobs) and each time bound
//! has its best and worst case equal (all the paths give the same time, so the future of the
//! node does not depend on the path). Two paths that place the same jobs in different orders
//! usually end at different times, so this is stricter than for [`AllDifferent`]. With setup
//! times the paths must also agree on the job at the open end: two paths that end at the same
//! time with different last jobs have different futures.
//!
//! # Design notes
//!
//! - **The permutation is embedded, not posted aside.** The time bounds are only meaningful if
//!   the jobs are all different, and the two tests (rule 1 and 2) are what makes the diagram
//!   small. The constraint owns an [`AllDifferent`] over the same scope and delegates to it.
//! - **The scope is read as a sequence.** Like [`Regular`], the variables are branched in the
//!   declared order, which `precedence_edges` forces. A shared structure of the arena therefore
//!   serves any scope with the same jobs.
//! - **Precedences are read from the permutation state.** They cost nothing in the property, so
//!   they do not make nodes harder to merge: they only remove edges, and the structural key holds
//!   them (sorted) next to the jobs. They do not tighten the time bounds, which could use them
//!   (a job cannot start before the end of its predecessors).
//! - **The filtering rules are tried by increasing cost, and share their work.** The time rule
//!   (3) is a few integer operations and comes first. The two permutation rules (1 and 2) read the
//!   same bit and the same union of sets, so [`AllDifferent`] evaluates them in one call. The
//!   precedence rule (4) reads the sets through the bits of the predecessors and successors,
//!   found once at construction, and is skipped when there is no precedence. The order does not
//!   change which edges are removed: an edge goes if any rule says so.
//! - **Time bounds are a single number per node**, not one per job. This is the recurrence of
//!   Hoda et al. and of Cire and van Hoeve without setup times, where the extra condition on the
//!   previous job (the arc rule $val(a) \neq val(a')$) is already implied by the permutation
//!   state.
//! - **Setup times are optional, and so is what they cost.** Everything that they add is behind
//!   one test of an `Option` in the constraint, as for the precedences, so the plain constraint
//!   executes the same instructions as before. The state of a property then holds the part that
//!   its [`SetupRelaxation`] asks for, as an enum whose cheapest case is nothing.
//! - **The relaxation is a parameter, not a decision.** An exact diagram needs none of it: a node
//!   is a single state only if its paths agree on the job at the open end, so the true setup of
//!   every edge is used. What the relaxations differ in is what a node that *merges* last jobs
//!   knows: the cheapest forgets which jobs were merged, the set remembers which, the arcs
//!   remember the time of each. The first costs an integer, the second $O(n/64)$ words to merge
//!   and $O(n)$ to filter, the third $O(n)$ for both. The enum of the state is matched once per
//!   operation, on a variant that is the same for the whole constraint, and the choice is part of
//!   the structural key so that the arena does not share a structure between two relaxations.
//! - **Time is an `isize`** and sentinels saturate, so a huge release date or deadline does not
//!   overflow.
//! - **Domains are read once**, by the embedded [`AllDifferent`], so the remarks of that module
//!   apply: shrinking a domain afterwards is fine.
//!
//! # Example
//!
//! ```
//! use aicad::constraints::{Constraint, Disjunctive};
//! use aicad::modelling::*;
//!
//! let mut problem = Problem::default();
//! let vars = problem.add_variables(3, vec![0, 1, 2]);
//! // Job 1 has to end before time 4, so it goes first or second.
//! let disjunctive = Disjunctive::new(
//!     vars.clone(),
//!     vec![0, 0, 0],
//!     vec![10, 4, 10],
//!     vec![3, 2, 2],
//!     None,
//!     None,
//!     &problem,
//! );
//!
//! assert!(disjunctive.is_satisfied(&[1, 0, 2]));
//! assert!(!disjunctive.is_satisfied(&[0, 1, 2]));
//!
//! // With the precedence "job 2 before job 1", only [2, 1, 0] is left.
//! let ordered = Disjunctive::new(
//!     vars,
//!     vec![0, 0, 0],
//!     vec![10, 4, 10],
//!     vec![3, 2, 2],
//!     Some(vec![(2, 1)]),
//!     None,
//!     &problem,
//! );
//! assert!(!ordered.is_satisfied(&[1, 0, 2]));
//! assert!(ordered.is_satisfied(&[2, 1, 0]));
//! ```
//!
//! # References
//!
//! - A. A. Cire and W.-J. van Hoeve. *Multivalued decision diagrams for sequencing problems.*
//!   Operations Research 61(6), 2013, pp. 1411-1428. The permutation model, the states $All$ and
//!   $Some$, the earliest completion and latest start times and the filtering of precedences
//!   (sections 4, 6.1, 6.2 and 6.3).
//! - S. Hoda, W.-J. van Hoeve and J. N. Hooker. *A systematic approach to MDD-based constraint
//!   programming.* CP 2010, pp. 266-280. The unary resource constraint (section 4.7), from which
//!   the two time bounds are taken.
use super::all_different::{AllDifferent, AllDifferentProperty};
use super::*;
use crate::modelling::VariableIndex;
use rustc_hash::FxHashSet;
use std::hash::Hasher;
use std::sync::Arc;

/// No path has reached the property yet, the neutral element of [`join_job`].
const NO_JOB: u32 = u32::MAX;
/// The empty path: the root (top-down) and the sink (bottom-up). No job, so no setup.
const START: u32 = u32::MAX - 1;
/// The paths of the property do not agree on the job at the open end.
const MANY: u32 = u32::MAX - 2;

/// The job at the open end of the paths of a node that two sets of paths agree on: the common job
/// if they have the same one, and `MANY` if not. `NO_JOB` is neutral. A join, so that merging
/// stays one.
fn join_job(a: u32, b: u32) -> u32 {
    if a == b || b == NO_JOB {
        a
    } else if a == NO_JOB {
        b
    } else {
        MANY
    }
}

/// The setup times of a [`Disjunctive`] and how their relaxation is represented, to give to
/// [`Disjunctive::new`].
///
/// ```
/// use aicad::constraints::{SetupRelaxation, SetupTimes};
///
/// let times = SetupTimes::new(vec![vec![0, 2], vec![1, 0]]);
/// assert_eq!(times.relaxation, SetupRelaxation::LastJob);
/// let times = times.with_relaxation(SetupRelaxation::ArcTimes);
/// assert_eq!(times.relaxation, SetupRelaxation::ArcTimes);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetupTimes {
    /// `matrix[i][j]` is the time that has to pass between the end of job `i` and the start of job
    /// `j` when `j` follows `i`. The diagonal is ignored.
    pub matrix: Vec<Vec<isize>>,
    /// What the states remember about the job at the open end; see [`SetupRelaxation`].
    pub relaxation: SetupRelaxation,
}

impl SetupTimes {
    /// The setup times `matrix`, with the default relaxation ([`SetupRelaxation::LastJob`]).
    pub fn new(matrix: Vec<Vec<isize>>) -> Self {
        Self {
            matrix,
            relaxation: SetupRelaxation::default(),
        }
    }

    /// The same setup times with another relaxation.
    pub fn with_relaxation(mut self, relaxation: SetupRelaxation) -> Self {
        self.relaxation = relaxation;
        self
    }
}

/// The jobs at the open end of the paths of a property, as a bitset over `n + 1` bits: bit `j` for
/// the job `j`, and bit `n` for the empty path. A node with no path has the empty set.
type JobSet = Vec<u64>;

/// Number of words of a [`JobSet`] for `n` jobs.
fn job_set_len(n: usize) -> usize {
    (n + 1).div_ceil(64)
}

/// The slots of the time bounds of [`SetupRelaxation::ArcTimes`], one per job at the open end and
/// one (the last, `n`) for the empty path. A slot that no path reaches holds the neutral element
/// of its merge: best case `isize::MAX` and worst case `isize::MIN` top-down, the reverse
/// bottom-up. A slot is therefore *populated* when its best and worst case are in the order of a
/// real value (`best <= worst` top-down, `best >= worst` bottom-up).
#[derive(Clone, deepsize::DeepSizeOf)]
struct Arcs {
    /// Top-down, for each last job: the smallest earliest completion time over the paths.
    ect_best: Vec<isize>,
    /// Top-down, for each last job: the largest earliest completion time over the paths.
    ect_worst: Vec<isize>,
    /// Bottom-up, for each first job: the largest latest start time over the paths.
    lst_best: Vec<isize>,
    /// Bottom-up, for each first job: the smallest latest start time over the paths.
    lst_worst: Vec<isize>,
}

impl Arcs {
    /// No path anywhere.
    fn identity(n: usize) -> Self {
        Self {
            ect_best: vec![isize::MAX; n + 1],
            ect_worst: vec![isize::MIN; n + 1],
            lst_best: vec![isize::MIN; n + 1],
            lst_worst: vec![isize::MAX; n + 1],
        }
    }

    /// The empty path, from the root ($-\infty$) and from the sink ($+\infty$).
    fn empty(n: usize) -> Self {
        let mut arcs = Self::identity(n);
        arcs.ect_best[n] = isize::MIN;
        arcs.ect_worst[n] = isize::MIN;
        arcs.lst_best[n] = isize::MAX;
        arcs.lst_worst[n] = isize::MAX;
        arcs
    }

    /// Slot-wise merge, which is a join like the scalar bounds.
    fn merge(&mut self, other: &Self) {
        for (mine, theirs) in self.ect_best.iter_mut().zip(&other.ect_best) {
            *mine = (*mine).min(*theirs);
        }
        for (mine, theirs) in self.ect_worst.iter_mut().zip(&other.ect_worst) {
            *mine = (*mine).max(*theirs);
        }
        for (mine, theirs) in self.lst_best.iter_mut().zip(&other.lst_best) {
            *mine = (*mine).max(*theirs);
        }
        for (mine, theirs) in self.lst_worst.iter_mut().zip(&other.lst_worst) {
            *mine = (*mine).min(*theirs);
        }
    }

    /// Number of slots reached by some path, top-down.
    fn populated_ect(&self) -> usize {
        self.ect_best
            .iter()
            .zip(&self.ect_worst)
            .filter(|(best, worst)| best <= worst)
            .count()
    }

    /// Number of slots reached by some path, bottom-up.
    fn populated_lst(&self) -> usize {
        self.lst_best
            .iter()
            .zip(&self.lst_worst)
            .filter(|(best, worst)| best >= worst)
            .count()
    }

    /// The smallest and the largest time at which `job` can start after the paths of this
    /// top-down state (the completion time of a path plus the setup after its last job), or
    /// `None` if no path can be followed by `job`.
    fn ready(&self, setup: &Setup, job: usize) -> Option<(isize, isize)> {
        let n = setup.n;
        let mut found: Option<(isize, isize)> = None;
        for last in 0..=n {
            let (best, worst) = (self.ect_best[last], self.ect_worst[last]);
            if best > worst || last == job {
                continue;
            }
            let time = if last == n {
                0
            } else {
                setup.times[last * n + job]
            };
            let (best, worst) = (best.saturating_add(time), worst.saturating_add(time));
            found = Some(match found {
                None => (best, worst),
                Some((min, max)) => (min.min(best), max.max(worst)),
            });
        }
        found
    }

    /// The largest and the smallest time at which `job` can end for the paths of this bottom-up
    /// state (the latest start of a path minus the setup before its first job), or `None` if
    /// `job` cannot be followed by any path.
    fn due(&self, setup: &Setup, job: usize) -> Option<(isize, isize)> {
        let n = setup.n;
        let mut found: Option<(isize, isize)> = None;
        for first in 0..=n {
            let (best, worst) = (self.lst_best[first], self.lst_worst[first]);
            if best < worst || first == job {
                continue;
            }
            let time = if first == n {
                0
            } else {
                setup.times[job * n + first]
            };
            let (best, worst) = (best.saturating_sub(time), worst.saturating_sub(time));
            found = Some(match found {
                None => (best, worst),
                Some((max, min)) => (max.max(best), min.min(worst)),
            });
        }
        found
    }
}

/// What a property remembers about the job at the open end of its paths, for the setup time that
/// comes next: nothing without setup times, and otherwise what the [`SetupRelaxation`] says. The
/// variant is the same for all the properties of a constraint.
#[derive(Clone, deepsize::DeepSizeOf)]
enum Open {
    /// No setup times.
    Nothing,
    /// [`SetupRelaxation::LastJob`]: a job, or one of `NO_JOB`, `START` and `MANY`.
    Job(u32),
    /// [`SetupRelaxation::LastSet`]: the jobs, as bits.
    Set(JobSet),
    /// [`SetupRelaxation::ArcTimes`]: the times of each job.
    Arcs(Box<Arcs>),
}

impl Open {
    /// The state of a property that no path has reached.
    fn identity(setup: &Option<Setup>) -> Self {
        match setup {
            None => Self::Nothing,
            Some(setup) => match setup.relaxation {
                SetupRelaxation::LastJob => Self::Job(NO_JOB),
                SetupRelaxation::LastSet => Self::Set(vec![0; job_set_len(setup.n)]),
                SetupRelaxation::ArcTimes => Self::Arcs(Box::new(Arcs::identity(setup.n))),
            },
        }
    }

    /// The state of the empty path: no job, so no setup.
    fn empty(setup: &Option<Setup>) -> Self {
        match setup {
            None => Self::Nothing,
            Some(setup) => match setup.relaxation {
                SetupRelaxation::LastJob => Self::Job(START),
                SetupRelaxation::LastSet => {
                    let mut set = vec![0; job_set_len(setup.n)];
                    set[setup.n / 64] |= 1 << (setup.n % 64);
                    Self::Set(set)
                }
                SetupRelaxation::ArcTimes => Self::Arcs(Box::new(Arcs::empty(setup.n))),
            },
        }
    }

    /// The two variants are not the same: a bug, since a constraint has a single relaxation.
    fn mismatch() -> ! {
        panic!("Disjunctive: properties with different setup relaxations")
    }

    /// Joins `other` into `self`, which is what merging does.
    fn merge(&mut self, other: &Self) {
        match (self, other) {
            (Self::Nothing, Self::Nothing) => {}
            (Self::Job(mine), Self::Job(theirs)) => *mine = join_job(*mine, *theirs),
            (Self::Set(mine), Self::Set(theirs)) => {
                for (mine, theirs) in mine.iter_mut().zip(theirs) {
                    *mine |= *theirs;
                }
            }
            (Self::Arcs(mine), Self::Arcs(theirs)) => mine.merge(theirs),
            _ => Self::mismatch(),
        }
    }

    /// Whether the paths agree on the job at the open end (and, for the arcs, on the time).
    fn is_single_state(&self) -> bool {
        match self {
            Self::Nothing => true,
            Self::Job(job) => *job != MANY,
            Self::Set(set) => set.iter().map(|word| word.count_ones()).sum::<u32>() <= 1,
            Self::Arcs(arcs) => arcs.populated_ect() <= 1 && arcs.populated_lst() <= 1,
        }
    }

    /// Feeds the state to `hasher`.
    fn hash(&self, hasher: &mut dyn Hasher) {
        match self {
            Self::Nothing => {}
            Self::Job(job) => hasher.write_u32(*job),
            Self::Set(set) => set.iter().for_each(|word| hasher.write_u64(*word)),
            Self::Arcs(arcs) => {
                for slots in [
                    &arcs.ect_best,
                    &arcs.ect_worst,
                    &arcs.lst_best,
                    &arcs.lst_worst,
                ] {
                    slots.iter().for_each(|time| hasher.write_isize(*time));
                }
            }
        }
    }

    /// Whether the two states are the same.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Nothing, Self::Nothing) => true,
            (Self::Job(mine), Self::Job(theirs)) => mine == theirs,
            (Self::Set(mine), Self::Set(theirs)) => mine == theirs,
            (Self::Arcs(mine), Self::Arcs(theirs)) => {
                mine.ect_best == theirs.ect_best
                    && mine.ect_worst == theirs.ect_worst
                    && mine.lst_best == theirs.lst_best
                    && mine.lst_worst == theirs.lst_worst
            }
            _ => Self::mismatch(),
        }
    }

    /// Adds to `key` the numbers that describe this state, a fixed count for a constraint.
    fn push_order_key(&self, n: usize, key: &mut Vec<f64>) {
        match self {
            Self::Nothing => {}
            // The jobs are numbered below `n`, and every marker is put after them.
            Self::Job(job) => key.push((*job).min(n as u32) as f64),
            Self::Set(set) => {
                key.push(set.iter().map(|word| word.count_ones()).sum::<u32>() as f64)
            }
            Self::Arcs(arcs) => {
                key.push(arcs.populated_ect() as f64);
                key.push(arcs.populated_lst() as f64);
            }
        }
    }
}

/// Sets the bit `bit` of `set`.
fn set_bit(set: &mut [u64], bit: usize) {
    set[bit / 64] |= 1 << (bit % 64);
}

/// The setup times, and the numbers that the filtering needs when the job at the open end is not
/// known. Built once, shared through [`Jobs`].
#[derive(Clone, deepsize::DeepSizeOf)]
struct Setup {
    /// How the open end of the paths is represented in the properties.
    relaxation: SetupRelaxation,
    /// Number of jobs.
    n: usize,
    /// `times[i * n + j]` is the setup between `i` and `j` when `j` follows `i`. The diagonal is
    /// 0 and never read in a way that matters (a job is never next to itself).
    times: Vec<isize>,
    /// For each job `j`, the smallest setup that can come right before it.
    min_into: Vec<isize>,
    /// For each job `j`, the largest setup that can come right before it.
    max_into: Vec<isize>,
    /// For each job `j`, the smallest setup that can come right after it.
    min_from: Vec<isize>,
    /// For each job `j`, the largest setup that can come right after it.
    max_from: Vec<isize>,
}

impl Setup {
    /// `times` is the matrix, `n` rows of `n` entries, flattened row by row, with a zero diagonal.
    fn new(n: usize, times: Vec<isize>, relaxation: SetupRelaxation) -> Self {
        let others = |job: usize, of_row: bool| -> Vec<isize> {
            (0..n)
                .filter(|&other| other != job)
                .map(|other| {
                    if of_row {
                        times[job * n + other]
                    } else {
                        times[other * n + job]
                    }
                })
                .collect()
        };
        let extreme = |values: Vec<isize>, largest: bool| -> isize {
            let found = if largest {
                values.into_iter().max()
            } else {
                values.into_iter().min()
            };
            found.unwrap_or(0)
        };
        let min_into = (0..n).map(|j| extreme(others(j, false), false)).collect();
        let max_into = (0..n).map(|j| extreme(others(j, false), true)).collect();
        let min_from = (0..n).map(|j| extreme(others(j, true), false)).collect();
        let max_from = (0..n).map(|j| extreme(others(j, true), true)).collect();
        Self {
            relaxation,
            n,
            times,
            min_into,
            max_into,
            min_from,
            max_from,
        }
    }

    /// The smallest setup before `job`, given the job at the open end of the paths above it.
    fn before_min(&self, last: u32, job: usize) -> isize {
        match last {
            NO_JOB | START => 0,
            MANY => self.min_into[job],
            last => self.times[last as usize * self.n + job],
        }
    }

    /// The largest setup before `job`, given the job at the open end of the paths above it.
    fn before_max(&self, last: u32, job: usize) -> isize {
        match last {
            NO_JOB | START => 0,
            MANY => self.max_into[job],
            last => self.times[last as usize * self.n + job],
        }
    }

    /// The smallest setup after `job`, given the job at the open end of the paths below it.
    fn after_min(&self, job: usize, first: u32) -> isize {
        match first {
            NO_JOB | START => 0,
            MANY => self.min_from[job],
            first => self.times[job * self.n + first as usize],
        }
    }

    /// The largest setup after `job`, given the job at the open end of the paths below it.
    fn after_max(&self, job: usize, first: u32) -> isize {
        match first {
            NO_JOB | START => 0,
            MANY => self.max_from[job],
            first => self.times[job * self.n + first as usize],
        }
    }

    /// The smallest (or the largest) of the values `time(i)` over the jobs `i` of `set` other
    /// than `job`, where the bit `n` is the empty path, whose setup is 0. A set with no such job
    /// gives 0: no setup is assumed, which keeps every bound safe.
    fn over_set(
        &self,
        set: &[u64],
        job: usize,
        largest: bool,
        time: impl Fn(usize) -> isize,
    ) -> isize {
        let mut found: Option<isize> = None;
        for (index, &word) in set.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let other = index * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if other == job {
                    continue;
                }
                let value = if other == self.n { 0 } else { time(other) };
                found = Some(match found {
                    None => value,
                    Some(best) if largest => best.max(value),
                    Some(best) => best.min(value),
                });
            }
        }
        found.unwrap_or(0)
    }

    /// The smallest (or the largest) setup before `job`, given the set of the jobs at the open
    /// end of the paths above it.
    fn set_before(&self, set: &[u64], job: usize, largest: bool) -> isize {
        self.over_set(set, job, largest, |last| self.times[last * self.n + job])
    }

    /// The smallest (or the largest) setup after `job`, given the set of the jobs at the open end
    /// of the paths below it.
    fn set_after(&self, set: &[u64], job: usize, largest: bool) -> isize {
        self.over_set(set, job, largest, |first| self.times[job * self.n + first])
    }
}

/// The data of the jobs, shared by the constraint and by all of its properties.
#[derive(Clone, deepsize::DeepSizeOf)]
struct Jobs {
    /// Release date of each job.
    release: Vec<isize>,
    /// Deadline (latest completion time) of each job.
    deadline: Vec<isize>,
    /// Duration of each job.
    duration: Vec<isize>,
    /// The precedences `(i, j)` (`i` before `j`), sorted and without duplicates.
    precedences: Vec<(usize, usize)>,
    /// For each job, the jobs that have to come before it.
    predecessors: Vec<Vec<usize>>,
    /// For each job, the jobs that have to come after it.
    successors: Vec<Vec<usize>>,
    /// `predecessors`, as the bits that stand for the jobs in the permutation state (see
    /// `AllDifferent::bit_of`), so that the filtering reads a state without a hash lookup per job.
    predecessor_bits: Vec<Vec<Option<usize>>>,
    /// `successors`, as bits.
    successor_bits: Vec<Vec<Option<usize>>>,
    /// The setup times, if some is positive.
    setup: Option<Setup>,
}

/// Per-node state of [`Disjunctive`]: the permutation state of [`AllDifferent`] and the time
/// bounds of the paths that reach the node (top-down, the `ect` fields) or leave it (bottom-up,
/// the `lst` fields). See the [module documentation](self).
///
/// One struct serves both directions, each one touching its own fields only. The fields of the
/// other direction hold the neutral element of their merge, so that merging stays a join.
#[derive(Clone, deepsize::DeepSizeOf)]
struct DisjunctiveProperty {
    /// The data of the jobs, shared with the constraint.
    jobs: Arc<Jobs>,
    /// $(A, S)$: the jobs placed on all, respectively on some, of the paths.
    permutation: AllDifferentProperty,
    /// Top-down: the smallest earliest completion time over the paths (best case).
    ect_best: isize,
    /// Top-down: the largest earliest completion time over the paths (worst case).
    ect_worst: isize,
    /// Bottom-up: the largest latest start time over the paths (best case).
    lst_best: isize,
    /// Bottom-up: the smallest latest start time over the paths (worst case).
    lst_worst: isize,
    /// With setup times only: what is known of the job at the open end of the paths, which is the
    /// last job placed (top-down) or the first job to place (bottom-up).
    open: Open,
}

impl DisjunctiveProperty {
    /// True if no path has contributed to the top-down bounds (the sentinels of the identity).
    fn has_no_ect(&self) -> bool {
        self.ect_best > self.ect_worst
    }

    /// True if no path has contributed to the bottom-up bounds.
    fn has_no_lst(&self) -> bool {
        self.lst_best < self.lst_worst
    }

    /// The property behind `other`, which must be a `DisjunctiveProperty` too.
    fn downcast<'a>(other: &'a dyn ConstraintProperty, action: &str) -> &'a Self {
        other.as_any().downcast_ref::<Self>().unwrap_or_else(|| {
            panic!(
                "Calling {action} on property DisjunctiveProperty with other property of type {}",
                other.name()
            );
        })
    }
}

/// The job of an assignment, or `None` if the value is not one of the `n` jobs.
fn job_of(assignment: isize, n: usize) -> Option<usize> {
    usize::try_from(assignment).ok().filter(|&job| job < n)
}

/// The constraint that the scope is a sequence of all the jobs that respects their time windows.
/// See the [module documentation](self) for the semantics and for how it is compiled.
#[derive(Clone, deepsize::DeepSizeOf)]
pub struct Disjunctive {
    /// Scope of the constraint: the variable at position `k` holds the job processed `k`-th.
    variables: Vec<VariableIndex>,
    /// The jobs.
    jobs: Arc<Jobs>,
    /// The permutation part, over the same scope.
    all_different: AllDifferent,
    /// Bitset telling if a layer is in the scope of the constraint. Empty until
    /// `update_variable_ordering` is called, which is how a missing ordering is detected.
    layer_in_scope: Vec<u64>,
}

impl Disjunctive {
    /// Builds the constraint over `variables`, one per job. Job $j$ has release date
    /// `release[j]`, deadline `deadline[j]` (latest completion time) and duration `duration[j]`.
    ///
    /// `precedences` lists pairs `(i, j)` of jobs, `i` having to come before `j`; `None` is no
    /// precedence at all.
    ///
    /// `setup_times` holds the matrix of the setup times, `matrix[i][j]` being the time that has
    /// to pass between the end of `i` and the start of `j` when `j` follows `i`, and the
    /// [`SetupRelaxation`] that tells how the states keep track of the job that the setup depends
    /// on; `None` is no setup time. The diagonal is ignored.
    ///
    /// `problem` is read for the domains of the variables, as in [`AllDifferent::new`].
    ///
    /// # Panics
    ///
    /// If a variable is repeated, if the three lists do not have one entry per variable, if a
    /// duration is negative, if the precedences are not valid (see
    /// [`check_precedences`](Self::check_precedences)), or if the setup times are not valid (see
    /// [`check_setup_times`](Self::check_setup_times)).
    pub fn new(
        variables: Vec<VariableIndex>,
        release: Vec<isize>,
        deadline: Vec<isize>,
        duration: Vec<isize>,
        precedences: Option<Vec<(usize, usize)>>,
        setup_times: Option<SetupTimes>,
        problem: &Problem,
    ) -> Self {
        let distinct: FxHashSet<VariableIndex> = variables.iter().copied().collect();
        if distinct.len() != variables.len() {
            panic!("Disjunctive does not support a variable repeated in its scope");
        }
        let n = variables.len();
        for (name, list) in [
            ("release", &release),
            ("deadline", &deadline),
            ("duration", &duration),
        ] {
            if list.len() != n {
                panic!(
                    "Disjunctive: {name} has {} entries for a scope of {n} variables",
                    list.len()
                );
            }
        }
        if let Some(job) = duration.iter().position(|&p| p < 0) {
            panic!("Disjunctive: job {job} has a negative duration");
        }
        let mut precedences = precedences.unwrap_or_default();
        if let Err(message) = Self::check_precedences(n, &precedences) {
            panic!("Disjunctive: {message}");
        }
        precedences.sort_unstable();
        precedences.dedup();
        let setup = setup_times.and_then(
            |SetupTimes {
                 matrix: rows,
                 relaxation,
             }| {
                if let Err(message) = Self::check_setup_times(n, &rows) {
                    panic!("Disjunctive: {message}");
                }
                // The matrix without its diagonal, flattened. A matrix with no positive entry is no
                // setup time at all.
                let times: Vec<isize> = (0..n * n)
                    .map(|cell| {
                        if cell / n == cell % n {
                            0
                        } else {
                            rows[cell / n][cell % n]
                        }
                    })
                    .collect();
                times
                    .iter()
                    .any(|&t| t > 0)
                    .then(|| Setup::new(n, times, relaxation))
            },
        );
        let all_different = AllDifferent::new(variables.clone(), problem);
        let mut predecessors = vec![vec![]; n];
        let mut successors = vec![vec![]; n];
        let mut predecessor_bits = vec![vec![]; n];
        let mut successor_bits = vec![vec![]; n];
        for &(before, after) in precedences.iter() {
            predecessors[after].push(before);
            successors[before].push(after);
            predecessor_bits[after].push(all_different.bit_of(before as isize));
            successor_bits[before].push(all_different.bit_of(after as isize));
        }
        Self {
            variables,
            jobs: Arc::new(Jobs {
                release,
                deadline,
                duration,
                precedences,
                predecessors,
                successors,
                predecessor_bits,
                successor_bits,
                setup,
            }),
            all_different,
            layer_in_scope: vec![],
        }
    }

    /// Checks that `precedences` are a valid set of precedences between `n` jobs: each one is
    /// between two different jobs of `0..n`, and together they have no cycle (which no sequence
    /// could respect). Returns the reason if not.
    pub fn check_precedences(n: usize, precedences: &[(usize, usize)]) -> Result<(), String> {
        for &(before, after) in precedences {
            if before >= n || after >= n {
                return Err(format!(
                    "the precedence ({before}, {after}) names a job that is not in 0..{n}"
                ));
            }
            if before == after {
                return Err(format!("job {before} cannot precede itself"));
            }
        }
        // Kahn's algorithm: a cycle is what stays when no job without predecessor is left.
        let mut missing = vec![0usize; n];
        let mut successors = vec![vec![]; n];
        for &(before, after) in precedences {
            missing[after] += 1;
            successors[before].push(after);
        }
        let mut ready: Vec<usize> = (0..n).filter(|&job| missing[job] == 0).collect();
        let mut done = 0;
        while let Some(job) = ready.pop() {
            done += 1;
            for &next in &successors[job] {
                missing[next] -= 1;
                if missing[next] == 0 {
                    ready.push(next);
                }
            }
        }
        if done < n {
            return Err("the precedences contain a cycle".to_string());
        }
        Ok(())
    }

    /// Checks that `setup_times` is a valid matrix of setup times between `n` jobs: `n` rows of
    /// `n` entries, none negative outside of the diagonal (which is ignored). Returns the reason
    /// if not.
    pub fn check_setup_times(n: usize, setup_times: &[Vec<isize>]) -> Result<(), String> {
        if setup_times.len() != n {
            return Err(format!(
                "the setup times have {} rows for {n} jobs",
                setup_times.len()
            ));
        }
        for (i, row) in setup_times.iter().enumerate() {
            if row.len() != n {
                return Err(format!(
                    "row {i} of the setup times has {} entries for {n} jobs",
                    row.len()
                ));
            }
            if let Some(j) = (0..n).find(|&j| j != i && row[j] < 0) {
                return Err(format!(
                    "the setup time from job {i} to job {j} is negative"
                ));
            }
        }
        Ok(())
    }

    /// Number of jobs, which is the size of the scope.
    fn len(&self) -> usize {
        self.variables.len()
    }

    /// With setup times, the earliest time at which `job` can start after the paths above (setup
    /// included, but not the release date) and the latest time at which it can end given the
    /// paths below (setup included, but not the deadline). The best case of each relaxation.
    fn setup_window(
        setup: &Setup,
        parent: &DisjunctiveProperty,
        child: &DisjunctiveProperty,
        job: usize,
    ) -> (isize, isize) {
        match (&parent.open, &child.open) {
            (Open::Job(last), Open::Job(first)) => (
                parent.ect_best.saturating_add(setup.before_min(*last, job)),
                child.lst_best.saturating_sub(setup.after_min(job, *first)),
            ),
            (Open::Set(last), Open::Set(first)) => (
                parent
                    .ect_best
                    .saturating_add(setup.set_before(last, job, false)),
                child
                    .lst_best
                    .saturating_sub(setup.set_after(first, job, false)),
            ),
            // A node that no path can follow with this job gets the values that make the edge
            // invalid, as the sentinels of an empty node do.
            (Open::Arcs(last), Open::Arcs(first)) => (
                last.ready(setup, job).map_or(isize::MAX, |(best, _)| best),
                first.due(setup, job).map_or(isize::MIN, |(best, _)| best),
            ),
            _ => Open::mismatch(),
        }
    }
}

impl Constraint for Disjunctive {
    fn structural_key(&self, _problem: &Problem) -> ConstraintShapeKey {
        ConstraintShapeKey::Disjunctive {
            arity: self.len(),
            release: self.jobs.release.clone(),
            deadline: self.jobs.deadline.clone(),
            duration: self.jobs.duration.clone(),
            precedences: self.jobs.precedences.clone(),
            setup_times: self.jobs.setup.as_ref().map(|setup| setup.times.clone()),
            setup_relaxation: self
                .jobs
                .setup
                .as_ref()
                .map(|setup| setup.relaxation)
                .unwrap_or_default(),
        }
    }

    fn update_variable_ordering(&mut self, order: &[VariableIndex]) {
        let scope: FxHashSet<VariableIndex> = self.variables.iter().copied().collect();
        self.layer_in_scope = vec![0; order.len() / 64 + 1];
        let mut observed_order: Vec<VariableIndex> = Vec::with_capacity(self.variables.len());
        for (layer, &variable) in order.iter().enumerate() {
            if scope.contains(&variable) {
                // Sets the bit of the layer to 1
                self.layer_in_scope[layer / 64] |= 1 << (layer % 64);
                observed_order.push(variable);
            }
        }
        // `Mdd::new` repairs the heuristic's order to satisfy `precedence_edges` (below) before
        // calling this, so this should be unreachable in practice: it is a safety net.
        if observed_order != self.variables {
            panic!(
                "Disjunctive constraint's variables must keep their declared relative order in the chosen variable ordering"
            );
        }
        self.all_different.update_variable_ordering(order);
    }

    /// The position of a job is the position of its variable in the scope, so the variables must
    /// be branched in the order they were declared in: variable `k` before variable `k+1`.
    fn precedence_edges(&self) -> Vec<(VariableIndex, VariableIndex)> {
        self.variables
            .windows(2)
            .map(|pair| (pair[0], pair[1]))
            .collect()
    }

    fn is_layer_in_scope(&self, layer: usize) -> bool {
        debug_assert!(
            !self.layer_in_scope.is_empty(),
            "update_variable_ordering has not been called"
        );
        self.layer_in_scope[layer / 64] & (1 << (layer % 64)) != 0
    }

    fn iter_scope(&self) -> Box<dyn Iterator<Item = VariableIndex> + '_> {
        Box::new(self.variables.iter().copied())
    }

    fn is_satisfied(&self, assignment: &[isize]) -> bool {
        // `assignment` is indexed by variable index and must cover the scope.
        let n = self.len();
        // The jobs placed so far, one bit each (on the stack up to 64 jobs). A job is placed
        // once, after its predecessors: that is the permutation and the precedences, checked in
        // one pass that stops at the first fault.
        let mut inline = [0u64; 1];
        let mut heap;
        let placed: &mut [u64] = if n <= 64 {
            &mut inline
        } else {
            heap = vec![0u64; n.div_ceil(64)];
            &mut heap
        };
        let is_placed = |placed: &[u64], job: usize| placed[job / 64] & (1 << (job % 64)) != 0;
        let has_precedences = !self.jobs.precedences.is_empty();
        let setup = self.jobs.setup.as_ref();
        let mut previous = None;
        let mut time = isize::MIN;
        for &variable in self.variables.iter() {
            let Some(job) = job_of(assignment[*variable], n) else {
                return false;
            };
            if is_placed(placed, job)
                || (has_precedences
                    && self.jobs.predecessors[job]
                        .iter()
                        .any(|&before| !is_placed(placed, before)))
            {
                return false;
            }
            placed[job / 64] |= 1 << (job % 64);
            let mut ready = time;
            if let Some(setup) = setup {
                if let Some(previous) = previous {
                    ready = time.saturating_add(setup.times[previous * n + job]);
                }
                previous = Some(job);
            }
            let end = ready
                .max(self.jobs.release[job])
                .saturating_add(self.jobs.duration[job]);
            if end > self.jobs.deadline[job] {
                return false;
            }
            time = end;
        }
        true
    }

    fn name(&self) -> &'static str {
        "Disjunctive"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn is_assignment_invalid(
        &self,
        parent: &dyn ConstraintProperty,
        child: &dyn ConstraintProperty,
        layer: usize,
        assignment: isize,
    ) -> bool {
        let parent = DisjunctiveProperty::downcast(parent, "is_assignment_invalid (parent)");
        let child = DisjunctiveProperty::downcast(child, "is_assignment_invalid (child)");
        // A value that is not a job is never allowed.
        let Some(job) = job_of(assignment, self.len()) else {
            return true;
        };

        // The rules are tried from the cheapest to the most expensive, which does not change the
        // answer: the edge is invalid if any of them says so.
        //
        // Rule 3: the job cannot end in time. A node with no path (the sentinels) fails here
        // too: its best end saturates to the largest value, or its latest start is the smallest.
        let mut from = parent.ect_best;
        let mut until = child.lst_best;
        if let Some(setup) = &self.jobs.setup {
            (from, until) = Self::setup_window(setup, parent, child, job);
        }
        let end = from
            .max(self.jobs.release[job])
            .saturating_add(self.jobs.duration[job]);
        if end > until.min(self.jobs.deadline[job]) {
            return true;
        }
        // Rule 1: the job is already placed, or the Hall sets are full.
        // Rule 2: some job would be on no path through the edge, and every job has to be placed.
        // Both read the same bit and the same union of the sets, so they are tested together.
        if self.all_different.is_assignment_invalid_covering(
            &parent.permutation,
            &child.permutation,
            layer,
            assignment,
            self.len(),
        ) {
            return true;
        }
        // Rule 4: a predecessor of the job is on no path above, or a successor is on no path
        // below.
        if self.jobs.precedences.is_empty() {
            return false;
        }
        self.jobs.predecessor_bits[job]
            .iter()
            .any(|&before| !parent.permutation.is_some_bit(before))
            || self.jobs.successor_bits[job]
                .iter()
                .any(|&after| !child.permutation.is_some_bit(after))
    }

    fn identity_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(DisjunctiveProperty {
            jobs: self.jobs.clone(),
            permutation: self.all_different.identity(),
            ect_best: isize::MAX,
            ect_worst: isize::MIN,
            lst_best: isize::MIN,
            lst_worst: isize::MAX,
            open: Open::identity(&self.jobs.setup),
        })
    }

    /// Nothing is placed at the root, which is not before any time ($-\infty$); and nothing
    /// remains at the sink, which is not after any time ($+\infty$). The two bounds that the
    /// other direction would read are not used.
    fn empty_property(&self) -> Box<dyn ConstraintProperty> {
        Box::new(DisjunctiveProperty {
            jobs: self.jobs.clone(),
            permutation: self.all_different.empty(),
            ect_best: isize::MIN,
            ect_worst: isize::MIN,
            lst_best: isize::MAX,
            lst_worst: isize::MAX,
            open: Open::empty(&self.jobs.setup),
        })
    }
}

impl ConstraintProperty for DisjunctiveProperty {
    fn update(&mut self, parent: &dyn ConstraintProperty, assignment: isize, in_scope: bool) {
        let parent = Self::downcast(parent, "update");
        if !in_scope {
            self.permutation
                .update(&parent.permutation, assignment, false);
            self.ect_best = self.ect_best.min(parent.ect_best);
            self.ect_worst = self.ect_worst.max(parent.ect_worst);
            self.open.merge(&parent.open);
            return;
        }
        // An edge that is not a job is removed by the filtering; it contributes nothing.
        let Some(job) = job_of(assignment, self.jobs.release.len()) else {
            return;
        };
        self.permutation
            .update(&parent.permutation, assignment, true);
        let mut best = parent.ect_best;
        let mut worst = parent.ect_worst;
        if let Some(setup) = &self.jobs.setup {
            match (&mut self.open, &parent.open) {
                (Open::Job(mine), Open::Job(last)) => {
                    *mine = join_job(*mine, job as u32);
                    best = best.saturating_add(setup.before_min(*last, job));
                    worst = worst.saturating_add(setup.before_max(*last, job));
                }
                (Open::Set(mine), Open::Set(last)) => {
                    set_bit(mine, job);
                    best = best.saturating_add(setup.set_before(last, job, false));
                    worst = worst.saturating_add(setup.set_before(last, job, true));
                }
                (Open::Arcs(mine), Open::Arcs(last)) => {
                    // The paths that end with `job` are all in its slot, whose bounds are the
                    // ones of the scalars, now computed from the slots of the parent.
                    if let Some((best, worst)) = last.ready(setup, job) {
                        let release = self.jobs.release[job];
                        let duration = self.jobs.duration[job];
                        let best = best.max(release).saturating_add(duration);
                        let worst = worst.max(release).saturating_add(duration);
                        mine.ect_best[job] = mine.ect_best[job].min(best);
                        mine.ect_worst[job] = mine.ect_worst[job].max(worst);
                        self.ect_best = self.ect_best.min(best);
                        self.ect_worst = self.ect_worst.max(worst);
                    }
                    return;
                }
                _ => Open::mismatch(),
            }
        }
        if !parent.has_no_ect() {
            let release = self.jobs.release[job];
            let duration = self.jobs.duration[job];
            self.ect_best = self
                .ect_best
                .min(best.max(release).saturating_add(duration));
            self.ect_worst = self
                .ect_worst
                .max(worst.max(release).saturating_add(duration));
        }
    }

    fn update_backward(
        &mut self,
        child: &dyn ConstraintProperty,
        assignment: isize,
        in_scope: bool,
    ) {
        let child = Self::downcast(child, "update_backward");
        if !in_scope {
            self.permutation
                .update_backward(&child.permutation, assignment, false);
            self.lst_best = self.lst_best.max(child.lst_best);
            self.lst_worst = self.lst_worst.min(child.lst_worst);
            self.open.merge(&child.open);
            return;
        }
        let Some(job) = job_of(assignment, self.jobs.release.len()) else {
            return;
        };
        self.permutation
            .update_backward(&child.permutation, assignment, true);
        let mut best = child.lst_best;
        let mut worst = child.lst_worst;
        if let Some(setup) = &self.jobs.setup {
            match (&mut self.open, &child.open) {
                (Open::Job(mine), Open::Job(first)) => {
                    *mine = join_job(*mine, job as u32);
                    best = best.saturating_sub(setup.after_min(job, *first));
                    worst = worst.saturating_sub(setup.after_max(job, *first));
                }
                (Open::Set(mine), Open::Set(first)) => {
                    set_bit(mine, job);
                    best = best.saturating_sub(setup.set_after(first, job, false));
                    worst = worst.saturating_sub(setup.set_after(first, job, true));
                }
                (Open::Arcs(mine), Open::Arcs(first)) => {
                    if let Some((best, worst)) = first.due(setup, job) {
                        let deadline = self.jobs.deadline[job];
                        let duration = self.jobs.duration[job];
                        let best = best.min(deadline).saturating_sub(duration);
                        let worst = worst.min(deadline).saturating_sub(duration);
                        mine.lst_best[job] = mine.lst_best[job].max(best);
                        mine.lst_worst[job] = mine.lst_worst[job].min(worst);
                        self.lst_best = self.lst_best.max(best);
                        self.lst_worst = self.lst_worst.min(worst);
                    }
                    return;
                }
                _ => Open::mismatch(),
            }
        }
        if !child.has_no_lst() {
            let deadline = self.jobs.deadline[job];
            let duration = self.jobs.duration[job];
            self.lst_best = self
                .lst_best
                .max(best.min(deadline).saturating_sub(duration));
            self.lst_worst = self
                .lst_worst
                .min(worst.min(deadline).saturating_sub(duration));
        }
    }

    fn merge(&mut self, other: &dyn ConstraintProperty) {
        let other = Self::downcast(other, "merge");
        self.permutation.merge(&other.permutation);
        self.ect_best = self.ect_best.min(other.ect_best);
        self.ect_worst = self.ect_worst.max(other.ect_worst);
        self.lst_best = self.lst_best.max(other.lst_best);
        self.lst_worst = self.lst_worst.min(other.lst_worst);
        self.open.merge(&other.open);
    }

    fn order_key(&self) -> Vec<f64> {
        // One allocation of the right size: this runs for every node when merging.
        let [all, some] = self.permutation.order_key_parts();
        let mut key = Vec::with_capacity(8);
        key.extend([
            all,
            some,
            self.ect_best as f64,
            self.ect_worst as f64,
            self.lst_best as f64,
            self.lst_worst as f64,
        ]);
        self.open.push_order_key(self.jobs.release.len(), &mut key);
        key
    }

    fn hash(&self, hasher: &mut dyn Hasher) {
        self.permutation.hash(hasher);
        hasher.write_isize(self.ect_best);
        hasher.write_isize(self.ect_worst);
        hasher.write_isize(self.lst_best);
        hasher.write_isize(self.lst_worst);
        self.open.hash(hasher);
    }

    fn eq(&self, other: &dyn ConstraintProperty) -> bool {
        let other = Self::downcast(other, "eq");
        ConstraintProperty::eq(&self.permutation, &other.permutation)
            && self.ect_best == other.ect_best
            && self.ect_worst == other.ect_worst
            && self.lst_best == other.lst_best
            && self.lst_worst == other.lst_worst
            && self.open.eq(&other.open)
    }

    /// $A = S$ and each time bound that holds a value has its two cases equal (see the module
    /// documentation).
    fn is_single_state(&self) -> bool {
        self.permutation.is_single_state()
            && (self.has_no_ect() || self.ect_best == self.ect_worst)
            && (self.has_no_lst() || self.lst_best == self.lst_worst)
            && self.open.is_single_state()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "DisjunctiveProperty"
    }
}

#[cfg(test)]
mod test_disjunctive {
    use crate::constraints::*;
    use crate::mdd::compiler::test_mdd::*;
    use crate::mdd::heuristics::*;
    use crate::mdd::*;
    use crate::modelling::*;
    use rustc_hash::FxHasher;
    use std::hash::Hasher;
    use std::sync::Arc;

    // ----------------------------------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------------------------------

    /// The jobs of a test: `(release, deadline, duration)` of each, in this order.
    type Spec = Vec<(isize, isize, isize)>;

    /// Two jobs that have to be first or second, and a free one; job 1 has to end before 4.
    fn tight() -> Spec {
        vec![(0, 10, 3), (0, 4, 2), (0, 10, 2)]
    }

    fn lists(spec: &Spec) -> (Vec<isize>, Vec<isize>, Vec<isize>) {
        (
            spec.iter().map(|j| j.0).collect(),
            spec.iter().map(|j| j.1).collect(),
            spec.iter().map(|j| j.2).collect(),
        )
    }

    /// `n` variables over the jobs `0..n`, and the constraint over them with the identity
    /// ordering already set.
    fn disj(spec: &Spec) -> Disjunctive {
        let n = spec.len() as isize;
        let mut problem = Problem::default();
        let vars = problem.add_variables(spec.len(), (0..n).collect());
        let (r, d, p) = lists(spec);
        let mut c = Disjunctive::new(vars.clone(), r, d, p, None, None, &problem);
        c.update_variable_ordering(&vars);
        c
    }

    /// The problem with only the constraint over its `n` variables.
    fn problem_of(spec: &Spec) -> Problem {
        let n = spec.len() as isize;
        let mut problem = Problem::default();
        let vars = problem.add_variables(spec.len(), (0..n).collect());
        let (r, d, p) = lists(spec);
        disjunctive(&mut problem, vars, r, d, p, None, None);
        problem
    }

    fn compile(problem: Problem, order: Vec<usize>, width: usize) -> Mdd {
        let problem = Arc::new(problem);
        let constraints: Vec<ConstraintIndex> = problem.iter_constraints().collect();
        let mut compiler = Compiler::new(
            problem,
            CompilationHeuristic {
                ordering: OrderingHeuristic::Custom(order),
                merge: MergeHeuristic::LessRelaxed,
                selection: SelectHeuristic::Greedy,
            },
            &constraints,
        );
        if width > 1 {
            compiler.refine(width, usize::MAX);
        }
        compiler.into_mdd()
    }

    fn accepted(mdd: &Mdd) -> Vec<Vec<isize>> {
        if mdd.is_unsat() {
            return vec![];
        }
        let mut solutions = get_all_solutions(mdd);
        solutions.sort();
        solutions
    }

    /// The permutations of `0..n` that can be scheduled, found by a search that does not use
    /// the constraint: jobs are added one at a time, keeping the time at which the machine is
    /// free.
    fn expected(spec: &Spec) -> Vec<Vec<isize>> {
        fn extend(
            spec: &Spec,
            free_at: isize,
            used: &mut Vec<bool>,
            prefix: &mut Vec<isize>,
            out: &mut Vec<Vec<isize>>,
        ) {
            if prefix.len() == spec.len() {
                out.push(prefix.clone());
                return;
            }
            for job in 0..spec.len() {
                if used[job] {
                    continue;
                }
                let (release, deadline, duration) = spec[job];
                let start = if free_at > release { free_at } else { release };
                if start + duration > deadline {
                    continue;
                }
                used[job] = true;
                prefix.push(job as isize);
                extend(spec, start + duration, used, prefix, out);
                prefix.pop();
                used[job] = false;
            }
        }
        let mut out = vec![];
        extend(
            spec,
            isize::MIN / 2,
            &mut vec![false; spec.len()],
            &mut vec![],
            &mut out,
        );
        out.sort();
        out
    }

    /// The property of a single path that places `jobs` in this order, top-down, built the way the
    /// compiler builds it.
    fn top_down(c: &Disjunctive, jobs: &[isize]) -> Box<dyn ConstraintProperty> {
        let mut property = c.empty_property();
        for &job in jobs {
            let mut next = c.identity_property();
            next.update(&*property, job, true);
            property = next;
        }
        property
    }

    /// The property of a single path that places `jobs` in this order, read from the sink.
    fn bottom_up(c: &Disjunctive, jobs: &[isize]) -> Box<dyn ConstraintProperty> {
        let mut property = c.empty_property_backward();
        for &job in jobs.iter().rev() {
            let mut next = c.identity_property();
            next.update_backward(&*property, job, true);
            property = next;
        }
        property
    }

    fn merged(properties: &[Box<dyn ConstraintProperty>]) -> Box<dyn ConstraintProperty> {
        let mut result = dyn_clone::clone_box(&*properties[0]);
        for other in &properties[1..] {
            result.merge(&**other);
        }
        result
    }

    /// `[ect_best, ect_worst, lst_best, lst_worst]`, the last four numbers of the order key.
    fn times(property: &dyn ConstraintProperty) -> [isize; 4] {
        let key = property.order_key();
        // With setup times the key ends with the job at the open end.
        let n = key.len().min(6);
        [
            key[n - 4] as isize,
            key[n - 3] as isize,
            key[n - 2] as isize,
            key[n - 1] as isize,
        ]
    }

    fn hash_of(property: &dyn ConstraintProperty) -> u64 {
        let mut hasher = FxHasher::default();
        property.hash(&mut hasher);
        hasher.finish()
    }

    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, bound: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % bound
        }
    }

    /// `n` random jobs: small durations, windows that are sometimes tight.
    fn random_spec(rng: &mut Lcg, n: usize) -> Spec {
        let total = 2 * n as u64;
        (0..n)
            .map(|_| {
                let release = rng.below(total) as isize;
                let duration = 1 + rng.below(4) as isize;
                let slack = rng.below(total + 2) as isize;
                (release, release + duration + slack, duration)
            })
            .collect()
    }

    fn all_tuples(n: usize, values: isize) -> Vec<Vec<isize>> {
        let mut all: Vec<Vec<isize>> = vec![vec![]];
        for _ in 0..n {
            all = all
                .into_iter()
                .flat_map(|prefix| {
                    (0..values).map(move |v| {
                        let mut t = prefix.clone();
                        t.push(v);
                        t
                    })
                })
                .collect();
        }
        all
    }

    // ----------------------------------------------------------------------------------------
    // Construction, scope and key
    // ----------------------------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "repeated")]
    fn a_repeated_variable_is_refused() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1]);
        let y = problem.add_variable(vec![0, 1]);
        Disjunctive::new(
            vec![x, y, x],
            vec![0; 3],
            vec![9; 3],
            vec![1; 3],
            None,
            None,
            &problem,
        );
    }

    #[test]
    #[should_panic(expected = "deadline has 1 entries for a scope of 2")]
    fn a_list_of_the_wrong_length_is_refused() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1]);
        Disjunctive::new(vars, vec![0, 0], vec![9], vec![1, 1], None, None, &problem);
    }

    #[test]
    #[should_panic(expected = "job 1 has a negative duration")]
    fn a_negative_duration_is_refused() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1]);
        Disjunctive::new(
            vars,
            vec![0, 0],
            vec![9, 9],
            vec![1, -1],
            None,
            None,
            &problem,
        );
    }

    #[test]
    fn scope_name_and_structural_key() {
        let spec = tight();
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        let (r, d, p) = lists(&spec);
        let c = Disjunctive::new(
            vars.clone(),
            r.clone(),
            d.clone(),
            p.clone(),
            None,
            None,
            &problem,
        );
        assert_eq!(c.iter_scope().collect::<Vec<_>>(), vars);
        assert_eq!(c.name(), "Disjunctive");
        assert_eq!(
            c.structural_key(&problem),
            ConstraintShapeKey::Disjunctive {
                arity: 3,
                release: r,
                deadline: d,
                duration: p,
                precedences: vec![],
                setup_times: None,
                setup_relaxation: SetupRelaxation::LastJob,
            }
        );
    }

    #[test]
    fn the_scope_is_read_as_a_sequence() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        let c = Disjunctive::new(
            vars.clone(),
            vec![0; 3],
            vec![9; 3],
            vec![1; 3],
            None,
            None,
            &problem,
        );
        assert_eq!(
            c.precedence_edges(),
            vec![(vars[0], vars[1]), (vars[1], vars[2])]
        );
    }

    #[test]
    fn layers_in_scope_follow_the_variable_ordering() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1]);
        let mut c = Disjunctive::new(
            vec![vars[1], vars[3]],
            vec![0; 2],
            vec![9; 2],
            vec![1; 2],
            None,
            None,
            &problem,
        );
        c.update_variable_ordering(&[vars[0], vars[1], vars[2], vars[3]]);
        assert!(!c.is_layer_in_scope(0));
        assert!(c.is_layer_in_scope(1));
        assert!(!c.is_layer_in_scope(2));
        assert!(c.is_layer_in_scope(3));
    }

    #[test]
    #[should_panic(expected = "declared relative order")]
    fn an_ordering_that_swaps_the_scope_is_refused() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1]);
        let mut c = Disjunctive::new(
            vars.clone(),
            vec![0; 2],
            vec![9; 2],
            vec![1; 2],
            None,
            None,
            &problem,
        );
        c.update_variable_ordering(&[vars[1], vars[0]]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn is_layer_in_scope_panics_before_the_ordering_is_set() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1]);
        Disjunctive::new(
            vars,
            vec![0; 2],
            vec![9; 2],
            vec![1; 2],
            None,
            None,
            &problem,
        )
        .is_layer_in_scope(0);
    }

    // ----------------------------------------------------------------------------------------
    // is_satisfied: the specification, with no MDD involved
    // ----------------------------------------------------------------------------------------

    #[test]
    fn is_satisfied_agrees_with_the_search_on_every_tuple() {
        let mut rng = Lcg(2024);
        for _ in 0..60 {
            let n = 1 + rng.below(4) as usize;
            let spec = random_spec(&mut rng, n);
            let c = disj(&spec);
            let feasible = expected(&spec);
            // Values from -1 to n, so that values that are not jobs are covered.
            for tuple in all_tuples(n, n as isize + 2) {
                let tuple: Vec<isize> = tuple.iter().map(|v| v - 1).collect();
                assert_eq!(
                    c.is_satisfied(&tuple),
                    feasible.contains(&tuple),
                    "{spec:?} on {tuple:?}"
                );
            }
        }
    }

    #[test]
    fn is_satisfied_on_hand_written_cases() {
        // Job 1 ends before 4: only these three orders work (see `tight`).
        let c = disj(&tight());
        assert!(c.is_satisfied(&[1, 0, 2]));
        assert!(c.is_satisfied(&[1, 2, 0]));
        assert!(c.is_satisfied(&[2, 1, 0]));
        assert!(!c.is_satisfied(&[0, 1, 2])); // job 1 ends at 5
        assert!(!c.is_satisfied(&[2, 0, 1])); // job 1 ends at 7
        assert!(!c.is_satisfied(&[1, 1, 2])); // not a permutation
        assert!(!c.is_satisfied(&[1, 0, 3])); // not a job
        assert!(!c.is_satisfied(&[1, 0, -1]));
    }

    #[test]
    fn a_job_waits_for_its_release_date() {
        // Job 0 is released at 5 and takes 2, job 1 has to end by 8.
        let spec = vec![(5, 20, 2), (0, 8, 2)];
        let c = disj(&spec);
        assert!(c.is_satisfied(&[1, 0])); // 0..2, then 5..7
        assert!(!c.is_satisfied(&[0, 1])); // 5..7, then 7..9: too late for job 1
        assert_eq!(expected(&spec), vec![vec![1, 0]]);
    }

    #[test]
    fn a_zero_duration_job_needs_no_time() {
        let spec = vec![(0, 0, 0), (0, 2, 2)];
        let c = disj(&spec);
        assert!(c.is_satisfied(&[0, 1]));
        // Job 0 would start at 2, after its deadline, even though it takes no time.
        assert!(!c.is_satisfied(&[1, 0]));
    }

    #[test]
    fn empty_and_single_scopes() {
        let problem = Problem::default();
        let empty = Disjunctive::new(vec![], vec![], vec![], vec![], None, None, &problem);
        assert!(empty.is_satisfied(&[]));
        let c = disj(&vec![(2, 5, 3)]);
        assert!(c.is_satisfied(&[0]));
        let late = disj(&vec![(2, 4, 3)]);
        assert!(!late.is_satisfied(&[0]));
    }

    #[test]
    fn huge_dates_do_not_overflow() {
        // The end of the first job saturates instead of wrapping around, and is late.
        let spec = vec![(isize::MAX - 1, isize::MAX - 1, 5), (0, 3, 1)];
        let c = disj(&spec);
        assert!(!c.is_satisfied(&[0, 1]));
        assert!(!c.is_satisfied(&[1, 0]));
    }

    // ----------------------------------------------------------------------------------------
    // The property, without any MDD
    // ----------------------------------------------------------------------------------------

    #[test]
    fn the_earliest_completion_of_a_path_is_the_schedule() {
        // Jobs of `tight`, then job 0 released at 2: 3 + 2 ... checked by hand.
        let spec = vec![(2, 20, 3), (0, 14, 4), (1, 14, 2)];
        let c = disj(&spec);
        assert_eq!(
            times(&*top_down(&c, &[])),
            [isize::MIN, isize::MIN, isize::MAX, isize::MAX]
        );
        // Job 2 starts at 1, ends at 3. Then job 1 starts at 3, ends at 7. Then job 0 at 7..10.
        assert_eq!(times(&*top_down(&c, &[2]))[..2], [3, 3]);
        assert_eq!(times(&*top_down(&c, &[2, 1]))[..2], [7, 7]);
        assert_eq!(times(&*top_down(&c, &[2, 1, 0]))[..2], [10, 10]);
        // Job 0 first waits for its release date: 2..5.
        assert_eq!(times(&*top_down(&c, &[0]))[..2], [5, 5]);
    }

    #[test]
    fn the_latest_start_of_a_path_is_the_schedule_read_backwards() {
        let spec = vec![(2, 20, 3), (0, 14, 4), (1, 14, 2)];
        let c = disj(&spec);
        // Nothing after: no limit.
        assert_eq!(times(&*bottom_up(&c, &[]))[2..], [isize::MAX, isize::MAX]);
        // Job 1 last: starts by 14 - 4 = 10. Job 2 before it: ends by 10, so starts by 8.
        assert_eq!(times(&*bottom_up(&c, &[1]))[2..], [10, 10]);
        assert_eq!(times(&*bottom_up(&c, &[2, 1]))[2..], [8, 8]);
        // Job 0 before them: ends by 8, starts by 5.
        assert_eq!(times(&*bottom_up(&c, &[0, 2, 1]))[2..], [5, 5]);
        // The deadline of the job itself can be the binding one.
        assert_eq!(times(&*bottom_up(&c, &[0]))[2..], [17, 17]);
    }

    #[test]
    fn folding_two_parents_keeps_the_best_and_the_worst_case() {
        let spec = vec![(0, 50, 3), (0, 50, 5), (0, 50, 1)];
        let c = disj(&spec);
        // Jobs 0 and 1 in both orders end at 8; with job 2 after, 9. Job 0 alone ends at 3.
        let a = top_down(&c, &[0]);
        let b = top_down(&c, &[1]);
        let mut node = c.identity_property();
        node.update(&*a, 2, true);
        node.update(&*b, 2, true);
        // Through job 0: 3 + 1 = 4. Through job 1: 5 + 1 = 6.
        assert_eq!(times(&*node)[..2], [4, 6]);
        assert!(!node.is_single_state());
    }

    #[test]
    fn the_same_jobs_in_two_orders_are_not_one_state() {
        let spec = vec![(0, 50, 3), (4, 50, 5), (0, 50, 1)];
        let c = disj(&spec);
        // Job 1 is released at 4: [0, 1] ends at 9, [1, 0] ends at 9 + 3 = 12.
        let m = merged(&[top_down(&c, &[0, 1]), top_down(&c, &[1, 0])]);
        assert_eq!(times(&*m)[..2], [9, 12]);
        assert!(!m.is_single_state());
        // When the time agrees, it is one state.
        let spec = vec![(0, 50, 3), (0, 50, 5), (0, 50, 1)];
        let c = disj(&spec);
        let m = merged(&[top_down(&c, &[0, 1]), top_down(&c, &[1, 0])]);
        assert_eq!(times(&*m)[..2], [8, 8]);
        assert!(m.is_single_state());
    }

    #[test]
    fn the_empty_and_the_identity_properties() {
        let c = disj(&tight());
        assert!(c.empty_property().is_single_state());
        assert!(c.empty_property_backward().is_single_state());
        assert!(!c.identity_property().is_single_state());
        assert!(!ConstraintProperty::eq(
            &*c.identity_property(),
            &*c.empty_property()
        ));
    }

    #[test]
    fn identity_is_neutral_for_merge() {
        let c = disj(&tight());
        for property in [
            top_down(&c, &[1, 0]),
            bottom_up(&c, &[1, 0]),
            c.empty_property(),
        ] {
            let mut with_identity = dyn_clone::clone_box(&*property);
            with_identity.merge(&*c.identity_property());
            assert!(ConstraintProperty::eq(&*with_identity, &*property));
        }
    }

    #[test]
    fn eq_and_hash_see_the_sets_and_the_times() {
        let spec = vec![(0, 50, 3), (4, 50, 5), (0, 50, 1)];
        let c = disj(&spec);
        let p01 = top_down(&c, &[0, 1]);
        let p10 = top_down(&c, &[1, 0]);
        // The same jobs, a different time.
        assert!(!ConstraintProperty::eq(&*p01, &*p10));
        assert_ne!(hash_of(&*p01), hash_of(&*p10));
        assert!(ConstraintProperty::eq(&*p01, &*top_down(&c, &[0, 1])));
        assert_eq!(hash_of(&*p01), hash_of(&*top_down(&c, &[0, 1])));
        // The same time, different jobs: [2, 0] ends at 4 (job 2: 0..1, job 0: 1..4),
        // [0, 2] ends at 4 as well (0..3, 3..4).
        let a = top_down(&c, &[2, 0]);
        let b = top_down(&c, &[0, 2]);
        assert_eq!(times(&*a)[..2], times(&*b)[..2]);
        assert!(ConstraintProperty::eq(&*a, &*b));
        let c2 = top_down(&c, &[2, 1]);
        assert!(!ConstraintProperty::eq(&*a, &*c2));
    }

    #[test]
    fn out_of_scope_layers_copy_the_property() {
        let c = disj(&tight());
        let parent = top_down(&c, &[1, 0]);
        let mut child = c.identity_property();
        child.update(&*parent, 99, false);
        assert!(ConstraintProperty::eq(&*child, &*parent));
        let below = bottom_up(&c, &[0, 2]);
        let mut above = c.identity_property();
        above.update_backward(&*below, 99, false);
        assert!(ConstraintProperty::eq(&*above, &*below));
    }

    #[test]
    fn an_edge_that_is_not_a_job_contributes_nothing() {
        let c = disj(&tight());
        let parent = top_down(&c, &[1]);
        let mut child = c.identity_property();
        child.update(&*parent, 7, true);
        child.update(&*parent, -2, true);
        assert!(ConstraintProperty::eq(&*child, &*c.identity_property()));
    }

    #[test]
    #[should_panic(expected = "Calling update on property DisjunctiveProperty")]
    fn update_with_a_property_of_another_constraint_is_rejected() {
        let c = disj(&tight());
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1]);
        let foreign = NotEquals::new(vars[0], vars[1], &problem).identity_property();
        c.identity_property().update(&*foreign, 0, true);
    }

    #[test]
    #[should_panic(expected = "Calling merge on property DisjunctiveProperty")]
    fn merge_with_a_property_of_another_constraint_is_rejected() {
        let c = disj(&tight());
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1]);
        let foreign = NotEquals::new(vars[0], vars[1], &problem).identity_property();
        c.identity_property().merge(&*foreign);
    }

    #[test]
    fn the_order_key_has_a_fixed_length_and_is_finite() {
        let c = disj(&tight());
        let length = c.empty_property().order_key().len();
        for property in [
            c.identity_property(),
            c.empty_property(),
            top_down(&c, &[1, 2]),
            bottom_up(&c, &[1, 2]),
        ] {
            let key = property.order_key();
            assert_eq!(key.len(), length);
            assert!(key.iter().all(|x| x.is_finite()));
        }
    }

    // ----------------------------------------------------------------------------------------
    // is_assignment_invalid called directly on properties
    // ----------------------------------------------------------------------------------------

    /// Is job `job` at layer `layer` removed, given the jobs `above` (before) and `below` (after)?
    fn removed(c: &Disjunctive, above: &[isize], job: isize, below: &[isize]) -> bool {
        c.is_assignment_invalid(
            &*top_down(c, above),
            &*bottom_up(c, below),
            above.len(),
            job,
        )
    }

    #[test]
    fn a_job_that_cannot_end_in_time_is_removed() {
        // `tight`: job 1 ends by 4. After job 0 (ends at 3) it would end at 5.
        let c = disj(&tight());
        assert!(removed(&c, &[0], 1, &[2]));
        assert!(!removed(&c, &[2], 1, &[0]));
        assert!(!removed(&c, &[], 1, &[0, 2]));
    }

    #[test]
    fn a_job_that_makes_the_next_one_late_is_removed() {
        // Placing job 0 (3) first makes job 1 end at 5 > 4, which only the latest start of what
        // remains can tell: the edge of job 0 itself ends at 3 <= 10.
        let c = disj(&tight());
        assert!(removed(&c, &[], 0, &[1, 2]));
        assert!(removed(&c, &[], 0, &[2, 1]));
        // Job 1 first, then anything: fine.
        assert!(!removed(&c, &[], 1, &[0, 2]));
    }

    #[test]
    fn a_job_that_waits_for_its_release_date_ends_late() {
        let spec = vec![(5, 6, 2), (0, 20, 1)];
        let c = disj(&spec);
        // Job 0 cannot end before 7, whatever comes before it.
        assert!(removed(&c, &[], 0, &[1]));
        assert!(removed(&c, &[1], 0, &[]));
    }

    #[test]
    fn a_job_that_is_already_placed_is_removed() {
        let c = disj(&tight());
        assert!(removed(&c, &[1], 1, &[0]));
        assert!(removed(&c, &[2], 0, &[0]));
    }

    #[test]
    fn a_value_that_is_not_a_job_is_removed() {
        let c = disj(&tight());
        for value in [-1, 3, 17] {
            assert!(c.is_assignment_invalid(
                &*c.empty_property(),
                &*c.empty_property_backward(),
                0,
                value
            ));
        }
    }

    #[test]
    fn a_job_that_would_be_placed_nowhere_is_removed() {
        // Rule 2 on merged nodes: above, the paths place job 0 or job 1, never job 2; below,
        // the paths place job 0 or job 1. Placing job 0 here leaves job 2 nowhere.
        let spec = vec![(0, 99, 1), (0, 99, 1), (0, 99, 1), (0, 99, 1)];
        let c = disj(&spec);
        let above = merged(&[top_down(&c, &[0]), top_down(&c, &[1])]);
        let below = merged(&[bottom_up(&c, &[0, 1]), bottom_up(&c, &[1, 0])]);
        // Layer 1 of 4. Some above: {0, 1}; some below: {0, 1}. Union with job 2: 3 < 4.
        assert!(c.is_assignment_invalid(&*above, &*below, 1, 2));
        // On the exact properties of the path 0, 1, 2, 3 the same edge is fine.
        assert!(!removed(&c, &[0], 1, &[2, 3]));
    }

    #[test]
    fn a_node_with_no_path_has_no_valid_edge() {
        let c = disj(&tight());
        for job in 0..3 {
            assert!(c.is_assignment_invalid(
                &*c.identity_property(),
                &*c.empty_property_backward(),
                0,
                job
            ));
            assert!(c.is_assignment_invalid(&*c.empty_property(), &*c.identity_property(), 0, job));
        }
    }

    #[test]
    #[should_panic(expected = "Calling is_assignment_invalid")]
    fn a_property_of_another_constraint_is_rejected() {
        let c = disj(&tight());
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1]);
        let foreign = NotEquals::new(vars[0], vars[1], &problem).identity_property();
        c.is_assignment_invalid(&*foreign, &*c.identity_property(), 0, 0);
    }

    #[test]
    fn a_single_path_is_removed_exactly_when_it_violates_the_constraint() {
        // On the exact properties of one path, the filtering removes some edge of every path
        // that is not a solution, and none of a solution.
        let mut rng = Lcg(77);
        for _ in 0..40 {
            let n = 2 + rng.below(3) as usize;
            let spec = random_spec(&mut rng, n);
            let c = disj(&spec);
            for tuple in all_tuples(n, n as isize) {
                let some_edge_removed = (0..n).any(|k| {
                    c.is_assignment_invalid(
                        &*top_down(&c, &tuple[..k]),
                        &*bottom_up(&c, &tuple[k + 1..]),
                        k,
                        tuple[k],
                    )
                });
                assert_eq!(
                    some_edge_removed,
                    !c.is_satisfied(&tuple),
                    "{spec:?} on {tuple:?}"
                );
            }
        }
    }

    // ----------------------------------------------------------------------------------------
    // Compiled MDDs against the search
    // ----------------------------------------------------------------------------------------

    #[test]
    fn the_exact_mdd_of_the_hand_made_case() {
        let mdd = compile(problem_of(&tight()), vec![0, 1, 2], usize::MAX);
        assert_eq!(
            accepted(&mdd),
            vec![vec![1, 0, 2], vec![1, 2, 0], vec![2, 1, 0]]
        );
    }

    #[test]
    fn the_exact_mdd_agrees_with_the_search_on_random_instances() {
        let mut rng = Lcg(5150);
        for _ in 0..120 {
            let n = 1 + rng.below(5) as usize;
            let spec = random_spec(&mut rng, n);
            let mdd = compile(problem_of(&spec), (0..n).collect(), usize::MAX);
            assert_eq!(accepted(&mdd), expected(&spec), "{spec:?}");
        }
    }

    #[test]
    fn the_order_of_the_heuristic_is_repaired_to_the_declared_one() {
        let spec = tight();
        for order in [vec![2, 1, 0], vec![1, 0, 2], vec![0, 2, 1]] {
            let mdd = compile(problem_of(&spec), order.clone(), usize::MAX);
            assert_eq!(accepted(&mdd), expected(&spec), "order {order:?}");
        }
    }

    #[test]
    fn relaxed_mdds_never_lose_a_solution() {
        let mut rng = Lcg(31337);
        for _ in 0..80 {
            let n = 2 + rng.below(4) as usize;
            let spec = random_spec(&mut rng, n);
            let want = expected(&spec);
            for width in [1usize, 2, 3, usize::MAX] {
                let mdd = compile(problem_of(&spec), (0..n).collect(), width);
                let got = accepted(&mdd);
                for solution in &want {
                    assert!(
                        got.contains(solution),
                        "lost {solution:?}: {spec:?} width {width}"
                    );
                }
            }
        }
    }

    #[test]
    fn an_unsatisfiable_instance_is_unsat() {
        // Both jobs have to end by 3 and take 2 each.
        let spec = vec![(0, 3, 2), (0, 3, 2)];
        let mdd = compile(problem_of(&spec), vec![0, 1], usize::MAX);
        assert!(mdd.is_unsat());
    }

    #[test]
    fn the_time_windows_prune_without_any_search() {
        // Job 1 ends by 4 and the others need 3 and 2: it has to be placed first or second, so the
        // diagram, before any refinement, never puts it last.
        let mdd = compile(problem_of(&tight()), vec![0, 1, 2], 1);
        for solution in accepted(&mdd) {
            assert_ne!(solution[2], 1, "{solution:?}");
        }
    }

    #[test]
    fn a_constraint_among_other_variables_ignores_their_layers() {
        // Other variables between the jobs: the layers of the scope are 0, 2 and 4.
        let spec = tight();
        let mut problem = Problem::default();
        let vars = problem.add_variables(5, vec![0, 1, 2]);
        let (r, d, p) = lists(&spec);
        disjunctive(
            &mut problem,
            vec![vars[0], vars[2], vars[4]],
            r,
            d,
            p,
            None,
            None,
        );
        // A constraint that holds always, so that the variables 1 and 3 are in the diagram.
        among(&mut problem, vars.clone(), vec![0, 1, 2], 0, 5);
        let mdd = compile(problem, vec![0, 1, 2, 3, 4], usize::MAX);
        let solutions = accepted(&mdd);
        let schedules = expected(&spec);
        assert_eq!(solutions.len(), schedules.len() * 9);
        for solution in solutions {
            assert!(schedules.contains(&vec![solution[0], solution[2], solution[4]]));
        }
    }

    #[test]
    fn two_disjunctives_share_a_scope() {
        // Two machines on the same variables: both must accept the order.
        let a = vec![(0, 4, 2), (0, 9, 3), (0, 9, 2)];
        let b = vec![(0, 9, 2), (0, 9, 2), (0, 5, 2)];
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        let (r, d, p) = lists(&a);
        disjunctive(&mut problem, vars.clone(), r, d, p, None, None);
        let (r, d, p) = lists(&b);
        disjunctive(&mut problem, vars, r, d, p, None, None);
        let mdd = compile(problem, vec![0, 1, 2], usize::MAX);
        let want: Vec<Vec<isize>> = expected(&a)
            .into_iter()
            .filter(|t| expected(&b).contains(t))
            .collect();
        assert_eq!(accepted(&mdd), want);
    }

    #[test]
    fn the_wide_domain_of_a_disjunctive_is_filtered() {
        // Domains larger than the number of jobs: the values that are not jobs are removed.
        let spec = tight();
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, (0..6).collect());
        let (r, d, p) = lists(&spec);
        disjunctive(&mut problem, vars, r, d, p, None, None);
        let mdd = compile(problem, vec![0, 1, 2], usize::MAX);
        assert_eq!(accepted(&mdd), expected(&spec));
    }

    // ----------------------------------------------------------------------------------------
    // Precedences
    // ----------------------------------------------------------------------------------------

    type Pairs = Vec<(usize, usize)>;

    /// Like `disj`, with precedences.
    fn disj_prec(spec: &Spec, precedences: &Pairs) -> Disjunctive {
        let n = spec.len() as isize;
        let mut problem = Problem::default();
        let vars = problem.add_variables(spec.len(), (0..n).collect());
        let (r, d, p) = lists(spec);
        let mut c = Disjunctive::new(
            vars.clone(),
            r,
            d,
            p,
            Some(precedences.clone()),
            None,
            &problem,
        );
        c.update_variable_ordering(&vars);
        c
    }

    /// Like `problem_of`, with precedences.
    fn problem_prec(spec: &Spec, precedences: &Pairs) -> Problem {
        let n = spec.len() as isize;
        let mut problem = Problem::default();
        let vars = problem.add_variables(spec.len(), (0..n).collect());
        let (r, d, p) = lists(spec);
        disjunctive(&mut problem, vars, r, d, p, Some(precedences.clone()), None);
        problem
    }

    /// The feasible permutations that also respect the precedences, read off the positions.
    fn expected_prec(spec: &Spec, precedences: &Pairs) -> Vec<Vec<isize>> {
        expected(spec)
            .into_iter()
            .filter(|order| {
                precedences.iter().all(|&(before, after)| {
                    let at = |job: usize| order.iter().position(|&j| j as usize == job).unwrap();
                    at(before) < at(after)
                })
            })
            .collect()
    }

    /// Random precedences without cycle: from a smaller to a larger number of a random relabelling.
    fn random_precedences(rng: &mut Lcg, n: usize) -> Pairs {
        let mut label: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            label.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let mut pairs = vec![];
        for _ in 0..rng.below(n as u64 + 1) {
            let (a, b) = (rng.below(n as u64) as usize, rng.below(n as u64) as usize);
            if a < b {
                pairs.push((label[a], label[b]));
            }
        }
        pairs
    }

    #[test]
    fn is_satisfied_works_beyond_one_word_of_jobs() {
        // 70 jobs: the set of placed jobs does not fit in one word.
        let n = 70;
        let mut problem = Problem::default();
        let vars = problem.add_variables(n, (0..n as isize).collect());
        let c = Disjunctive::new(
            vars,
            vec![0; n],
            vec![1000; n],
            vec![1; n],
            Some(vec![(0, n - 1)]),
            None,
            &problem,
        );
        let forward: Vec<isize> = (0..n as isize).collect();
        assert!(c.is_satisfied(&forward));
        // Job 69 before job 0.
        let backward: Vec<isize> = (0..n as isize).rev().collect();
        assert!(!c.is_satisfied(&backward));
        // A repeated job, and a value that is not a job.
        let mut repeated = forward.clone();
        repeated[65] = 64;
        assert!(!c.is_satisfied(&repeated));
        let mut unknown = forward.clone();
        unknown[3] = n as isize;
        assert!(!c.is_satisfied(&unknown));
    }

    #[test]
    fn no_precedences_and_an_empty_list_are_the_same_constraint() {
        let spec = tight();
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        let (r, d, p) = lists(&spec);
        let none = Disjunctive::new(
            vars.clone(),
            r.clone(),
            d.clone(),
            p.clone(),
            None,
            None,
            &problem,
        );
        let empty = Disjunctive::new(vars, r, d, p, Some(vec![]), None, &problem);
        assert_eq!(
            none.structural_key(&problem),
            empty.structural_key(&problem)
        );
    }

    #[test]
    fn the_key_holds_the_precedences_sorted_and_without_duplicates() {
        let spec = tight();
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2]);
        let (r, d, p) = lists(&spec);
        let a = Disjunctive::new(
            vars.clone(),
            r.clone(),
            d.clone(),
            p.clone(),
            Some(vec![(2, 1), (0, 2), (2, 1)]),
            None,
            &problem,
        );
        let b = Disjunctive::new(vars, r, d, p, Some(vec![(0, 2), (2, 1)]), None, &problem);
        assert_eq!(a.structural_key(&problem), b.structural_key(&problem));
        let ConstraintShapeKey::Disjunctive { precedences, .. } = a.structural_key(&problem) else {
            panic!("not a disjunctive key");
        };
        assert_eq!(precedences, vec![(0, 2), (2, 1)]);
    }

    #[test]
    #[should_panic(expected = "names a job that is not in 0..3")]
    fn a_precedence_on_a_job_that_does_not_exist_is_refused() {
        disj_prec(&tight(), &vec![(0, 3)]);
    }

    #[test]
    #[should_panic(expected = "job 1 cannot precede itself")]
    fn a_precedence_of_a_job_on_itself_is_refused() {
        disj_prec(&tight(), &vec![(1, 1)]);
    }

    #[test]
    #[should_panic(expected = "contain a cycle")]
    fn a_cycle_of_precedences_is_refused() {
        disj_prec(&tight(), &vec![(0, 1), (1, 2), (2, 0)]);
    }

    #[test]
    fn check_precedences_accepts_a_diamond_and_reports_the_reason() {
        assert!(Disjunctive::check_precedences(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]).is_ok());
        assert!(Disjunctive::check_precedences(0, &[]).is_ok());
        assert!(
            Disjunctive::check_precedences(3, &[(0, 1), (1, 0)])
                .unwrap_err()
                .contains("cycle")
        );
    }

    #[test]
    fn is_satisfied_reads_the_precedences() {
        // `tight` has the solutions [1, 0, 2], [1, 2, 0] and [2, 1, 0]; "2 before 0" keeps the
        // last two, and "2 before 1" the last one only.
        let c = disj_prec(&tight(), &vec![(2, 0)]);
        assert!(!c.is_satisfied(&[1, 0, 2]));
        assert!(c.is_satisfied(&[1, 2, 0]));
        assert!(c.is_satisfied(&[2, 1, 0]));
        let c = disj_prec(&tight(), &vec![(2, 1)]);
        assert!(!c.is_satisfied(&[1, 2, 0]));
        assert!(c.is_satisfied(&[2, 1, 0]));
    }

    #[test]
    fn a_precedence_is_not_adjacency() {
        // 0 before 2, with job 1 in between, is fine.
        let spec = vec![(0, 99, 1), (0, 99, 1), (0, 99, 1)];
        let c = disj_prec(&spec, &vec![(0, 2)]);
        assert!(c.is_satisfied(&[0, 1, 2]));
        assert!(c.is_satisfied(&[0, 2, 1]));
        assert!(!c.is_satisfied(&[2, 0, 1]));
    }

    #[test]
    fn is_satisfied_agrees_with_the_search_and_the_precedences() {
        let mut rng = Lcg(808);
        for _ in 0..60 {
            let n = 1 + rng.below(4) as usize;
            let spec = random_spec(&mut rng, n);
            let precedences = random_precedences(&mut rng, n);
            let c = disj_prec(&spec, &precedences);
            let feasible = expected_prec(&spec, &precedences);
            for tuple in all_tuples(n, n as isize + 2) {
                let tuple: Vec<isize> = tuple.iter().map(|v| v - 1).collect();
                assert_eq!(
                    c.is_satisfied(&tuple),
                    feasible.contains(&tuple),
                    "{spec:?} {precedences:?} on {tuple:?}"
                );
            }
        }
    }

    #[test]
    fn a_job_whose_predecessor_was_not_placed_is_removed() {
        // 2 before 0: job 0 is removed at the root, and after 1 alone, but not after 2.
        let spec = vec![(0, 99, 1), (0, 99, 1), (0, 99, 1)];
        let c = disj_prec(&spec, &vec![(2, 0)]);
        assert!(removed(&c, &[], 0, &[1, 2]));
        assert!(removed(&c, &[1], 0, &[2]));
        assert!(!removed(&c, &[2], 0, &[1]));
        assert!(!removed(&c, &[1, 2], 0, &[]));
    }

    #[test]
    fn a_job_whose_successor_cannot_follow_is_removed() {
        // 2 before 0: job 2 is removed when job 0, its successor, is on no path below it.
        let spec = vec![(0, 99, 1), (0, 99, 1), (0, 99, 1)];
        let c = disj_prec(&spec, &vec![(2, 0)]);
        // Job 2 at the last position has nothing below it.
        assert!(removed(&c, &[0, 1], 2, &[]));
        assert!(!removed(&c, &[1], 2, &[0]));
    }

    #[test]
    fn the_precedences_of_a_merged_node_are_a_relaxation() {
        // One path above has placed 2 and the other has not: job 0, which has to come after 2,
        // is still possible, since some path above has placed 2.
        let spec = vec![(0, 99, 1), (0, 99, 1), (0, 99, 1), (0, 99, 1)];
        let c = disj_prec(&spec, &vec![(2, 0)]);
        let above = merged(&[top_down(&c, &[2]), top_down(&c, &[1])]);
        let below = bottom_up(&c, &[3]);
        assert!(!c.is_assignment_invalid(&*above, &*below, 1, 0));
    }

    #[test]
    fn a_single_path_is_removed_exactly_when_it_violates_the_precedences_too() {
        let mut rng = Lcg(4242);
        for _ in 0..40 {
            let n = 2 + rng.below(3) as usize;
            let spec = random_spec(&mut rng, n);
            let precedences = random_precedences(&mut rng, n);
            let c = disj_prec(&spec, &precedences);
            for tuple in all_tuples(n, n as isize) {
                let some_edge_removed = (0..n).any(|k| {
                    c.is_assignment_invalid(
                        &*top_down(&c, &tuple[..k]),
                        &*bottom_up(&c, &tuple[k + 1..]),
                        k,
                        tuple[k],
                    )
                });
                assert_eq!(
                    some_edge_removed,
                    !c.is_satisfied(&tuple),
                    "{spec:?} {precedences:?} on {tuple:?}"
                );
            }
        }
    }

    #[test]
    fn the_exact_mdd_with_precedences_agrees_with_the_search() {
        let mut rng = Lcg(99);
        for _ in 0..120 {
            let n = 1 + rng.below(5) as usize;
            let spec = random_spec(&mut rng, n);
            let precedences = random_precedences(&mut rng, n);
            let mdd = compile(
                problem_prec(&spec, &precedences),
                (0..n).collect(),
                usize::MAX,
            );
            assert_eq!(
                accepted(&mdd),
                expected_prec(&spec, &precedences),
                "{spec:?} {precedences:?}"
            );
        }
    }

    #[test]
    fn precedences_alone_give_the_linear_extensions() {
        // No time pressure: the solutions are the orders that respect 0 < 1 and 2 < 1.
        let spec = vec![(0, 99, 1), (0, 99, 1), (0, 99, 1)];
        let mdd = compile(
            problem_prec(&spec, &vec![(0, 1), (2, 1)]),
            vec![0, 1, 2],
            usize::MAX,
        );
        assert_eq!(accepted(&mdd), vec![vec![0, 2, 1], vec![2, 0, 1]]);
    }

    #[test]
    fn a_chain_of_precedences_fixes_the_order_without_search() {
        // Before any refinement, the first job has no predecessor to wait for and the last one
        // has no successor to precede, which is all that the sets can tell on a 1-width diagram.
        let spec = vec![(0, 99, 1); 4];
        let mdd = compile(
            problem_prec(&spec, &vec![(0, 1), (1, 2), (2, 3)]),
            vec![0, 1, 2, 3],
            1,
        );
        let got = accepted(&mdd);
        assert!(got.contains(&vec![0, 1, 2, 3]));
        for solution in &got {
            assert_eq!(solution[0], 0, "{solution:?}");
            assert_eq!(solution[3], 3, "{solution:?}");
        }
    }

    #[test]
    fn relaxed_mdds_with_precedences_never_lose_a_solution() {
        let mut rng = Lcg(2718);
        for _ in 0..60 {
            let n = 2 + rng.below(4) as usize;
            let spec = random_spec(&mut rng, n);
            let precedences = random_precedences(&mut rng, n);
            let want = expected_prec(&spec, &precedences);
            for width in [1usize, 2, 3, usize::MAX] {
                let mdd = compile(problem_prec(&spec, &precedences), (0..n).collect(), width);
                let got = accepted(&mdd);
                for solution in &want {
                    assert!(
                        got.contains(solution),
                        "lost {solution:?}: {spec:?} {precedences:?} width {width}"
                    );
                }
            }
        }
    }

    #[test]
    fn precedences_that_clash_with_the_windows_make_it_unsat() {
        // Job 1 has to end by 2 and job 0 takes 2: "0 before 1" cannot be met.
        let spec = vec![(0, 9, 2), (0, 2, 2)];
        let mdd = compile(problem_prec(&spec, &vec![(0, 1)]), vec![0, 1], usize::MAX);
        assert!(mdd.is_unsat());
    }

    // ----------------------------------------------------------------------------------------
    // Setup times
    // ----------------------------------------------------------------------------------------

    type Matrix = Vec<Vec<isize>>;

    /// The constraint with setup times, and precedences if any.
    fn disj_setup(
        mode: SetupRelaxation,
        spec: &Spec,
        precedences: &Pairs,
        setup: &Matrix,
    ) -> Disjunctive {
        let n = spec.len() as isize;
        let mut problem = Problem::default();
        let vars = problem.add_variables(spec.len(), (0..n).collect());
        let (r, d, p) = lists(spec);
        let precedences = Some(precedences.clone()).filter(|p| !p.is_empty());
        let setup = SetupTimes::new(setup.clone()).with_relaxation(mode);
        let mut c = Disjunctive::new(vars.clone(), r, d, p, precedences, Some(setup), &problem);
        c.update_variable_ordering(&vars);
        c
    }

    fn problem_setup(
        mode: SetupRelaxation,
        spec: &Spec,
        precedences: &Pairs,
        setup: &Matrix,
    ) -> Problem {
        let n = spec.len() as isize;
        let mut problem = Problem::default();
        let vars = problem.add_variables(spec.len(), (0..n).collect());
        let (r, d, p) = lists(spec);
        disjunctive(
            &mut problem,
            vars,
            r,
            d,
            p,
            Some(precedences.clone()),
            Some(SetupTimes::new(setup.clone()).with_relaxation(mode)),
        );
        problem
    }

    /// Declares, for a test function `$inner(mode)`, one test per relaxation.
    macro_rules! each_relaxation {
        ($name:ident => $inner:ident) => {
            mod $name {
                use super::*;

                #[test]
                fn last_job() {
                    $inner(SetupRelaxation::LastJob)
                }

                #[test]
                fn last_set() {
                    $inner(SetupRelaxation::LastSet)
                }

                #[test]
                fn arc_times() {
                    $inner(SetupRelaxation::ArcTimes)
                }
            }
        };
    }

    /// The feasible permutations with setup times and precedences, by a search that does not use
    /// the constraint.
    fn expected_setup(spec: &Spec, precedences: &Pairs, setup: &Matrix) -> Vec<Vec<isize>> {
        fn extend(
            spec: &Spec,
            setup: &Matrix,
            previous: Option<(usize, isize)>,
            used: &mut Vec<bool>,
            prefix: &mut Vec<isize>,
            out: &mut Vec<Vec<isize>>,
        ) {
            if prefix.len() == spec.len() {
                out.push(prefix.clone());
                return;
            }
            for job in 0..spec.len() {
                if used[job] {
                    continue;
                }
                let (release, deadline, duration) = spec[job];
                let ready = match previous {
                    Some((last, end)) => end + setup[last][job],
                    None => release,
                };
                let start = ready.max(release);
                if start + duration > deadline {
                    continue;
                }
                used[job] = true;
                prefix.push(job as isize);
                extend(
                    spec,
                    setup,
                    Some((job, start + duration)),
                    used,
                    prefix,
                    out,
                );
                prefix.pop();
                used[job] = false;
            }
        }
        let mut out = vec![];
        extend(
            spec,
            setup,
            None,
            &mut vec![false; spec.len()],
            &mut vec![],
            &mut out,
        );
        out.retain(|order| {
            precedences.iter().all(|&(before, after)| {
                let at = |job: usize| order.iter().position(|&j| j as usize == job).unwrap();
                at(before) < at(after)
            })
        });
        out.sort();
        out
    }

    fn random_setup(rng: &mut Lcg, n: usize) -> Matrix {
        (0..n)
            .map(|_| (0..n).map(|_| rng.below(4) as isize).collect())
            .collect()
    }

    #[test]
    fn a_matrix_of_the_wrong_shape_is_refused() {
        assert!(Disjunctive::check_setup_times(2, &[vec![0, 1], vec![1, 0]]).is_ok());
        assert!(Disjunctive::check_setup_times(0, &[]).is_ok());
        let rows = Disjunctive::check_setup_times(3, &[vec![0; 3], vec![0; 3]]).unwrap_err();
        assert!(rows.contains("2 rows for 3 jobs"), "{rows}");
        let row = Disjunctive::check_setup_times(2, &[vec![0, 1], vec![1]]).unwrap_err();
        assert!(row.contains("row 1"), "{row}");
        let negative = Disjunctive::check_setup_times(2, &[vec![0, -1], vec![1, 0]]).unwrap_err();
        assert!(
            negative.contains("from job 0 to job 1 is negative"),
            "{negative}"
        );
        // The diagonal is ignored, even when it is negative.
        assert!(Disjunctive::check_setup_times(2, &[vec![-4, 1], vec![1, -4]]).is_ok());
    }

    #[test]
    #[should_panic(expected = "Disjunctive: row 1 of the setup times has 1 entries for 2 jobs")]
    fn the_constructor_refuses_a_ragged_matrix() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1]);
        Disjunctive::new(
            vars,
            vec![0; 2],
            vec![9; 2],
            vec![1; 2],
            None,
            Some(SetupTimes::new(vec![vec![0, 1], vec![1]])),
            &problem,
        );
    }

    #[test]
    fn the_key_holds_the_matrix_without_its_diagonal_and_none_for_no_setup() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1]);
        let new = |setup: Option<Matrix>| {
            Disjunctive::new(
                vars.clone(),
                vec![0; 2],
                vec![9; 2],
                vec![1; 2],
                None,
                setup.map(SetupTimes::new),
                &problem,
            )
            .structural_key(&problem)
        };
        let none = new(None);
        // A matrix with no positive entry is no setup time at all.
        assert_eq!(new(Some(vec![vec![0, 0], vec![0, 0]])), none);
        assert_eq!(new(Some(vec![vec![5, 0], vec![0, 7]])), none);
        let with = new(Some(vec![vec![9, 2], vec![3, 9]]));
        assert_ne!(with, none);
        // The diagonal does not matter.
        assert_eq!(with, new(Some(vec![vec![0, 2], vec![3, 1]])));
        let ConstraintShapeKey::Disjunctive { setup_times, .. } = with else {
            panic!("not a disjunctive key");
        };
        assert_eq!(setup_times, Some(vec![0, 2, 3, 0]));
    }

    fn a_setup_is_between_consecutive_jobs_and_not_before_the_first(mode: SetupRelaxation) {
        // Both jobs last 2. Setup 3 from 0 to 1, none the other way.
        let spec = vec![(0, 20, 2), (0, 7, 2)];
        let setup = vec![vec![0, 3], vec![0, 0]];
        let c = disj_setup(mode, &spec, &vec![], &setup);
        // [0, 1]: job 0 is over at 2, job 1 starts at 5 and is over at 7: just in time.
        assert!(c.is_satisfied(&[0, 1]));
        // With one more unit of setup it would be late.
        let c = disj_setup(mode, &spec, &vec![], &vec![vec![0, 4], vec![0, 0]]);
        assert!(!c.is_satisfied(&[0, 1]));
        // [1, 0]: no setup is paid before the first job and the setup from 1 to 0 is 0.
        assert!(c.is_satisfied(&[1, 0]));
    }

    each_relaxation!(a_setup_is_between_consecutive_jobs_and_not_before_the_first_modes => a_setup_is_between_consecutive_jobs_and_not_before_the_first);

    fn a_setup_does_not_add_to_a_wait_for_the_release_date(mode: SetupRelaxation) {
        // Job 1 is released at 10: the setup of 3 after job 0 (over at 2) is absorbed in the wait.
        let spec = vec![(0, 20, 2), (10, 12, 2)];
        let c = disj_setup(mode, &spec, &vec![], &vec![vec![0, 3], vec![0, 0]]);
        assert!(c.is_satisfied(&[0, 1]));
    }

    each_relaxation!(a_setup_does_not_add_to_a_wait_for_the_release_date_modes => a_setup_does_not_add_to_a_wait_for_the_release_date);

    fn is_satisfied_agrees_with_the_search_and_the_setup_times(mode: SetupRelaxation) {
        let mut rng = Lcg(31);
        for _ in 0..60 {
            let n = 1 + rng.below(4) as usize;
            let spec = random_spec(&mut rng, n);
            let precedences = random_precedences(&mut rng, n);
            let setup = random_setup(&mut rng, n);
            let c = disj_setup(mode, &spec, &precedences, &setup);
            let feasible = expected_setup(&spec, &precedences, &setup);
            for tuple in all_tuples(n, n as isize + 2) {
                let tuple: Vec<isize> = tuple.iter().map(|v| v - 1).collect();
                assert_eq!(
                    c.is_satisfied(&tuple),
                    feasible.contains(&tuple),
                    "{spec:?} {precedences:?} {setup:?} on {tuple:?}"
                );
            }
        }
    }

    each_relaxation!(is_satisfied_agrees_with_the_search_and_the_setup_times_modes => is_satisfied_agrees_with_the_search_and_the_setup_times);

    fn the_completion_time_of_a_path_includes_the_setups(mode: SetupRelaxation) {
        // Jobs of 2 time units. Setups: 0 to 1 is 3, 1 to 2 is 1.
        let spec = vec![(0, 99, 2); 3];
        let setup = vec![vec![0, 3, 0], vec![0, 0, 1], vec![0, 0, 0]];
        let c = disj_setup(mode, &spec, &vec![], &setup);
        assert_eq!(times(&*top_down(&c, &[0]))[..2], [2, 2]);
        assert_eq!(times(&*top_down(&c, &[0, 1]))[..2], [7, 7]);
        assert_eq!(times(&*top_down(&c, &[0, 1, 2]))[..2], [10, 10]);
        // Read from the sink: job 2 starts by 97, job 1 by 97 - 1 - 2 = 94, job 0 by 94 - 3 - 2.
        assert_eq!(times(&*bottom_up(&c, &[2]))[2..], [97, 97]);
        assert_eq!(times(&*bottom_up(&c, &[1, 2]))[2..], [94, 94]);
        assert_eq!(times(&*bottom_up(&c, &[0, 1, 2]))[2..], [89, 89]);
    }

    each_relaxation!(the_completion_time_of_a_path_includes_the_setups_modes => the_completion_time_of_a_path_includes_the_setups);

    fn paths_that_end_at_the_same_time_with_different_last_jobs_are_not_one_state(
        mode: SetupRelaxation,
    ) {
        // [0, 1] and [1, 0] both end at 3 (two jobs of 1, one setup of 1 whichever the order),
        // but the setup that follows depends on the last job.
        let spec = vec![(0, 99, 1); 3];
        let setup = vec![vec![0, 1, 1], vec![1, 0, 5], vec![1, 1, 0]];
        let c = disj_setup(mode, &spec, &vec![], &setup);
        let a = top_down(&c, &[0, 1]);
        let b = top_down(&c, &[1, 0]);
        assert_eq!(times(&*a)[..2], times(&*b)[..2]);
        let m = merged(&[a, b]);
        assert_eq!(times(&*m)[..2], [3, 3]);
        assert!(!m.is_single_state());
        // Without setup times the same two paths are one state.
        let c = disj(&spec);
        let m = merged(&[top_down(&c, &[0, 1]), top_down(&c, &[1, 0])]);
        assert!(m.is_single_state());
    }

    each_relaxation!(paths_that_end_at_the_same_time_with_different_last_jobs_are_not_one_state_modes => paths_that_end_at_the_same_time_with_different_last_jobs_are_not_one_state);

    fn the_same_last_job_keeps_a_state_single(mode: SetupRelaxation) {
        // [0, 2, 1] and [2, 0, 1] end with job 1 at the same time when the setups into 1 agree.
        let spec = vec![(0, 99, 1); 3];
        let setup = vec![vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]];
        let c = disj_setup(mode, &spec, &vec![], &setup);
        let m = merged(&[top_down(&c, &[0, 2, 1]), top_down(&c, &[2, 0, 1])]);
        assert!(m.is_single_state());
    }

    each_relaxation!(the_same_last_job_keeps_a_state_single_modes => the_same_last_job_keeps_a_state_single);

    fn a_setup_can_make_an_edge_late_on_either_side(mode: SetupRelaxation) {
        // Job 1 has to end by 3, and takes 1. After job 0 there is a setup of 5.
        let spec = vec![(0, 99, 1), (0, 3, 1)];
        let setup = vec![vec![0, 5], vec![0, 0]];
        let c = disj_setup(mode, &spec, &vec![], &setup);
        // Seen from the job placed second: it would end at 1 + 5 + 1.
        assert!(removed(&c, &[0], 1, &[]));
        assert!(!removed(&c, &[], 1, &[0]));
        // Seen from the job placed first: the job after it has to start by 2.
        assert!(removed(&c, &[], 0, &[1]));
        assert!(!removed(&c, &[1], 0, &[]));
        // Without the setup the same edges are fine.
        let c = disj(&spec);
        assert!(!removed(&c, &[0], 1, &[]));
        assert!(!removed(&c, &[], 0, &[1]));
    }

    each_relaxation!(a_setup_can_make_an_edge_late_on_either_side_modes => a_setup_can_make_an_edge_late_on_either_side);

    fn a_node_that_mixes_last_jobs_uses_the_smallest_setup(mode: SetupRelaxation) {
        // Above: job 0 or job 1 was placed last. The setup into job 2 is 6 after 0 and 2 after 1,
        // so the filtering has to assume 2. Job 2 ends by 4: 1 + 2 + 1 = 4 is just in time.
        let spec = vec![(0, 99, 1), (0, 99, 1), (0, 4, 1)];
        let setup = vec![vec![0, 0, 6], vec![0, 0, 2], vec![0, 0, 0]];
        let c = disj_setup(mode, &spec, &vec![], &setup);
        let above = merged(&[top_down(&c, &[0]), top_down(&c, &[1])]);
        let below = c.empty_property_backward();
        assert!(!c.is_assignment_invalid(&*above, &*below, 1, 2));
        // With the setup after 1 raised to 3, it would be late whatever came before.
        let setup = vec![vec![0, 0, 6], vec![0, 0, 3], vec![0, 0, 0]];
        let c = disj_setup(mode, &spec, &vec![], &setup);
        let above = merged(&[top_down(&c, &[0]), top_down(&c, &[1])]);
        assert!(c.is_assignment_invalid(&*above, &*c.empty_property_backward(), 1, 2));
        // The exact nodes know the setup: after job 0 it is late, after job 1 it is not.
        let c = disj_setup(
            mode,
            &spec,
            &vec![],
            &vec![vec![0, 0, 6], vec![0, 0, 2], vec![0, 0, 0]],
        );
        assert!(removed(&c, &[0], 2, &[1]));
        assert!(!removed(&c, &[1], 2, &[0]));
    }

    each_relaxation!(a_node_that_mixes_last_jobs_uses_the_smallest_setup_modes => a_node_that_mixes_last_jobs_uses_the_smallest_setup);

    fn a_single_path_is_removed_exactly_when_it_violates_the_setup_times_too(
        mode: SetupRelaxation,
    ) {
        let mut rng = Lcg(606);
        for _ in 0..40 {
            let n = 2 + rng.below(3) as usize;
            let spec = random_spec(&mut rng, n);
            let precedences = random_precedences(&mut rng, n);
            let setup = random_setup(&mut rng, n);
            let c = disj_setup(mode, &spec, &precedences, &setup);
            for tuple in all_tuples(n, n as isize) {
                let some_edge_removed = (0..n).any(|k| {
                    c.is_assignment_invalid(
                        &*top_down(&c, &tuple[..k]),
                        &*bottom_up(&c, &tuple[k + 1..]),
                        k,
                        tuple[k],
                    )
                });
                assert_eq!(
                    some_edge_removed,
                    !c.is_satisfied(&tuple),
                    "{spec:?} {precedences:?} {setup:?} on {tuple:?}"
                );
            }
        }
    }

    each_relaxation!(a_single_path_is_removed_exactly_when_it_violates_the_setup_times_too_modes => a_single_path_is_removed_exactly_when_it_violates_the_setup_times_too);

    fn the_exact_mdd_with_setup_times_agrees_with_the_search(mode: SetupRelaxation) {
        let mut rng = Lcg(1234);
        for _ in 0..120 {
            let n = 1 + rng.below(5) as usize;
            let spec = random_spec(&mut rng, n);
            let precedences = random_precedences(&mut rng, n);
            let setup = random_setup(&mut rng, n);
            let mdd = compile(
                problem_setup(mode, &spec, &precedences, &setup),
                (0..n).collect(),
                usize::MAX,
            );
            assert_eq!(
                accepted(&mdd),
                expected_setup(&spec, &precedences, &setup),
                "{spec:?} {precedences:?} {setup:?}"
            );
        }
    }

    each_relaxation!(the_exact_mdd_with_setup_times_agrees_with_the_search_modes => the_exact_mdd_with_setup_times_agrees_with_the_search);

    fn relaxed_mdds_with_setup_times_never_lose_a_solution(mode: SetupRelaxation) {
        let mut rng = Lcg(8);
        for _ in 0..60 {
            let n = 2 + rng.below(4) as usize;
            let spec = random_spec(&mut rng, n);
            let precedences = random_precedences(&mut rng, n);
            let setup = random_setup(&mut rng, n);
            let want = expected_setup(&spec, &precedences, &setup);
            for width in [1usize, 2, 3, usize::MAX] {
                let mdd = compile(
                    problem_setup(mode, &spec, &precedences, &setup),
                    (0..n).collect(),
                    width,
                );
                let got = accepted(&mdd);
                for solution in &want {
                    assert!(
                        got.contains(solution),
                        "lost {solution:?}: {spec:?} {precedences:?} {setup:?} width {width}"
                    );
                }
            }
        }
    }

    each_relaxation!(relaxed_mdds_with_setup_times_never_lose_a_solution_modes => relaxed_mdds_with_setup_times_never_lose_a_solution);

    #[test]
    fn setup_times_only_remove_solutions() {
        // Adding setup times to a constraint can only make the schedules longer.
        let mut rng = Lcg(77);
        for _ in 0..40 {
            let n = 2 + rng.below(3) as usize;
            let spec = random_spec(&mut rng, n);
            let setup = random_setup(&mut rng, n);
            let plain = expected(&spec);
            for solution in expected_setup(&spec, &vec![], &setup) {
                assert!(plain.contains(&solution), "{spec:?} {setup:?}");
            }
        }
    }

    fn a_matrix_of_zeros_compiles_to_the_same_diagram_as_no_matrix(mode: SetupRelaxation) {
        let spec = tight();
        let zeros = vec![vec![0; 3]; 3];
        let with = compile(
            problem_setup(mode, &spec, &vec![], &zeros),
            vec![0, 1, 2],
            usize::MAX,
        );
        let without = compile(problem_of(&spec), vec![0, 1, 2], usize::MAX);
        assert_eq!(accepted(&with), accepted(&without));
    }

    each_relaxation!(a_matrix_of_zeros_compiles_to_the_same_diagram_as_no_matrix_modes => a_matrix_of_zeros_compiles_to_the_same_diagram_as_no_matrix);

    fn setup_times_make_a_tight_instance_unsat(mode: SetupRelaxation) {
        // Two jobs of 2 in [0, 5]: fine in either order, until a setup of 2 is needed.
        let spec = vec![(0, 5, 2), (0, 5, 2)];
        let setup = vec![vec![0, 2], vec![2, 0]];
        let mdd = compile(
            problem_setup(mode, &spec, &vec![], &setup),
            vec![0, 1],
            usize::MAX,
        );
        assert!(mdd.is_unsat());
    }

    each_relaxation!(setup_times_make_a_tight_instance_unsat_modes => setup_times_make_a_tight_instance_unsat);

    fn the_order_key_has_more_numbers_with_setup_times_and_stays_finite(mode: SetupRelaxation) {
        let spec = vec![(0, 99, 1); 3];
        let setup = vec![vec![0, 1, 1], vec![1, 0, 5], vec![1, 1, 0]];
        let c = disj_setup(mode, &spec, &vec![], &setup);
        let plain = disj(&spec);
        // One number for the open job or the size of the set, two for the populated slots.
        let extra = if mode == SetupRelaxation::ArcTimes {
            2
        } else {
            1
        };
        assert_eq!(
            c.empty_property().order_key().len(),
            plain.empty_property().order_key().len() + extra
        );
        let length = c.empty_property().order_key().len();
        for property in [
            c.identity_property(),
            c.empty_property(),
            top_down(&c, &[1, 2]),
            bottom_up(&c, &[1, 2]),
            merged(&[top_down(&c, &[0, 1]), top_down(&c, &[1, 0])]),
        ] {
            let key = property.order_key();
            assert_eq!(key.len(), length);
            assert!(key.iter().all(|x| x.is_finite()));
        }
    }

    each_relaxation!(the_order_key_has_more_numbers_with_setup_times_and_stays_finite_modes => the_order_key_has_more_numbers_with_setup_times_and_stays_finite);

    fn eq_and_hash_see_the_job_at_the_open_end(mode: SetupRelaxation) {
        let spec = vec![(0, 99, 1); 3];
        let setup = vec![vec![0, 1, 1], vec![1, 0, 5], vec![1, 1, 0]];
        let c = disj_setup(mode, &spec, &vec![], &setup);
        // Same jobs, same time, different last job.
        let a = top_down(&c, &[0, 1]);
        let b = top_down(&c, &[1, 0]);
        assert_eq!(times(&*a)[..2], times(&*b)[..2]);
        assert!(!ConstraintProperty::eq(&*a, &*b));
        assert_ne!(hash_of(&*a), hash_of(&*b));
        assert!(ConstraintProperty::eq(&*a, &*top_down(&c, &[0, 1])));
    }

    each_relaxation!(eq_and_hash_see_the_job_at_the_open_end_modes => eq_and_hash_see_the_job_at_the_open_end);

    fn out_of_scope_layers_copy_the_job_at_the_open_end(mode: SetupRelaxation) {
        let spec = vec![(0, 99, 1); 2];
        let c = disj_setup(mode, &spec, &vec![], &vec![vec![0, 2], vec![2, 0]]);
        let parent = top_down(&c, &[1]);
        let mut child = c.identity_property();
        child.update(&*parent, 99, false);
        assert!(ConstraintProperty::eq(&*child, &*parent));
        let below = bottom_up(&c, &[0]);
        let mut above = c.identity_property();
        above.update_backward(&*below, 99, false);
        assert!(ConstraintProperty::eq(&*above, &*below));
    }

    each_relaxation!(out_of_scope_layers_copy_the_job_at_the_open_end_modes => out_of_scope_layers_copy_the_job_at_the_open_end);

    fn identity_is_neutral_for_merge_with_setup_times(mode: SetupRelaxation) {
        let spec = vec![(0, 99, 1); 3];
        let setup = vec![vec![0, 1, 1], vec![1, 0, 5], vec![1, 1, 0]];
        let c = disj_setup(mode, &spec, &vec![], &setup);
        for property in [
            top_down(&c, &[1, 0]),
            bottom_up(&c, &[1, 0]),
            c.empty_property(),
        ] {
            let mut with_identity = dyn_clone::clone_box(&*property);
            with_identity.merge(&*c.identity_property());
            assert!(ConstraintProperty::eq(&*with_identity, &*property));
        }
    }

    each_relaxation!(identity_is_neutral_for_merge_with_setup_times_modes => identity_is_neutral_for_merge_with_setup_times);

    #[test]
    fn a_setup_larger_than_one_word_of_jobs_works() {
        // 70 jobs in a row, a setup of 1 from each job to the next one only.
        let n = 70;
        let mut problem = Problem::default();
        let vars = problem.add_variables(n, (0..n as isize).collect());
        let mut setup = vec![vec![0; n]; n];
        for (job, row) in setup.iter_mut().enumerate().take(n - 1) {
            row[job + 1] = 1;
        }
        let c = Disjunctive::new(
            vars,
            vec![0; n],
            vec![2 * n as isize - 2; n],
            vec![1; n],
            None,
            Some(SetupTimes::new(setup)),
            &problem,
        );
        let forward: Vec<isize> = (0..n as isize).collect();
        // The last job ends at 70 + 69 setups = 139 = 2 * 70 - 1: one unit too late.
        assert!(!c.is_satisfied(&forward));
        let mut backward: Vec<isize> = forward.clone();
        backward.reverse();
        assert!(c.is_satisfied(&backward));
    }

    // ----------------------------------------------------------------------------------------
    // The relaxations of the setup times
    // ----------------------------------------------------------------------------------------

    /// The constraint of `spec` with the setup times `setup`, for `mode`, on a scope of its own.
    fn setup_constraint(mode: SetupRelaxation, spec: &Spec, setup: &Matrix) -> Disjunctive {
        disj_setup(mode, spec, &vec![], setup)
    }

    /// The edge of `job` between the merged states of `above` (a list of paths) and the path
    /// `below`.
    fn removed_above_merged(
        c: &Disjunctive,
        above: &[&[isize]],
        job: isize,
        below: &[isize],
    ) -> bool {
        let parents: Vec<_> = above.iter().map(|path| top_down(c, path)).collect();
        let layer = above[0].len();
        c.is_assignment_invalid(&*merged(&parents), &*bottom_up(c, below), layer, job)
    }

    #[test]
    fn the_names_of_the_relaxations_round_trip() {
        for mode in SetupRelaxation::ALL {
            assert_eq!(SetupRelaxation::from_name(mode.name()), Some(mode));
        }
        assert_eq!(SetupRelaxation::from_name("everything"), None);
        assert_eq!(SetupRelaxation::default(), SetupRelaxation::LastJob);
    }

    #[test]
    fn the_relaxation_is_in_the_key_only_when_there_are_setup_times() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1]);
        let key = |setup: Option<SetupTimes>| {
            Disjunctive::new(
                vars.clone(),
                vec![0; 2],
                vec![9; 2],
                vec![1; 2],
                None,
                setup,
                &problem,
            )
            .structural_key(&problem)
        };
        let relaxation_of = |key: &ConstraintShapeKey| match key {
            ConstraintShapeKey::Disjunctive {
                setup_relaxation, ..
            } => *setup_relaxation,
            _ => panic!("not a disjunctive key"),
        };
        let matrix = vec![vec![0, 2], vec![3, 0]];
        let zeros = vec![vec![0, 0], vec![0, 0]];
        let none = key(None);
        let mut keys = vec![];
        for mode in SetupRelaxation::ALL {
            let with = key(Some(SetupTimes::new(matrix.clone()).with_relaxation(mode)));
            assert_eq!(relaxation_of(&with), mode);
            // Without any positive setup, the choice is not a different constraint.
            let without = key(Some(SetupTimes::new(zeros.clone()).with_relaxation(mode)));
            assert_eq!(without, none);
            assert_eq!(relaxation_of(&without), SetupRelaxation::LastJob);
            keys.push(with);
        }
        assert_ne!(keys[0], keys[1]);
        assert_ne!(keys[1], keys[2]);
        assert_ne!(keys[0], keys[2]);
    }

    #[test]
    fn the_properties_of_a_relaxation_hold_the_expected_state() {
        let spec = vec![(0, 99, 1); 3];
        let setup = vec![vec![0, 1, 1], vec![1, 0, 5], vec![1, 1, 0]];
        for mode in SetupRelaxation::ALL {
            let c = setup_constraint(mode, &spec, &setup);
            // The root and the sink are the empty path, which is a single state.
            assert!(c.empty_property().is_single_state(), "{mode:?}");
            let one = top_down(&c, &[0]);
            let two = top_down(&c, &[1]);
            assert!(one.is_single_state(), "{mode:?}");
            assert!(!merged(&[one, two]).is_single_state(), "{mode:?}");
            let one = bottom_up(&c, &[0]);
            let two = bottom_up(&c, &[1]);
            assert!(one.is_single_state(), "{mode:?}");
            assert!(!merged(&[one, two]).is_single_state(), "{mode:?}");
        }
    }

    #[test]
    fn a_set_of_last_jobs_is_tighter_than_the_smallest_setup_of_all_jobs() {
        // Job 3 has to be over by 6. After job 0 the setup into job 3 is 5, after job 1 it is 6
        // and after job 2 it is 0. Two paths end with job 0 or job 1, both at time 1.
        let spec = vec![(0, 99, 1), (0, 99, 1), (0, 99, 1), (0, 6, 1)];
        let setup = vec![
            vec![0, 0, 0, 5],
            vec![0, 0, 0, 6],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
        ];
        let removed_with = |mode| {
            let c = setup_constraint(mode, &spec, &setup);
            removed_above_merged(&c, &[&[0], &[1]], 3, &[2, 0])
        };
        // The last job is not known: the smallest setup of any job is 0, the edge looks fine.
        assert!(!removed_with(SetupRelaxation::LastJob));
        // But the last job is one of the two first jobs, and 1 + 5 + 1 is late.
        assert!(removed_with(SetupRelaxation::LastSet));
        assert!(removed_with(SetupRelaxation::ArcTimes));
        // The exact edges are removed by all of them.
        for mode in SetupRelaxation::ALL {
            let c = setup_constraint(mode, &spec, &setup);
            assert!(removed(&c, &[0], 3, &[1, 2]), "{mode:?}");
            assert!(removed(&c, &[1], 3, &[0, 2]), "{mode:?}");
            assert!(!removed(&c, &[2], 3, &[0, 1]), "{mode:?}");
        }
    }

    #[test]
    fn the_same_for_the_jobs_after_the_edge() {
        // The mirror image: job 0 is followed by job 1 (setup 5) or job 2 (setup 6), which both
        // have to be over by 6, where job 3 would follow job 0 with a setup of 0.
        let spec = vec![(0, 99, 1), (0, 6, 1), (0, 6, 1), (0, 99, 1)];
        let setup = vec![
            vec![0, 5, 6, 0],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
        ];
        let removed_with = |mode| {
            let c = setup_constraint(mode, &spec, &setup);
            let below: Vec<_> = [&[1, 3][..], &[2, 3][..]]
                .iter()
                .map(|path| bottom_up(&c, path))
                .collect();
            c.is_assignment_invalid(&*top_down(&c, &[]), &*merged(&below), 0, 0)
        };
        assert!(!removed_with(SetupRelaxation::LastJob));
        assert!(removed_with(SetupRelaxation::LastSet));
        assert!(removed_with(SetupRelaxation::ArcTimes));
    }

    #[test]
    fn arc_times_keep_the_time_of_each_last_job() {
        // One path ends with job 0 at time 10, the other with job 1 at time 2. The setup into job
        // 2 is 0 after job 0 and 9 after job 1: the first path arrives at 10, the second at 11.
        // Job 2 has to be over by 10 and lasts 1, so neither can place it, but a state that keeps
        // the best time (2) and the best setup (0) does not see it.
        let spec = vec![(0, 99, 10), (0, 99, 2), (0, 10, 1)];
        let setup = vec![vec![0, 0, 0], vec![0, 0, 9], vec![0, 0, 0]];
        let removed_with = |mode| {
            let c = setup_constraint(mode, &spec, &setup);
            removed_above_merged(&c, &[&[0], &[1]], 2, &[1])
        };
        assert!(!removed_with(SetupRelaxation::LastJob));
        assert!(!removed_with(SetupRelaxation::LastSet));
        assert!(removed_with(SetupRelaxation::ArcTimes));
    }

    #[test]
    fn arc_times_do_not_remove_an_edge_that_one_path_can_take() {
        // The same, but the setup after job 1 is 5 and the path of job 1 can place job 2.
        let spec = vec![(0, 99, 10), (0, 99, 2), (0, 10, 1)];
        let setup = vec![vec![0, 0, 0], vec![0, 0, 5], vec![0, 0, 0]];
        let c = setup_constraint(SetupRelaxation::ArcTimes, &spec, &setup);
        // 2 + 5 + 1 = 8 is in time, and 10 + 0 + 1 = 11 is not: the best of the two counts.
        assert!(!removed_above_merged(&c, &[&[0], &[1]], 2, &[1]));
    }

    #[test]
    fn the_bounds_in_the_scalars_of_arc_times_are_the_best_and_worst_of_the_slots() {
        let spec = vec![(0, 99, 10), (0, 99, 2), (0, 99, 1)];
        let setup = vec![vec![0, 0, 0], vec![0, 0, 9], vec![0, 0, 0]];
        let c = setup_constraint(SetupRelaxation::ArcTimes, &spec, &setup);
        let m = merged(&[top_down(&c, &[0]), top_down(&c, &[1])]);
        assert_eq!(times(&*m)[..2], [2, 10]);
        // After job 2, whose setup is 0 or 9: ends at 11 (from 0) or at 12 (from 1).
        let mut child = c.identity_property();
        child.update(&*m, 2, true);
        assert_eq!(times(&*child)[..2], [11, 12]);
        // With the last job known, the setup is the one of the path.
        assert_eq!(times(&*top_down(&c, &[1, 2]))[..2], [12, 12]);
        assert_eq!(times(&*top_down(&c, &[0, 2]))[..2], [11, 11]);
    }

    #[test]
    fn the_sets_and_slots_work_across_words() {
        // 70 jobs, so the set (with its bit for the empty path) has two words, and so do the
        // slots. A setup of 1 from each job to the next one only.
        let n = 70;
        let spec = vec![(0, 999, 1); n];
        let mut setup = vec![vec![0; n]; n];
        for (job, row) in setup.iter_mut().enumerate().take(n - 1) {
            row[job + 1] = 1;
        }
        for mode in SetupRelaxation::ALL {
            let c = setup_constraint(mode, &spec, &setup);
            // Job 66 ends at 1, then job 67 comes after a setup of 1: it ends at 3.
            assert_eq!(times(&*top_down(&c, &[66, 67]))[..2], [3, 3], "{mode:?}");
            // In the other order, job 66 comes after 67 without setup.
            assert_eq!(times(&*top_down(&c, &[67, 66]))[..2], [2, 2], "{mode:?}");
            assert_eq!(
                times(&*bottom_up(&c, &[65, 66]))[2..],
                [996, 996],
                "{mode:?}"
            );
            let m = merged(&[top_down(&c, &[66]), top_down(&c, &[67])]);
            assert!(!m.is_single_state(), "{mode:?}");
            assert!(top_down(&c, &[69]).is_single_state(), "{mode:?}");
            // The first job of the path is on the other side of the word boundary.
            assert_eq!(times(&*top_down(&c, &[0, 1, 2]))[..2], [5, 5], "{mode:?}");
        }
    }

    #[test]
    fn a_relaxed_diagram_with_a_stronger_relaxation_has_no_more_paths() {
        // The three relaxations never lose a solution (tested above). A stronger one removes
        // more edges on the nodes it merges, so for the same merging it keeps no more paths.
        // Merging depends on the order of the states, so what is compared is the count of the
        // paths that are not solutions, for the width where all of them merge the same way:
        // width 1, one node per layer.
        let mut rng = Lcg(2026);
        for _ in 0..60 {
            let n = 2 + rng.below(4) as usize;
            let spec = random_spec(&mut rng, n);
            let setup = random_setup(&mut rng, n);
            let paths = |mode| {
                let mdd = compile(
                    problem_setup(mode, &spec, &vec![], &setup),
                    (0..n).collect(),
                    1,
                );
                accepted(&mdd).len()
            };
            let last_job = paths(SetupRelaxation::LastJob);
            let last_set = paths(SetupRelaxation::LastSet);
            let arc_times = paths(SetupRelaxation::ArcTimes);
            assert!(last_set <= last_job, "{spec:?} {setup:?}");
            assert!(arc_times <= last_set, "{spec:?} {setup:?}");
        }
    }
}
