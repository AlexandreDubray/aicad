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
//! Setup times are not part of this constraint (yet).
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
//! ## Node merging
//!
//! Merging two nodes merges the sets as in [`AllDifferent`] (intersection of $A$, union of $S$) and
//! takes, for each time bound, the smallest best case and the largest worst case in the top-down
//! direction (resp. the largest and the smallest in the bottom-up one). The merged node can only
//! be more permissive than each of them, so this is a relaxation.
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
//! 4. a precedence is broken: some job $i \ll j$ is in no set $S$ of the parent, so no path above
//!    has placed it, or some job $j \ll k$ is in no set $S$ of the child, so no path below will.
//!
//! Rule 3 is exact on a single path, and on an exact node: it is the schedule that is built, and
//! the deadlines of the jobs after the edge are taken into account through $l$. With merged
//! nodes it is a necessary condition only. Rule 4 is exact as well: on a single path, a job placed
//! before one of its predecessors is caught at the edge that places it. It needs no state of its
//! own, the sets $S$ that the permutation already carries are enough.
//!
//! ## Single state
//!
//! A node is a single state when $A = S$ (all the paths place the same jobs) and each time bound
//! has its best and worst case equal (all the paths give the same time, so the future of the
//! node does not depend on the path). Two paths that place the same jobs in different orders
//! usually end at different times, so this is stricter than for [`AllDifferent`].
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
//!   state. Setup times would need a number per last job.
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
    /// `problem` is read for the domains of the variables, as in [`AllDifferent::new`].
    ///
    /// # Panics
    ///
    /// If a variable is repeated, if the three lists do not have one entry per variable, if a
    /// duration is negative, or if the precedences are not valid (see
    /// [`check_precedences`](Self::check_precedences)).
    pub fn new(
        variables: Vec<VariableIndex>,
        release: Vec<isize>,
        deadline: Vec<isize>,
        duration: Vec<isize>,
        precedences: Option<Vec<(usize, usize)>>,
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

    /// Number of jobs, which is the size of the scope.
    fn len(&self) -> usize {
        self.variables.len()
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
            let end = time
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
        let end = parent
            .ect_best
            .max(self.jobs.release[job])
            .saturating_add(self.jobs.duration[job]);
        if end > child.lst_best.min(self.jobs.deadline[job]) {
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
            return;
        }
        // An edge that is not a job is removed by the filtering; it contributes nothing.
        let Some(job) = job_of(assignment, self.jobs.release.len()) else {
            return;
        };
        self.permutation
            .update(&parent.permutation, assignment, true);
        if !parent.has_no_ect() {
            let release = self.jobs.release[job];
            let duration = self.jobs.duration[job];
            self.ect_best = self
                .ect_best
                .min(parent.ect_best.max(release).saturating_add(duration));
            self.ect_worst = self
                .ect_worst
                .max(parent.ect_worst.max(release).saturating_add(duration));
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
            return;
        }
        let Some(job) = job_of(assignment, self.jobs.release.len()) else {
            return;
        };
        self.permutation
            .update_backward(&child.permutation, assignment, true);
        if !child.has_no_lst() {
            let deadline = self.jobs.deadline[job];
            let duration = self.jobs.duration[job];
            self.lst_best = self
                .lst_best
                .max(child.lst_best.min(deadline).saturating_sub(duration));
            self.lst_worst = self
                .lst_worst
                .min(child.lst_worst.min(deadline).saturating_sub(duration));
        }
    }

    fn merge(&mut self, other: &dyn ConstraintProperty) {
        let other = Self::downcast(other, "merge");
        self.permutation.merge(&other.permutation);
        self.ect_best = self.ect_best.min(other.ect_best);
        self.ect_worst = self.ect_worst.max(other.ect_worst);
        self.lst_best = self.lst_best.max(other.lst_best);
        self.lst_worst = self.lst_worst.min(other.lst_worst);
    }

    fn order_key(&self) -> Vec<f64> {
        // One allocation of the right size: this runs for every node when merging.
        let [all, some] = self.permutation.order_key_parts();
        vec![
            all,
            some,
            self.ect_best as f64,
            self.ect_worst as f64,
            self.lst_best as f64,
            self.lst_worst as f64,
        ]
    }

    fn hash(&self, hasher: &mut dyn Hasher) {
        self.permutation.hash(hasher);
        hasher.write_isize(self.ect_best);
        hasher.write_isize(self.ect_worst);
        hasher.write_isize(self.lst_best);
        hasher.write_isize(self.lst_worst);
    }

    fn eq(&self, other: &dyn ConstraintProperty) -> bool {
        let other = Self::downcast(other, "eq");
        ConstraintProperty::eq(&self.permutation, &other.permutation)
            && self.ect_best == other.ect_best
            && self.ect_worst == other.ect_worst
            && self.lst_best == other.lst_best
            && self.lst_worst == other.lst_worst
    }

    /// $A = S$ and each time bound that holds a value has its two cases equal (see the module
    /// documentation).
    fn is_single_state(&self) -> bool {
        self.permutation.is_single_state()
            && (self.has_no_ect() || self.ect_best == self.ect_worst)
            && (self.has_no_lst() || self.lst_best == self.lst_worst)
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
        let mut c = Disjunctive::new(vars.clone(), r, d, p, None, &problem);
        c.update_variable_ordering(&vars);
        c
    }

    /// The problem with only the constraint over its `n` variables.
    fn problem_of(spec: &Spec) -> Problem {
        let n = spec.len() as isize;
        let mut problem = Problem::default();
        let vars = problem.add_variables(spec.len(), (0..n).collect());
        let (r, d, p) = lists(spec);
        disjunctive(&mut problem, vars, r, d, p, None);
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
        let n = key.len();
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
            &problem,
        );
    }

    #[test]
    #[should_panic(expected = "deadline has 1 entries for a scope of 2")]
    fn a_list_of_the_wrong_length_is_refused() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1]);
        Disjunctive::new(vars, vec![0, 0], vec![9], vec![1, 1], None, &problem);
    }

    #[test]
    #[should_panic(expected = "job 1 has a negative duration")]
    fn a_negative_duration_is_refused() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1]);
        Disjunctive::new(vars, vec![0, 0], vec![9, 9], vec![1, -1], None, &problem);
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
        Disjunctive::new(vars, vec![0; 2], vec![9; 2], vec![1; 2], None, &problem)
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
        let empty = Disjunctive::new(vec![], vec![], vec![], vec![], None, &problem);
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
        disjunctive(&mut problem, vec![vars[0], vars[2], vars[4]], r, d, p, None);
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
        disjunctive(&mut problem, vars.clone(), r, d, p, None);
        let (r, d, p) = lists(&b);
        disjunctive(&mut problem, vars, r, d, p, None);
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
        disjunctive(&mut problem, vars, r, d, p, None);
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
        let mut c = Disjunctive::new(vars.clone(), r, d, p, Some(precedences.clone()), &problem);
        c.update_variable_ordering(&vars);
        c
    }

    /// Like `problem_of`, with precedences.
    fn problem_prec(spec: &Spec, precedences: &Pairs) -> Problem {
        let n = spec.len() as isize;
        let mut problem = Problem::default();
        let vars = problem.add_variables(spec.len(), (0..n).collect());
        let (r, d, p) = lists(spec);
        disjunctive(&mut problem, vars, r, d, p, Some(precedences.clone()));
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
            &problem,
        );
        let empty = Disjunctive::new(vars, r, d, p, Some(vec![]), &problem);
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
            &problem,
        );
        let b = Disjunctive::new(vars, r, d, p, Some(vec![(0, 2), (2, 1)]), &problem);
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
}
