//! Chapter "Modelling".

// [docs:start model]
use aicad::constraints::{among, gcc, regular};
use aicad::modelling::{Problem, equal};

const DAYS: usize = 5;
const NURSES: usize = 3;
const OFF: isize = 0;
const DAY: isize = 1;
const NIGHT: isize = 2;

/// The shift roster of a small team: `roster[n][d]` is the shift of nurse `n` on day `d`.
pub fn roster_problem() -> Problem {
    let mut problem = Problem::default();
    let roster: Vec<_> = (0..NURSES)
        .map(|_| problem.add_variables(DAYS, vec![OFF, DAY, NIGHT]))
        .collect();

    // Every day has exactly one nurse on the night shift and one or two on the day shift.
    for d in 0..DAYS {
        let column = roster.iter().map(|nurse| nurse[d]).collect();
        gcc(&mut problem, column, vec![(NIGHT, 1, 1), (DAY, 1, 2)]);
    }

    // A nurse works at most five days a week.
    for row in &roster {
        among(&mut problem, row.clone(), vec![DAY, NIGHT], 0, 5);
    }

    // A night shift is never directly followed by a day shift. State 0 is "not after a night",
    // state 1 is "after a night"; the columns of the table are the sorted values of the domain
    // (off, day, night), and `None` forbids a transition.
    let after_night = vec![
        vec![Some(0), Some(0), Some(1)], // state 0: off -> 0, day -> 0, night -> 1
        vec![Some(0), None, Some(1)],    // state 1: off -> 0, day is forbidden, night -> 1
    ];
    for row in &roster {
        regular(
            &mut problem,
            row.clone(),
            after_night.clone(),
            0,
            vec![0, 1],
        );
    }

    // The first nurse is on leave on the last day.
    equal(&mut problem, roster[0][DAYS - 1], OFF);
    problem
}
// [docs:end model]

#[test]
fn a_roster_model_checks_assignments() {
    // [docs:start check]
    let problem = roster_problem();
    println!(
        "{} variables, {} constraints",
        problem.number_variables(),
        problem.number_constraints()
    );

    // An assignment lists one value per variable, in the order of their indices.
    let everyone_off = vec![OFF; problem.number_variables()];
    assert!(!problem.is_solution(&everyone_off));

    let violated = problem
        .iter_constraints()
        .filter(|&c| !problem[c].is_satisfied(&everyone_off))
        .count();
    println!(
        "{violated} of {} constraints are violated",
        problem.number_constraints()
    );
    // [docs:end check]
    assert_eq!(problem.number_variables(), NURSES * DAYS);
    assert_eq!(problem.number_constraints(), DAYS + NURSES + NURSES);
    assert!(violated > 0);
}
