pub mod constraints;
pub mod core;
pub mod mdd;

#[cfg(test)]
mod tests {

    use crate::core::problem::*;
    use crate::constraints::all_different::AllDifferent;
    use crate::mdd::mdd::Mdd;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn sudoku() {
        // a 4 x 4 sudoku grid
        let mut problem = Problem::default();
        let vars = problem.add_variables(16, vec![0, 1, 2, 3]);

        // Row constraints
        problem.add_constraint(AllDifferent::new(vec![vars[0], vars[1], vars[2], vars[3]]));
        problem.add_constraint(AllDifferent::new(vec![vars[4], vars[5], vars[6], vars[7]]));
        problem.add_constraint(AllDifferent::new(vec![vars[8], vars[9], vars[10], vars[11]]));
        problem.add_constraint(AllDifferent::new(vec![vars[12], vars[13], vars[14], vars[15]]));

        // Column constraints
        problem.add_constraint(AllDifferent::new(vec![vars[0], vars[4], vars[8], vars[12]]));
        problem.add_constraint(AllDifferent::new(vec![vars[1], vars[5], vars[9], vars[13]]));
        problem.add_constraint(AllDifferent::new(vec![vars[2], vars[6], vars[10], vars[14]]));
        problem.add_constraint(AllDifferent::new(vec![vars[3], vars[7], vars[11], vars[15]]));

        // Block constraints
        problem.add_constraint(AllDifferent::new(vec![vars[0], vars[1], vars[4], vars[5]]));
        problem.add_constraint(AllDifferent::new(vec![vars[2], vars[3], vars[6], vars[7]]));
        problem.add_constraint(AllDifferent::new(vec![vars[8], vars[9], vars[12], vars[13]]));
        problem.add_constraint(AllDifferent::new(vec![vars[10], vars[11], vars[14], vars[15]]));

        let mut mdd = Mdd::new(&problem);
        let repr = mdd.as_graphviz();
        let path = "mdd.dot";
        let mut output = File::create(path).unwrap();
        write!(output, "{}", repr);

        /*
         * int n = 4;
         *
         * for x in .... do
         *      problem.addConstraint(AllDifferent(x));
         *
         * problem.solve();
         *
         */
        let x = 1;
        assert!(x == 1);
    }
}
