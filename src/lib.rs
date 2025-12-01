pub mod modelling;
pub mod constraints;
pub mod mdd;

#[cfg(test)]
mod tests {

    use crate::modelling::*;
    use crate::mdd::mdd::Mdd;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn sudoku() {
        // a 4 x 4 sudoku grid
        /*
        let mut problem = Problem::default();
        let vars = problem.add_variables(16, vec![0, 1, 2, 3]);

        // Row constraints
        all_different(&mut problem, vec![vars[0], vars[1], vars[2], vars[3]]);
        all_different(&mut problem, vec![vars[4], vars[5], vars[6], vars[7]]);
        all_different(&mut problem, vec![vars[8], vars[9], vars[10], vars[11]]);
        all_different(&mut problem, vec![vars[12], vars[13], vars[14], vars[15]]);

        // Column constraints
        all_different(&mut problem, vec![vars[0], vars[4], vars[8], vars[12]]);
        all_different(&mut problem, vec![vars[1], vars[5], vars[9], vars[13]]);
        all_different(&mut problem, vec![vars[2], vars[6], vars[10], vars[14]]);
        all_different(&mut problem, vec![vars[3], vars[7], vars[11], vars[15]]);

        // Block constraints
        all_different(&mut problem, vec![vars[0], vars[1], vars[4], vars[5]]);
        all_different(&mut problem, vec![vars[2], vars[3], vars[6], vars[7]]);
        all_different(&mut problem, vec![vars[8], vars[9], vars[12], vars[13]]);
        all_different(&mut problem, vec![vars[10], vars[11], vars[14], vars[15]]);

        // Evidence
        equal(&mut problem, vars[0], 0);
        equal(&mut problem, vars[5], 1);
        equal(&mut problem, vars[11], 2);
        equal(&mut problem, vars[12], 1);
        equal(&mut problem, vars[14], 0);

        problem.set_variable_ordering((0..vars.len()).collect());

        // Evidence:
        //  x0 = 0
        //  x5 = 1
        //  x11 = 2
        //  x12 = 1
        //  x14 = 0
        //
        // Then all other values can be infered

        let mut mdd = Mdd::new(&problem);
        mdd.refine(&problem);
        //assert!(false);
        */
    }
}
