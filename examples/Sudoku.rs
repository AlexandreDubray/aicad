use std::env;
use std::process;
use std::io::{self, BufRead, BufReader};
use std::fs::File;

use aicad::modelling::*;
use aicad::mdd::*;
use aicad::mdd::heuristics::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 && args.len() != 4 {
        eprintln!("Error: Wrong number of arguments. Usage: {} [inputFile] maximum-width (symbolic|neural)", args[0]);
        process::exit(1);
    }
    let (reader, max_width, _use_neural_heuristic): (Box<dyn BufRead>, usize, bool) = if args.len() == 4 {
        let file = File::open(&args[1]).expect("Failed to open input file");
        (Box::new(BufReader::new(file)), args[2].parse::<usize>().unwrap(), args[3] == "neural")
    } else {
        (Box::new(BufReader::new(io::stdin())), args[1].parse::<usize>().unwrap(), args[2] == "neural")
    };
    let mut tokens = reader.lines().flat_map(|line| line.unwrap().split_whitespace().map(String::from).collect::<Vec<_>>());

    let n = tokens.next().unwrap().parse::<usize>().unwrap();
    let block_size = (n as f64).sqrt() as usize;
    let mut rows: Vec<Vec<VariableIndex>> = (0..n).map(|_| vec![]).collect();
    let mut cols: Vec<Vec<VariableIndex>> = (0..n).map(|_| vec![]).collect();
    let mut blocks: Vec<Vec<VariableIndex>> = (0..n).map(|_| vec![]).collect();

    let mut problem = Problem::default();

    for (row_id, row) in rows.iter_mut().enumerate() {
        for (col_id, col) in cols.iter_mut().enumerate() {
            let domain_size = tokens.next().unwrap().parse::<usize>().unwrap();
            let mut domain: Vec<isize> = vec![];
            let mut probabilities: Vec<f64> = vec![];
            for _ in 0..domain_size {
                let value = tokens.next().unwrap().parse::<isize>().unwrap();
                let proba = tokens.next().unwrap().parse::<f64>().unwrap();
                domain.push(value);
                probabilities.push(proba);
            }

            let variable = problem.add_variable(domain, Some(probabilities));
            row.push(variable);
            col.push(variable);
            let block_id = (row_id / block_size) * block_size + (col_id / block_size);
            blocks[block_id].push(variable);
        }
    }

    for row in rows.into_iter() {
        all_different(&mut problem, row);
    }

    for col in cols.into_iter() {
        all_different(&mut problem, col);
    }

    for block in blocks.into_iter() {
        all_different(&mut problem, block);
    }

    let mut mdd = Mdd::new(problem, max_width, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
    mdd.refine();
    let solution = mdd.get_solution().unwrap();
    assert!(mdd.is_solution(&solution));
}
