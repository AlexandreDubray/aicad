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

    let _number_colors = tokens.next().unwrap().parse::<usize>().unwrap();
    let number_nodes = tokens.next().unwrap().parse::<usize>().unwrap();
    let mut variables: Vec<VariableIndex> = vec![];

    let mut problem = Problem::default();

    for _ in 0..number_nodes {
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
        variables.push(variable);
    }

    while let Some(token) = tokens.next() {
        let u = token.parse::<usize>().unwrap();
        let v = tokens.next().unwrap().parse::<usize>().unwrap();
        //all_different(&mut problem, vec![variables[u], variables[v]]);
        not_equals(&mut problem, variables[u], variables[v]);
    }

    let mut mdd = Mdd::new(problem, max_width, OrderingHeuristic::MinDomMaxLinked, MergeHeuristic::LessRelaxed);
    mdd.refine();
    if let Some(solution) = mdd.get_solution() {
        assert!(mdd.is_solution(&solution));
    }
    println!("{:?}", mdd);
}
