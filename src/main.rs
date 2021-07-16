use std::io::stdin;
use std::io::BufRead;
use std::io::BufReader;

use chawuek::Chawuek;

fn main() {
    let chawuek = Chawuek::new().unwrap();
    for line_opt in BufReader::new(stdin()).lines() {
        let line = line_opt.unwrap();
        let out = chawuek.tokenize(&line).unwrap().join("|");
        println!("{}", out);
    }
}
