use std::fs;
use std::io;
use std::iter::zip;
use std::path::Path;

fn day1(data_dir: &Path) {
    let file_path = data_dir.join("day1.txt");
    let contents = fs::read_to_string(file_path).expect("Can't read the file");

    let (mut first_list, mut second_list): (Vec<i32>, Vec<i32>) = contents
        .lines()
        .map(|line| {
            let (first, second) = line.split_once("   ").unwrap();
            let first_num = first.parse::<i32>().unwrap();
            let second_num = second.parse::<i32>().unwrap();
            (first_num, second_num)
        })
        .unzip();

    first_list.sort();
    second_list.sort();

    let sum = zip(first_list, second_list)
        .map(|(left, right)| (left - right).abs())
        .reduce(|total, next| total + next)
        .unwrap();
    println!("Day 1: {}", sum);
}

fn main() {
    println!("Please input data dir");

    let mut input = String::new();

    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");

    let data_dir = Path::new(input.trim());
    day1(data_dir);
}
