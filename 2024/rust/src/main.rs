use clap::Parser;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::iter::zip;
use std::path::Path;

fn read_file(data_dir: &Path, day: i32) -> String {
    let file_path = data_dir.join(format!("day{}.txt", day));
    fs::read_to_string(file_path.clone())
        .unwrap_or_else(|_| panic!("Can't read file {}", file_path.display()))
}

fn day1(data_dir: &Path) {
    let contents = read_file(data_dir, 1);

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

    let sum: i32 = zip(&first_list, &second_list)
        .map(|(left, right)| (*left - *right).abs())
        .sum();

    let mut counts: HashMap<i32, i32> = HashMap::new();
    for i in second_list {
        *counts.entry(i).or_default() += 1;
    }

    let similarity: i32 = first_list
        .into_iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|v| v * *counts.entry(v).or_default())
        .sum();

    println!("Day 1: {} P2: {}", sum, similarity);
}

fn day2(data_dir: &Path) {
    let contents = read_file(data_dir, 2);

    let reports: Vec<Vec<i32>> = contents
        .lines()
        .map(|line| {
            line.split(' ')
                .map(|chars| chars.parse::<i32>().unwrap())
                .collect()
        })
        .collect();

    fn check_report(report: &Vec<i32>) -> bool {
        let mut increasing = report.clone();
        increasing.sort();
        let mut decreasing = report.clone();
        decreasing.sort();
        decreasing.reverse();

        if *report != increasing && *report != decreasing {
            return false;
        }

        zip(&increasing[..increasing.len() - 1], &increasing[1..])
            .map(|(smaller, larger)| {
                let diff = *larger - *smaller;
                diff > 0 && diff < 4
            })
            .all(|x| x)
    }

    let total_valid = reports
        .clone()
        .into_iter()
        .map(|report| check_report(&report))
        .filter(|x| *x)
        .count();

    // Reports are fairly short and I can't be bothered to re-write the checking logic properly
    let valid_with_dampner = reports
        .into_iter()
        .map(|report| {
            if check_report(&report) {
                return true;
            }
            for idx in 0..report.len() {
                let mut candidate = report.clone();
                candidate.remove(idx);
                if check_report(&candidate) {
                    return true;
                }
            }
            false
        })
        .filter(|x| *x)
        .count();

    println!("Day 2: {} P2: {}", total_valid, valid_with_dampner);
}

fn day3(data_dir: &Path) {
    let contents = read_file(data_dir, 3);

    let mut parts: Vec<(i32, i32)> = Vec::new();

    let mut enabled = true;
    let mut enabled_parts: Vec<(i32, i32)> = Vec::new();

    for (idx, c) in contents.chars().enumerate() {
        if c != '(' {
            continue;
        }
        if contents[idx - 2..idx + 2] == *"do()" {
            enabled = true;
        }
        if contents[idx - 5..idx + 2] == *"don't()" {
            enabled = false;
        }
        if contents[idx - 3..idx] != *"mul" {
            continue;
        }
        let mut arg1_end: Option<usize> = None;
        let mut arg2_end: Option<usize> = None;
        for (idx_end, c_end) in contents[idx..idx + 9].chars().enumerate() {
            if c_end == ',' {
                arg1_end = Some(idx + idx_end);
            }
            if c_end == ')' {
                arg2_end = Some(idx + idx_end);
                break;
            }
        }
        if arg1_end.is_none() || arg2_end.is_none() || arg1_end.unwrap() > arg2_end.unwrap() {
            continue;
        }

        let arg1 = contents[idx + 1..arg1_end.unwrap()].parse::<i32>();
        let arg2 = contents[arg1_end.unwrap() + 1..arg2_end.unwrap()].parse::<i32>();

        if arg1.is_ok() && arg2.is_ok() {
            let part = (arg1.unwrap(), arg2.unwrap());
            parts.push(part);
            if enabled {
                enabled_parts.push(part);
            }
        }
    }

    let sum: i32 = parts.into_iter().map(|(a1, a2)| a1 * a2).sum();
    let sum2: i32 = enabled_parts.into_iter().map(|(a1, a2)| a1 * a2).sum();
    println!("Day 3: {} P2: {}", sum, sum2);
}

#[derive(Debug, Clone)]
struct Matrix<T> {
    rows: i32,
    cols: i32,
    data: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Default + Clone,
{
    pub fn new(rows: i32, cols: i32) -> Matrix<T> {
        Matrix {
            rows,
            cols,
            data: vec![T::default(); (rows * cols) as usize],
        }
    }

    fn get_index(&self, row: i32, col: i32) -> Option<usize> {
        if row < 0 || row >= self.rows || col < 0 || col >= self.cols {
            return None;
        }
        Some((row * self.cols + col) as usize)
    }

    pub fn set(&mut self, row: i32, col: i32, value: T) {
        let index = self.get_index(row, col);
        self.data[index.unwrap()] = value;
    }

    pub fn get(&self, row: i32, col: i32) -> Option<&T> {
        let index = self.get_index(row, col);
        self.data.get(index?)
    }
}

fn day4(data_dir: &Path) {
    let contents = read_file(data_dir, 4);

    let lines: Vec<&str> = contents.lines().collect();

    let mut data: Matrix<char> = Matrix::new(lines.len() as i32, lines[0].len() as i32);
    for (row, line) in lines.into_iter().enumerate() {
        for (col, char) in line.chars().enumerate() {
            data.set(row as i32, col as i32, char);
        }
    }

    let mut xmas_total = 0;

    let directions: Vec<(i32, i32)> = vec![
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ];
    let word_to_find = "XMAS";

    for row in 0..(data.rows as i32) {
        for col in 0..(data.cols as i32) {
            if *data.get(row, col).unwrap() != 'X' {
                continue;
            }
            for (row_dir, col_dir) in directions.iter() {
                let mut candidate = String::new();
                for (char_idx, char) in word_to_find.chars().enumerate().skip(1) {
                    let char_get = data.get(
                        row + char_idx as i32 * row_dir,
                        col + char_idx as i32 * col_dir,
                    );
                    if char_get.is_none() || *char_get.unwrap() != char {
                        break;
                    }
                    candidate.push(*char_get.unwrap());
                    if char_idx == word_to_find.len() - 1 {
                        xmas_total += 1;
                    }
                }
            }
        }
    }

    let mut x_mas_total = 0;

    // TODO: Try using higher order functions here
    for row in 0..(data.rows as i32) {
        for col in 0..(data.cols as i32) {
            if *data.get(row, col).unwrap() != 'A' {
                continue;
            }
            let mut good_dir = [false; 2];
            for (dir_idx, (row_dir, col_dir)) in
                directions.iter().skip(1).step_by(2).take(2).enumerate()
            {
                let forward = data.get(row + row_dir, col + col_dir);
                let backward = data.get(row + row_dir * -1, col + col_dir * -1);
                if forward.is_none() || backward.is_none() {
                    continue;
                }
                let f = *forward.unwrap();
                let b = *backward.unwrap();
                if (f, b) == ('M', 'S') || (f, b) == ('S', 'M') {
                    good_dir[dir_idx] = true;
                }
            }

            if good_dir.iter().all(|v| *v) {
                x_mas_total += 1;
            }
        }
    }

    println!("Day 4: {} P2: {}", xmas_total, x_mas_total);
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "data")]
    data_dir: String,
}

fn main() {
    let args = Args::parse();

    let data_dir = Path::new(args.data_dir.trim());
    day1(data_dir);
    day2(data_dir);
    day3(data_dir);
    day4(data_dir);
}
