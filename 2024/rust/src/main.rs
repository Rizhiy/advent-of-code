use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::cmp::{Ordering, PartialEq};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::iter::zip;
use std::path::Path;

fn read_file_helper(data_dir: &Path, day: i32, example: bool) -> String {
    let file_path = data_dir.join(format!(
        "day{}{}.txt",
        day,
        if example { "_example" } else { "" }
    ));
    fs::read_to_string(file_path.clone())
        .unwrap_or_else(|_| panic!("Can't read file {}", file_path.display()))
}

fn read_file(data_dir: &Path, day: i32) -> String {
    read_file_helper(data_dir, day, false)
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

fn day5(data_dir: &Path) {
    let contents = read_file(data_dir, 5);

    let lines: Vec<&str> = contents.lines().collect();
    let mut rules: HashSet<(i32, i32)> = HashSet::new();
    let mut manuals: Vec<Vec<i32>> = Vec::new();
    for line in lines.iter() {
        if let Some((earlier, later)) = line.split_once('|') {
            rules.insert((
                earlier.parse::<i32>().unwrap(),
                later.parse::<i32>().unwrap(),
            ));
        } else if !line.is_empty() {
            manuals.push(
                line.split(',')
                    .map(|num| num.parse::<i32>().unwrap())
                    .collect(),
            )
        }
    }

    let is_valid = |man: &Vec<i32>| {
        man.iter().enumerate().all(|(idx, earlier)| {
            man.iter()
                .skip(idx + 1)
                .all(|later| !rules.contains(&(*later, *earlier)))
        })
    };

    let sum: i32 = manuals
        .clone()
        .into_iter()
        .filter(is_valid)
        .map(|man| man[man.len() / 2])
        .sum();

    let sum_invalid: i32 = manuals
        .into_iter()
        .filter(|man| !is_valid(man))
        .map(|mut man| {
            man.sort_by(|left, right| {
                if rules.contains(&(*left, *right)) {
                    Ordering::Less
                } else if rules.contains(&(*right, *left)) {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            });
            man
        })
        .map(|r_man| r_man[r_man.len() / 2])
        .sum();

    println!("Day 5: {} P2: {}", sum, sum_invalid);
}

fn day6(data_dir: &Path) {
    let contents = read_file(data_dir, 6);

    let lines: Vec<&str> = contents.lines().collect();
    let (rows, cols) = (lines.len() as i32, lines[0].len() as i32);
    // Not using Matrix, to try another approach
    let mut obstractions: HashSet<(i32, i32)> = HashSet::new();
    let mut starting_pos: (i32, i32) = (0, 0);
    for (row_idx, row) in lines.iter().enumerate() {
        for (col_idx, char) in row.chars().enumerate() {
            let pos = (row_idx as i32, col_idx as i32);
            if char == '#' {
                obstractions.insert(pos);
            } else if char == '^' {
                starting_pos = pos;
            }
        }
    }

    fn run_guard_loop(
        shape: (i32, i32),
        obstractions: HashSet<(i32, i32)>,
        starting_pos: (i32, i32),
    ) -> usize {
        let mut guard_pos = starting_pos;
        let directions: HashMap<usize, (i32, i32)> = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            .iter()
            .cloned()
            .enumerate()
            .collect();
        let mut guard_direction: usize = 0;

        let mut visited: HashSet<(i32, i32)> = HashSet::new();
        let mut total_steps = 0;

        let get_next_pos = |guard_pos: (i32, i32), guard_direction: usize| {
            let next_move = directions[&guard_direction];
            (guard_pos.0 + next_move.0, guard_pos.1 + next_move.1)
        };

        // CBA with rays, lazy approach
        loop {
            visited.insert(guard_pos);
            let mut next_pos = get_next_pos(guard_pos, guard_direction);
            while obstractions.contains(&next_pos) {
                guard_direction = (guard_direction + 1) % directions.len();
                next_pos = get_next_pos(guard_pos, guard_direction);
            }
            if next_pos.0 < 0 || next_pos.0 >= shape.0 || next_pos.1 < 0 || next_pos.1 >= shape.1 {
                return visited.len();
            }
            // This is faster than storing and checking visited+direction
            if total_steps > shape.0 * shape.1 {
                return 0;
            }

            total_steps += 1;
            guard_pos = next_pos;
        }
    }

    let total_visited = run_guard_loop((rows, cols), obstractions.clone(), starting_pos);

    let pb = ProgressBar::new(rows as u64).with_prefix("Looking for obstractions to create loops");
    let template = "{prefix} {spinner} [{elapsed}] {wide_bar} {pos}/{len} ({eta})";
    pb.set_style(ProgressStyle::with_template(template).unwrap());
    // There is probably a smarter way to do this, but I can just wait a few seconds
    let total_new_observations: i32 = (0..rows)
        .into_par_iter()
        .map(|row_idx| {
            let mut col_new_observations = 0;
            for col_idx in 0..cols {
                let new_obstraction = (row_idx, col_idx);
                if obstractions.contains(&new_obstraction) || new_obstraction == starting_pos {
                    continue;
                }
                let mut new_obstractions = obstractions.clone();
                new_obstractions.insert(new_obstraction);
                if run_guard_loop((rows, cols), new_obstractions, starting_pos) == 0 {
                    col_new_observations += 1;
                }
            }
            pb.inc(1);
            col_new_observations
        })
        .sum();
    pb.finish_and_clear();

    println!("Day 6: {} P2: {}", total_visited, total_new_observations);
}

fn day7(data_dir: &Path) {
    let contents = read_file(data_dir, 7);

    #[derive(PartialEq)]
    enum Op {
        Add,
        Mul,
        Join,
    }

    fn check_ops(existing: i64, numbers: &[i64], target: i64, ops: &Vec<Op>) -> bool {
        if numbers.is_empty() {
            return existing == target;
        }
        for op in ops.iter() {
            if existing == 0 && (*op == Op::Mul || *op == Op::Join) {
                continue;
            }
            let res = match *op {
                Op::Add => check_ops(existing + numbers[0], &numbers[1..], target, ops),
                Op::Mul => check_ops(existing * numbers[0], &numbers[1..], target, ops),
                Op::Join => {
                    let new: i64 = format!("{}{}", existing, numbers[0]).parse().unwrap();
                    check_ops(new, &numbers[1..], target, ops)
                }
            };
            if res {
                return true;
            }
        }
        false
    }

    let input: Vec<(Vec<i64>, i64)> = contents
        .lines()
        .map(|line| {
            let (target_str, numbers_str) = line.split_once(':').unwrap();
            let target: i64 = target_str.parse().unwrap();
            let numbers: Vec<i64> = numbers_str[1..]
                .split(' ')
                .map(|s| s.trim().parse::<i64>().unwrap())
                .collect();

            (numbers, target)
        })
        .collect();
    let sum: i64 = input
        .iter()
        .filter(|&(numbers, target)| check_ops(0, numbers, *target, &vec![Op::Add, Op::Mul]))
        .map(|(_, target)| target)
        .sum();

    let sum_with_join: i64 = input
        .iter()
        .filter(|&(numbers, target)| {
            check_ops(0, numbers, *target, &vec![Op::Add, Op::Mul, Op::Join])
        })
        .map(|(_, target)| target)
        .sum();

    println!("Day 7: {} P2: {}", sum, sum_with_join);
}

fn day8(data_dir: &Path) {
    let contents = read_file(data_dir, 8);

    let mut locations: HashMap<char, Vec<(i32, i32)>> = HashMap::new();

    let lines: Vec<&str> = contents.lines().collect();

    let (rows, cols) = (
        lines.len() as i32,
        lines.iter().next().unwrap().len() as i32,
    );

    for (row_idx, line) in lines.iter().enumerate() {
        for (col_idx, char) in line.chars().enumerate() {
            if char != '.' {
                locations
                    .entry(char)
                    .or_default()
                    .push((row_idx as i32, col_idx as i32));
            }
        }
    }

    fn get_all_antinodes(
        locations: &[(i32, i32)],
        rows: i32,
        cols: i32,
        harmonics: bool,
    ) -> HashSet<(i32, i32)> {
        let mut antinodes: HashSet<(i32, i32)> = HashSet::new();
        if harmonics {
            antinodes.insert(locations[0]);
        }
        if locations.len() < 2 {
            return antinodes;
        }
        let current_ant = locations.iter().next().unwrap();
        for other_ant in locations.iter().skip(1) {
            let distance = (other_ant.0 - current_ant.0, other_ant.1 - current_ant.1);
            for mult in [-1, 1] {
                let mut scale = 1;
                loop {
                    if scale > 1 && !harmonics {
                        break;
                    }
                    let base_ant = if mult == -1 { current_ant } else { other_ant };
                    let antinode = (
                        base_ant.0 + distance.0 * mult * scale,
                        base_ant.1 + distance.1 * mult * scale,
                    );
                    if antinode.0 >= 0 && antinode.0 < rows && antinode.1 >= 0 && antinode.1 < cols
                    {
                        antinodes.insert(antinode);
                    } else {
                        break;
                    }
                    scale += 1;
                }
            }
        }
        antinodes
            .union(&get_all_antinodes(&locations[1..], rows, cols, harmonics))
            .copied()
            .collect()
    }

    let all_antinodes = locations
        .values()
        .map(|char_locations| get_all_antinodes(char_locations, rows, cols, false))
        .reduce(|acc, item| acc.union(&item).copied().collect())
        .unwrap();
    let all_antinodes_with_harmonics = locations
        .values()
        .map(|char_locations| get_all_antinodes(char_locations, rows, cols, true))
        .reduce(|acc, item| acc.union(&item).copied().collect())
        .unwrap();

    println!(
        "Day 8: {} P2: {}",
        all_antinodes.len(),
        all_antinodes_with_harmonics.len()
    );
}

fn day9(data_dir: &Path) {
    let contents = read_file(data_dir, 9);

    let digits: Vec<u32> = contents
        .trim()
        .chars()
        .map(|c| c.to_digit(10).unwrap())
        .collect();

    let mut working_digits = digits.clone();
    let mut sum: usize = 0;
    let mut counter: usize = 0;
    let mut is_file = true;
    let mut front_idx: usize = 0;
    let mut back_idx = working_digits.len() / 2;

    while !working_digits.is_empty() {
        if working_digits[0] == 0 {
            working_digits.remove(0);
            if is_file {
                front_idx += 1;
            }
            is_file = !is_file;
            continue;
        }
        if is_file {
            sum += counter * front_idx;
        } else {
            sum += counter * back_idx;
            let end_idx = working_digits.len().wrapping_sub(1);
            working_digits[end_idx] -= 1;
            if working_digits[end_idx] == 0 {
                // Remove last number and space before it
                working_digits.pop();
                working_digits.pop();
                back_idx -= 1;
            }
        }
        if working_digits.is_empty() {
            break;
        }
        working_digits[0] -= 1;
        counter += 1;
    }

    #[allow(dead_code)]
    fn print_expanded(digits: &[u32], digit_idxs: &[usize]) {
        let mut expanded = String::new();
        for (idx, digit) in digits.iter().enumerate() {
            for _ in 0..*digit {
                if idx % 2 == 1 {
                    expanded.push('.');
                } else {
                    expanded.push(char::from_digit(digit_idxs[idx] as u32, 10).unwrap());
                }
            }
        }
        println!("Expanded: {}", expanded);
    }

    let mut working_digits2 = digits.clone();
    let mut back_idx = working_digits2.len() - 1;
    let mut digit_idxs: Vec<usize> = (0..working_digits2.len())
        .map(|d| if d % 2 == 1 { 0 } else { d / 2 })
        .collect();

    while back_idx > 0 {
        let digit_to_move = working_digits2[back_idx];
        for space_idx in (0..working_digits2.len()).skip(1).step_by(2) {
            if space_idx > back_idx {
                break;
            }
            if working_digits2[space_idx] >= digit_to_move {
                // Remove digit from the end
                working_digits2[back_idx] -= digit_to_move;
                working_digits2[back_idx - 1] += digit_to_move;
                let digit_idx = digit_idxs[back_idx];
                digit_idxs[back_idx] = 0;
                // insert digit at the front
                working_digits2[space_idx] -= digit_to_move;
                working_digits2.insert(space_idx, 0);
                working_digits2.insert(space_idx + 1, digit_to_move);
                digit_idxs.insert(space_idx, 0);
                digit_idxs.insert(space_idx + 1, digit_idx);
                // adjust since we inserted at the front
                back_idx += 2;
                break;
            }
        }
        back_idx -= 2;
    }

    let mut sum2: usize = 0;
    let mut counter: usize = 0;
    for (digit, digit_idx) in working_digits2.iter().zip(digit_idxs) {
        for _ in 0..*digit {
            sum2 += counter * digit_idx;
            counter += 1;
        }
    }

    println!("Day 9: {} P2: {}", sum, sum2);
}

fn day10(data_dir: &Path) {
    let contents = read_file_helper(data_dir, 10, false);

    let mut heights: HashMap<u32, HashSet<(i32, i32)>> = HashMap::new();

    for (row_idx, line) in contents.lines().enumerate() {
        for (col_idx, char) in line.chars().enumerate() {
            heights
                .entry(char.to_digit(10).unwrap())
                .or_default()
                .insert((row_idx as i32, col_idx as i32));
        }
    }

    type ReachablePeaks = HashSet<(i32, i32)>;
    let mut reachable_by_height: HashMap<u32, HashMap<(i32, i32), ReachablePeaks>> = HashMap::new();
    for (row, col) in heights.get(&9).unwrap().iter().cloned() {
        reachable_by_height
            .entry(9)
            .or_default()
            .entry((row, col))
            .or_default()
            .insert((row, col));
    }

    for height in (0..*heights.keys().max().unwrap()).rev() {
        for (row, col) in heights.get(&height).unwrap().iter().cloned() {
            for (row_dir, col_dir) in [(-1, 0), (0, 1), (1, 0), (0, -1)].into_iter() {
                let target_loc = (row + row_dir, col + col_dir);
                let reachable_higher_all = reachable_by_height.get(&(height + 1)).unwrap();
                if !reachable_higher_all.contains_key(&(target_loc)) {
                    continue;
                }
                let reachable_higher_at_loc =
                    reachable_higher_all.get(&target_loc).unwrap().clone();
                reachable_by_height
                    .entry(height)
                    .or_default()
                    .entry((row, col))
                    .or_default()
                    .extend(reachable_higher_at_loc);
            }
        }
    }

    let sum: i32 = reachable_by_height
        .get(&0)
        .unwrap()
        .values()
        .map(|set| set.len() as i32)
        .sum();

    let mut reachable_by_height_num: HashMap<u32, HashMap<(i32, i32), i32>> = HashMap::new();
    for (row, col) in heights.get(&9).unwrap().iter().cloned() {
        reachable_by_height_num
            .entry(9)
            .or_default()
            .insert((row, col), 1);
    }

    for height in (0..*heights.keys().max().unwrap()).rev() {
        for (row, col) in heights.get(&height).unwrap().iter().cloned() {
            for (row_dir, col_dir) in [(-1, 0), (0, 1), (1, 0), (0, -1)].into_iter() {
                let target_loc = (row + row_dir, col + col_dir);
                let reachable_higher_all = reachable_by_height_num.get(&(height + 1)).unwrap();
                if !reachable_higher_all.contains_key(&(target_loc)) {
                    continue;
                }
                let reachable_higher_at_loc = *reachable_higher_all.get(&target_loc).unwrap();
                *reachable_by_height_num
                    .entry(height)
                    .or_default()
                    .entry((row, col))
                    .or_default() += reachable_higher_at_loc;
            }
        }
    }

    let sum_num: i32 = reachable_by_height_num.get(&0).unwrap().values().sum();

    println!("Day 10: {} P2: {}", sum, sum_num);
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
    day5(data_dir);
    day6(data_dir);
    day7(data_dir);
    day8(data_dir);
    day9(data_dir);
    day10(data_dir);
}
