use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::cmp::{max, min, Ordering, PartialEq};
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

type Loc = (i32, i32);

fn get_neighbours(loc: Loc, diagonals: bool) -> Vec<Loc> {
    let mut neighbours: Vec<Loc> = Vec::new();
    let mut directions = vec![(-1, 0), (0, 1), (1, 0), (0, -1)];
    if diagonals {
        directions.extend([(-1, 1), (1, 1), (1, -1), (-1, -1)]);
    }
    for dir in directions {
        neighbours.push((loc.0 + dir.0, loc.1 + dir.1));
    }
    neighbours
}

fn check_loc(loc: Loc, shape: (i32, i32)) -> bool {
    loc.0 >= 0 && loc.0 < shape.0 && loc.1 >= 0 && loc.1 < shape.1
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
        if !check_loc((row, col), (self.rows, self.cols)) {
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
    let mut obstractions: HashSet<Loc> = HashSet::new();
    let mut starting_pos: Loc = (0, 0);
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

    fn run_guard_loop(shape: (i32, i32), obstractions: HashSet<Loc>, starting_pos: Loc) -> usize {
        let mut guard_pos = starting_pos;
        let directions: HashMap<usize, (i32, i32)> = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            .iter()
            .cloned()
            .enumerate()
            .collect();
        let mut guard_direction: usize = 0;

        let mut visited: HashSet<Loc> = HashSet::new();
        let mut total_steps = 0;

        let get_next_pos = |guard_pos: Loc, guard_direction: usize| {
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
            if !check_loc(next_pos, shape) {
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

    let mut locations: HashMap<char, Vec<Loc>> = HashMap::new();

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

    fn get_all_antinodes(locations: &[Loc], rows: i32, cols: i32, harmonics: bool) -> HashSet<Loc> {
        let mut antinodes: HashSet<Loc> = HashSet::new();
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

    let mut heights: HashMap<u32, HashSet<Loc>> = HashMap::new();

    for (row_idx, line) in contents.lines().enumerate() {
        for (col_idx, char) in line.chars().enumerate() {
            heights
                .entry(char.to_digit(10).unwrap())
                .or_default()
                .insert((row_idx as i32, col_idx as i32));
        }
    }

    type ReachablePeaks = HashSet<Loc>;
    let mut reachable_by_height: HashMap<u32, HashMap<Loc, ReachablePeaks>> = HashMap::new();
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

    let mut reachable_by_height_num: HashMap<u32, HashMap<Loc, i32>> = HashMap::new();
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

fn day11(data_dir: &Path) {
    let contents = read_file(data_dir, 11);

    let stones: Vec<i64> = contents
        .trim()
        .split(' ')
        .map(|s| s.parse::<i64>().unwrap())
        .collect();

    // initial implementation, not using in interest of execution speed
    #[allow(dead_code)]
    fn modify_stones(stones: &mut Vec<i64>, iters: usize) {
        let pb = ProgressBar::new(iters as u64).with_prefix("Updating stones");
        let template = "{prefix} {spinner} [{elapsed}] {wide_bar} {pos}/{len} ({eta})";
        pb.set_style(ProgressStyle::with_template(template).unwrap());
        for _ in 0..iters {
            let mut idx = 0;
            while idx < stones.len() {
                let stone = stones[idx];
                if stone == 0 {
                    stones[idx] = 1;
                } else if (stone as f32).log10().floor() as usize % 2 == 1 {
                    let string = stone.to_string();
                    let (first, second) = string.split_at(string.len() / 2);
                    stones.remove(idx);
                    stones.insert(idx, first.parse::<i64>().unwrap());
                    stones.insert(idx + 1, second.parse::<i64>().unwrap());
                    idx += 1;
                } else {
                    stones[idx] *= 2024;
                }
                idx += 1;
            }
            pb.inc(1);
        }
        pb.finish_and_clear();
    }

    fn sum_up_stones_dyn(stones: &mut [i64], iters: usize) -> usize {
        // How many stones are produced after iterating number x by y times (x,y)
        let mut cache: HashMap<(i64, usize), usize> = HashMap::new();

        fn helper(num: i64, iter: usize, cache: &mut HashMap<(i64, usize), usize>) -> usize {
            if cache.contains_key(&(num, iter)) {
                return cache[&(num, iter)];
            }
            if iter == 0 {
                return 1;
            }
            let mut new: Vec<i64> = Vec::new();
            if num == 0 {
                new.push(1);
            } else if (num as f32).log10().floor() as usize % 2 == 1 {
                let string = num.to_string();
                let (first, second) = string.split_at(string.len() / 2);
                new.push(first.parse::<i64>().unwrap());
                new.push(second.parse::<i64>().unwrap());
            } else {
                new.push(num * 2024);
            }
            let mut total = 0;
            for val in new.into_iter() {
                let res = helper(val, iter - 1, cache);
                cache.insert((val, iter - 1), res);
                total += res;
            }
            total
        }

        let mut total = 0;
        for stone in stones.iter() {
            total += helper(*stone, iters, &mut cache);
        }
        total
    }
    let mut stones_p1 = stones.to_vec();
    let sum = sum_up_stones_dyn(&mut stones_p1, 25);
    let mut stones_p2 = stones.to_vec();
    let sum2 = sum_up_stones_dyn(&mut stones_p2, 75);

    println!("Day 11: {} P2: {}", sum, sum2);
}

fn day12(data_dir: &Path) {
    let contents = read_file_helper(data_dir, 12, false);

    let mut regions: HashMap<usize, HashSet<Loc>> = HashMap::new();
    let mut loc2char: HashMap<Loc, char> = HashMap::new();
    let mut loc2region: HashMap<Loc, usize> = HashMap::new();

    let mut counter = 0;
    for (row, line) in contents.lines().enumerate() {
        for (col, char) in line.chars().enumerate() {
            let loc = (row as i32, col as i32);
            let region = counter;
            counter += 1;
            regions.entry(region).or_default().insert(loc);
            loc2char.insert(loc, char);
            loc2region.insert(loc, region);

            for neighbour in get_neighbours(loc, false) {
                if loc2char.contains_key(&neighbour) && *loc2char.get(&neighbour).unwrap() == char {
                    let current_region = *loc2region.get(&loc).unwrap();
                    let other_region = *loc2region.get(&neighbour).unwrap();
                    if other_region != current_region {
                        let joined_region = min(current_region, other_region);
                        let region_to_remove = max(current_region, other_region);
                        let region_to_remove_cells =
                            regions.get(&region_to_remove).unwrap().clone();

                        regions
                            .entry(joined_region)
                            .or_default()
                            .extend(region_to_remove_cells.clone());

                        for cell in region_to_remove_cells {
                            loc2region.insert(cell, joined_region);
                        }

                        regions.remove(&region_to_remove);
                    }
                }
            }
        }
    }

    let perimiters: HashMap<usize, usize> = regions
        .iter()
        .map(|(region, cells)| {
            let mut perimiter = 0;
            for cell in cells {
                for neighbour in get_neighbours(*cell, false) {
                    if cells.contains(&neighbour) {
                        continue;
                    }
                    perimiter += 1;
                }
            }
            (*region, perimiter)
        })
        .clone()
        .collect();

    let total_price: usize = regions
        .iter()
        .map(|(region, cells)| cells.len() * perimiters.get(region).unwrap())
        .sum();

    let sides: HashMap<usize, usize> = regions
        .iter()
        .map(|(region, cells)| {
            let mut side_cells: HashMap<(usize, i32), HashSet<i32>> = HashMap::new();
            for cell in cells {
                for (idx, neighbour) in get_neighbours(*cell, false).iter().enumerate() {
                    if cells.contains(neighbour) {
                        continue;
                    }
                    let horizontal = idx % 2 == 0;
                    let loc = if horizontal { neighbour.0 } else { neighbour.1 };
                    let cell_loc = if horizontal { cell.1 } else { cell.0 };
                    side_cells.entry((idx, loc)).or_default().insert(cell_loc);
                }
            }
            let mut sides = 0;
            for values in side_cells.values() {
                let mut v_vec: Vec<i32> = values.iter().cloned().collect();
                v_vec.sort();
                let mut prev = -2;
                for value in v_vec {
                    if value - prev > 1 {
                        sides += 1;
                    }
                    prev = value;
                }
            }

            (*region, sides)
        })
        .clone()
        .collect();

    let discount_price: usize = regions
        .iter()
        .map(|(region, cells)| cells.len() * sides.get(region).unwrap())
        .sum();

    println!("Day 12: {} P2: {}", total_price, discount_price);
}

fn day13(data_dir: &Path) {
    let contents = read_file_helper(data_dir, 13, false);

    fn str_coords(line: &str) -> (i64, i64) {
        let (_, rest) = line.split_once('X').to_owned().unwrap();
        let (x, rest) = rest.split_once(',').to_owned().unwrap();
        let (_, y) = rest.split_once('Y').to_owned().unwrap();
        let to_i64 = |v: &str| v[1..].parse::<i64>().unwrap();
        (to_i64(x), to_i64(y))
    }

    fn check_prize(inputs: &Vec<(i64, i64)>) -> Option<usize> {
        let [a, b, target] = <[(i64, i64); 3]>::try_from(inputs.to_vec()).unwrap();

        let mut b_mult = target.0 / b.0;
        loop {
            if b_mult < 0 {
                break;
            }
            let a_target = (target.0 - b.0 * b_mult, target.1 - b.1 * b_mult);
            if a_target.0 % a.0 == 0
                && a_target.1 % a.1 == 0
                && a_target.0 / a.0 == a_target.1 / a.1
            {
                return Some((a_target.0 / a.0 * 3 + b_mult) as usize);
            }
            b_mult -= 1;
        }

        None
    }

    // Why is there maths in my coding challenge???
    fn check_prize_math(inputs: &Vec<(i64, i64)>, extra_target: i64) -> Option<usize> {
        let [a, b, mut t] = <[(i64, i64); 3]>::try_from(inputs.to_vec()).unwrap();
        t = (t.0 + extra_target, t.1 + extra_target);

        // a.0 * i + b.0 * j = t.0
        // a.1 * i + b.1 * j = t.1
        // i = (t.0 - b.0 * j) / a.0
        // a.1 * (t.0 - b.0 * j) / a.0 + b.1 * j = t.1
        // a.1 * (t.0 - b.0 * j) + b.1 * j * a.0 = t.1 * a.0
        // a.1 * t.0 - a.1 * b.0 * j + b.1 * j * a.0 = t.1 * a.0
        // - a.1 * b.0 * j + b.1 * j * a.0 = t.1 * a.0 - a.1 * t.0
        // j * (- a.1 * b.0 + b.1 * a.0) = t.1 * a.0 - a.1 * t.0

        // j = (a.0 * t.1 - a.1 * t.0) / (a.0 * b.1 - a.1 * b.0)
        // i = (t.0 - b.0 * j) / a.0

        let b_mult = (a.0 * t.1 - a.1 * t.0) / (a.0 * b.1 - a.1 * b.0);
        let a_mult = (t.0 - b.0 * b_mult) / a.0;
        if a.0 * a_mult + b.0 * b_mult == t.0 && a.1 * a_mult + b.1 * b_mult == t.1 {
            Some((a_mult * 3 + b_mult) as usize)
        } else {
            None
        }
    }

    let machines: Vec<Vec<(i64, i64)>> = contents
        .trim()
        .split("\n\n")
        .map(|lines| {
            lines
                .split('\n')
                .map(str_coords)
                .collect::<Vec<(i64, i64)>>()
        })
        .collect();

    let score: usize = machines.iter().flat_map(check_prize).sum();
    let score2: usize = machines
        .iter()
        .flat_map(|m| check_prize_math(m, 10000000000000))
        .sum();

    println!("Day 13: {} P2: {}", score, score2);
}
fn day14(data_dir: &Path) {
    let contents = read_file_helper(data_dir, 14, false);

    let width = 101;
    let height = 103;
    // let width = 11;
    // let height = 7;

    type Robot = (Loc, Loc);

    let robots: Vec<(Loc, Loc)> = contents
        .lines()
        .map(|line| {
            let (pos_str, vel_str) = line.split_once(' ').unwrap();
            let to_num = |s: &str| s.parse::<i32>().unwrap();
            let convert = |s: &str| {
                let (left, right) = s[2..].split_once(',').unwrap();
                (to_num(left), to_num(right))
            };
            (convert(pos_str), convert(vel_str))
        })
        .collect();

    let calc_future_pos = |&(pos, vec): &Robot, iters: i32| {
        let mut pos2 = (pos.0 + vec.0 * iters, pos.1 + vec.1 * iters);
        while pos2.0 < 0 {
            pos2.0 += width;
        }
        while pos2.1 < 0 {
            pos2.1 += height;
        }
        (pos2.0 % width, pos2.1 % height)
    };

    let advance_all_robots = |robots: &Vec<Robot>, iters: i32| {
        robots
            .iter()
            .map(|robot| calc_future_pos(robot, iters))
            .collect::<Vec<Loc>>()
    };

    let final_pos: Vec<Loc> = advance_all_robots(&robots, 100);

    let mut quadrants: HashMap<i32, i32> = HashMap::new();
    let (half_w, half_h) = (width / 2, height / 2);
    for f_pos in final_pos.iter() {
        if f_pos.0 == half_w || f_pos.1 == half_h {
            continue;
        }
        *quadrants
            .entry(f_pos.0 / (half_w + 1) + f_pos.1 / (half_h + 1) * 2)
            .or_default() += 1;
    }

    let prod: i32 = quadrants.values().product();

    #[allow(unused_variables)]
    let print_pos = |positions: &[Loc]| {
        let mut pic = String::new();
        let pos_set: HashSet<Loc> = positions.iter().cloned().collect();
        for row in 0..height {
            for col in 0..width {
                if pos_set.contains(&(col, row)) {
                    pic.push('x');
                } else {
                    pic.push('.');
                }
            }
            pic.push('\n');
        }
        print!("{}", pic);
    };

    #[allow(unused_variables)]
    let is_tree_like = |positions: &[Loc]| {
        let mut row_counts: HashMap<i32, i32> = HashMap::new();
        let mut col_counts: HashMap<i32, i32> = HashMap::new();

        for (col, row) in positions.iter() {
            *row_counts.entry(*row).or_default() += 1;
            *col_counts.entry(*col).or_default() += 1;
        }
        *row_counts.values().max().unwrap() > 30 || *col_counts.values().max().unwrap() > 30
    };

    // for idx in 0..10000 {
    //     let idx_positions = advance_all_robots(&robots, idx);
    //     if is_tree_like(&idx_positions) {
    //         println!("{}", idx);
    //         print_pos(&idx_positions);
    //     }
    // }

    println!("Day 14: {} P2: {}", prod, "<run code and check visually>");
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
    day11(data_dir);
    day12(data_dir);
    day13(data_dir);
    day14(data_dir);
}
