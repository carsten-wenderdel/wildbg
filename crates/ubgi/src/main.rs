use engine::composite::CompositeEvaluator;
use engine::dice::Dice;
use engine::evaluator::Evaluator;
use engine::position::Position;
use logic::bg_move::BgMove;
use logic::wildbg_api::ScoreConfig;
use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    let evaluator = match CompositeEvaluator::try_default() {
        Ok(evaluator) => evaluator,
        Err(err) => {
            reply(
                &mut stdout,
                &format!("error missing_context evaluator_init_failed {err}"),
            );
            return;
        }
    };

    let mut context = Context::default();

    for line in stdin.lock().lines() {
        let Ok(line) = line else {
            break;
        };
        let cmd = line.trim();
        if cmd.is_empty() {
            continue;
        }

        if cmd == "ubgi" {
            reply(&mut stdout, "id name wildbg-ubgi 0.1");
            reply(&mut stdout, "id author wildbg contributors");
            reply(
                &mut stdout,
                "option name Threads type spin default 1 min 1 max 256",
            );
            reply(
                &mut stdout,
                "option name Seed type spin default 0 min 0 max 4294967295",
            );
            reply(
                &mut stdout,
                "option name Deterministic type check default true",
            );
            reply(
                &mut stdout,
                "option name EvalMode type combo default cubeless var cubeless var cubeful",
            );
            reply(
                &mut stdout,
                "option name Variant type combo default backgammon var backgammon",
            );
            reply(&mut stdout, "ubgiok");
            continue;
        }

        if cmd == "isready" {
            reply(&mut stdout, "readyok");
            continue;
        }

        if cmd == "newgame" {
            context.position = None;
            context.dice = None;
            continue;
        }

        if let Some(err) = handle_setoption(cmd, &mut context) {
            reply(&mut stdout, &err);
            continue;
        }
        if cmd.starts_with("setoption ") {
            continue;
        }

        if let Some(id) = cmd.strip_prefix("position gnubgid ") {
            match parse_position_id(id.trim()) {
                Some(position) => context.position = Some(position),
                None => reply(&mut stdout, "error bad_argument invalid_position"),
            }
            continue;
        }

        if cmd == "position xgid" || cmd.starts_with("position xgid ") {
            reply(&mut stdout, "error unsupported_feature position_xgid");
            continue;
        }

        if let Some(rest) = cmd.strip_prefix("dice ") {
            match parse_dice(rest) {
                Some(dice) => context.dice = Some(dice),
                None => reply(&mut stdout, "error bad_argument dice"),
            }
            continue;
        }

        if cmd == "setturn p0" || cmd == "setturn p1" {
            continue;
        }

        if cmd.starts_with("newsession ") {
            continue;
        }

        if cmd.starts_with("setscore ") {
            continue;
        }

        if cmd.starts_with("setcube ") {
            continue;
        }

        if cmd.starts_with("go") {
            if let Some(err) = validate_go(cmd) {
                reply(&mut stdout, &err);
                continue;
            }

            let Some(position) = context.position else {
                reply(&mut stdout, "error missing_context position");
                continue;
            };
            let Some(dice) = context.dice else {
                reply(&mut stdout, "error missing_context dice");
                continue;
            };

            let best_position =
                evaluator.best_position(&position, &dice, context.score_config.value());
            let best_move = BgMove::new(&position, &best_position.sides_switched(), &dice);
            let best_move_text = format_move(best_move);

            reply(
                &mut stdout,
                &format!("info role chequer pv {}", best_move_text),
            );
            reply(&mut stdout, &format!("bestmove {best_move_text}"));
            continue;
        }

        if cmd == "stop" {
            continue;
        }

        if cmd == "quit" {
            break;
        }

        reply(&mut stdout, "error unknown_command");
    }
}

struct Context {
    position: Option<Position>,
    dice: Option<Dice>,
    score_config: ScoreConfig,
    deterministic: bool,
    threads: usize,
    seed: u64,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            position: None,
            dice: None,
            score_config: ScoreConfig::MoneyGame,
            deterministic: true,
            threads: 1,
            seed: 0,
        }
    }
}

fn handle_setoption(cmd: &str, context: &mut Context) -> Option<String> {
    let rest = cmd.strip_prefix("setoption name ")?;
    let Some((name, value)) = rest.split_once(" value ") else {
        return Some("error bad_argument setoption".to_string());
    };

    match name.trim() {
        "Threads" => match value.trim().parse::<usize>() {
            Ok(threads) if threads > 0 => {
                context.threads = threads;
                None
            }
            _ => Some("error bad_argument threads".to_string()),
        },
        "Seed" => match value.trim().parse::<u64>() {
            Ok(seed) => {
                context.seed = seed;
                None
            }
            _ => Some("error bad_argument seed".to_string()),
        },
        "Deterministic" => match parse_bool(value.trim()) {
            Some(deterministic) => {
                context.deterministic = deterministic;
                None
            }
            None => Some("error bad_argument deterministic".to_string()),
        },
        "EvalMode" => match value.trim() {
            "cubeless" => {
                context.score_config = ScoreConfig::MoneyGame;
                None
            }
            "cubeful" => Some("error unsupported_feature evalmode_cubeful".to_string()),
            _ => Some("error bad_argument evalmode".to_string()),
        },
        "Variant" => match value.trim() {
            "backgammon" => None,
            _ => Some("error unsupported_feature variant".to_string()),
        },
        _ => Some("error unsupported_feature setoption".to_string()),
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value {
        "true" | "on" | "1" => Some(true),
        "false" | "off" | "0" => Some(false),
        _ => None,
    }
}

fn parse_position_id(position_id: &str) -> Option<Position> {
    std::panic::catch_unwind(|| Position::from_id(position_id)).ok()
}

fn parse_dice(rest: &str) -> Option<Dice> {
    let parts: Vec<&str> = rest.split_whitespace().collect();
    if parts.len() != 2 {
        return None;
    }
    let d1 = parts[0].parse::<usize>().ok()?;
    let d2 = parts[1].parse::<usize>().ok()?;
    if !(1..=6).contains(&d1) || !(1..=6).contains(&d2) {
        return None;
    }
    Some(Dice::new(d1, d2))
}

fn validate_go(cmd: &str) -> Option<String> {
    if cmd == "go" || cmd == "go role chequer" || cmd.starts_with("go role chequer ") {
        return None;
    }
    if cmd.contains("role cube") {
        return Some("error unsupported_feature role_cube".to_string());
    }
    if cmd.contains("role turn") {
        return Some("error unsupported_feature role_turn".to_string());
    }
    if cmd.starts_with("go role ") {
        return Some("error bad_argument role".to_string());
    }
    Some("error bad_argument go".to_string())
}

fn format_move(bg_move: BgMove) -> String {
    let details = bg_move.into_details();
    if details.is_empty() {
        return "pass".to_string();
    }

    fn point_text(point: usize) -> &'static str {
        if point == 25 {
            "bar"
        } else if point == 0 {
            "off"
        } else {
            ""
        }
    }

    details
        .iter()
        .map(|d| {
            let from = d.from();
            let to = d.to();
            let from_text = point_text(from);
            let to_text = point_text(to);

            let from_part = if from_text.is_empty() {
                from.to_string()
            } else {
                from_text.to_string()
            };
            let to_part = if to_text.is_empty() {
                to.to_string()
            } else {
                to_text.to_string()
            };

            format!("{from_part}/{to_part}")
        })
        .collect::<Vec<String>>()
        .join(" ")
}

fn reply(out: &mut impl Write, line: &str) {
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}
