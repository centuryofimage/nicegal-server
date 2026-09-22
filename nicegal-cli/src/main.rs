use std::env;
use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow};
use camino::Utf8PathBuf as PathBuf;
use clap::builder::{PossibleValuesParser, TypedValueParser};
use clap::{ArgMatches, Command, arg, crate_description, crate_version, value_parser};

use nicegal_core::db::{DB, SearchFilters, SearchType};

#[cfg(not(target_env = "msvc"))]
#[cfg(not(debug_assertions))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[cfg(not(debug_assertions))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() -> Result<()> {
    run(cli().get_matches())
}

fn run(matches: ArgMatches) -> Result<()> {
    if let Some(pwd) = matches.get_one::<String>("pwd") {
        env::set_current_dir(pwd).with_context(|| format!("changing directory to: {pwd}"))?;
    }

    let database: &PathBuf = matches.get_one("database").unwrap();
    let mut db = DB::new(database)?;
    let queries = matches
        .get_many::<String>("QUERIES")
        .ok_or_else(|| anyhow!("No queries were provided"))?;
    let cwd = PathBuf::from_path_buf(env::current_dir()?)
        .map_err(|path| anyhow!("current directory is not valid UTF-8: {}", path.display()))?;
    let filters = SearchFilters::new(&cwd)
        .with_exclude(matches.get_one::<String>("exclude").map(String::as_str));
    let results = db.search(
        queries.map(String::as_str).collect(),
        &filters,
        *matches.get_one::<usize>("limit").unwrap(),
        *matches.get_one::<SearchType>("search-type").unwrap(),
    )?;

    for result in results {
        println!("{}\t{}", result.contents.escape_debug(), result.path);
    }
    Ok(())
}

fn cli() -> Command {
    static DBPATH: LazyLock<PathBuf> = LazyLock::new(|| {
        PathBuf::try_from(
            dirs::data_local_dir().expect("the user's local data directory should exist"),
        )
        .expect("the user's local data directory should be valid UTF-8")
        .join("nicegal-server")
        .join("index.db")
    });

    Command::new("nicegal-cli")
        .version(crate_version!())
        .about(crate_description!())
        .args([
            arg!(-d --database <FILE> "Location of the OCR index database")
                .value_parser(value_parser!(PathBuf))
                .env("NICEGAL_DB")
                .default_value(DBPATH.as_os_str()),
            arg!(-x --exclude <PATTERN> "Exclude indexed paths matching this glob"),
            arg!(-l --limit <LIMIT> "Maximum number of results")
                .value_parser(value_parser!(usize))
                .default_value("100"),
            arg!(-s --"search-type" <TYPE> "Search query type")
                .default_value("simple")
                .value_parser(
                    PossibleValuesParser::new(["simple", "match", "glob", "regex"]).map(
                        |value| -> SearchType {
                            match value.to_ascii_lowercase().as_str() {
                                "simple" => SearchType::Simple,
                                "match" => SearchType::Match,
                                "glob" => SearchType::Glob,
                                #[cfg(feature = "regex")]
                                "regex" => SearchType::Regex,
                                #[cfg(not(feature = "regex"))]
                                "regex" => panic!("this build was not compiled with regex support"),
                                _ => unreachable!(),
                            }
                        },
                    ),
                ),
            arg!(--pwd <PWD> "Set working directory").hide(true),
            arg!(<QUERIES> ... "Strings to search for").required(true),
        ])
}

#[cfg(test)]
mod tests {
    use super::{cli, run};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn searches_an_existing_database_without_other_stores() {
        let temp = unique_test_directory();
        let database = temp.join("index.db");
        let database = database.to_str().unwrap();
        let result = run(cli()
            .try_get_matches_from(["nicegal-cli", "--database", database, "absent"])
            .unwrap());
        std::fs::remove_dir_all(&temp).unwrap();
        result.unwrap();
    }

    fn unique_test_directory() -> std::path::PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "nicegal-cli-test-{}-{timestamp}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }
}
