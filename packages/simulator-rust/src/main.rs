mod cli;

use clap::{Parser, Subcommand, ValueEnum};
use cli::{
    cli_process_historical_data_part_2_derive::cli_process_historical_data_part_2_derive,
    cli_process_historical_data_part_1_parse_csv::cli_process_historical_data_part_1_parse_csv, cli_scratch::cli_scratch,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, ValueEnum)]
enum Version {
    V1,
    V2,
}

#[derive(Subcommand)]
enum Commands {
    ProcessHistoricalDataPart1 { returns_id: Version },
    ProcessHistoricalDataPart2 { returns_id: Version },
    Scratch,
}

fn return_id_map(
    returns_id: Version,
) -> simulator::historical_monthly_returns::HistoricalReturnsId {
    match returns_id {
        Version::V1 => simulator::historical_monthly_returns::HistoricalReturnsId::V1,
        Version::V2 => simulator::historical_monthly_returns::HistoricalReturnsId::V2,
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::ProcessHistoricalDataPart1 { returns_id } => {
            cli_process_historical_data_part_1_parse_csv(return_id_map(returns_id));
        }
        Commands::ProcessHistoricalDataPart2 {
            returns_id,
        } => {
            cli_process_historical_data_part_2_derive(return_id_map(returns_id));
        }
        Commands::Scratch => {
            cli_scratch();
        }
    }
}
