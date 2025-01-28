#![allow(non_camel_case_types)]

use clap::{Parser, Subcommand, ValueEnum};
use dotenv::dotenv;
use simulator::{
    cli::{
        cli_process_historical_data_part_1_parse_csv::cli_process_historical_data_part_1_parse_csv,
        cli_process_historical_data_part_2_derive::cli_process_historical_data_part_2_derive,
        cli_scratch::cli_scratch, cli_serve::cli_serve,
        cli_update_daily_market_data_series::cli_update_daily_market_data_series,
    },
    config::CONFIG,
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
    V3,
    V4,
    V5,
}

#[derive(Subcommand)]
enum Commands {
    ProcessHistoricalDataPart1 {
        returns_id: Version,
    },
    ProcessHistoricalDataPart2 {
        returns_id: Version,
    },

    UpdateDailyMarketDataSeries {
        #[arg(long)]
        dry_run: bool,
    },
    Scratch,
    Serve,
}

fn return_id_map(
    returns_id: Version,
) -> simulator::historical_monthly_returns::HistoricalReturnsId {
    match returns_id {
        Version::V1 => simulator::historical_monthly_returns::HistoricalReturnsId::V1,
        Version::V2 => simulator::historical_monthly_returns::HistoricalReturnsId::V2,
        Version::V3 => simulator::historical_monthly_returns::HistoricalReturnsId::V3,
        Version::V4 => simulator::historical_monthly_returns::HistoricalReturnsId::V4,
        Version::V5 => simulator::historical_monthly_returns::HistoricalReturnsId::V5,
    }
}

fn main() {
    dotenv().ok();
    let _guard = sentry::init((
        CONFIG.sentry_dsn.as_str(),
        sentry::ClientOptions {
            release: sentry::release_name!(),
            ..Default::default()
        },
    ));

    // Sentry does not like #[tokio::main]: https://docs.sentry.io/platforms/rust/#async-main-function
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {

            let cli = Cli::parse();
            match cli.command {
                Commands::ProcessHistoricalDataPart1 { returns_id } => {
                    cli_process_historical_data_part_1_parse_csv(return_id_map(returns_id));
                }
                Commands::ProcessHistoricalDataPart2 { returns_id } => {
                    cli_process_historical_data_part_2_derive(return_id_map(returns_id));
                }
                 
                Commands::UpdateDailyMarketDataSeries { dry_run } => {
                    cli_update_daily_market_data_series(dry_run).await;
                }
                Commands::Serve => {
                    cli_serve().await;
                }
                Commands::Scratch => {
                    cli_scratch().await;
                }
            }
        });
}
