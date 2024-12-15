use crate::{
    market_data::{
        market_data_defs::{DailyMarketDataForPresets, VTAndBNDData},
        get_daily_market_data_series_from_source::get_daily_market_data_series_from_source,
        upload_daily_market_data_series::upload_daily_market_data_series,
    },
    utils::ny_tz::interpret_timestamp_as_ny_tz,
};

pub async fn cli_update_daily_market_data_series(dry_run: bool) {
    let (for_presets_series, vt_and_bnd_series) = get_daily_market_data_series_from_source().await;
    print(&for_presets_series, &vt_and_bnd_series);
    if !dry_run {
        upload_daily_market_data_series(&for_presets_series, &vt_and_bnd_series).await;
    }
}

fn print(for_presets_series: &Vec<DailyMarketDataForPresets>, vt_and_bnd_series: &Vec<VTAndBNDData>) {
    let format_date = |closing_time_ms: i64| {
        interpret_timestamp_as_ny_tz(closing_time_ms)
            .format("%Y-%m-%d %H:%M:%S %Z")
            .to_string()
    };
    {
        let last = for_presets_series.last().unwrap();
        println!("-----------------------");
        println!("Daily Market Data for Presets");
        println!("-----------------------");
        println!("{: >20}: {}", "Overall", format_date(last.closing_time_ms));
        println!(
            "{: >20}: {}",
            "Inflation",
            format_date(last.inflation.closing_time_ms)
        );
        println!(
            "{: >20}: {}",
            "SP500",
            format_date(last.sp500.closing_time_ms)
        );
        println!(
            "{: >20}: {}",
            "Bond Rates",
            format_date(last.bond_rates.closing_time_ms)
        );
    }
    {
        let last = vt_and_bnd_series.last().unwrap();
        println!("-----------------------");
        println!("VT and BND");
        println!("-----------------------");
        println!("{: >20}: {}", "VT and BND", format_date(last.closing_time_ms));
    }
}
