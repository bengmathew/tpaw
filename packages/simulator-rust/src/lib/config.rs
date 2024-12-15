use lazy_static::lazy_static;

pub struct Config {
    pub port: u64,
    pub market_data_bucket: String,
    pub fred_api_key: String,
    pub eod_api_key: String,
    pub sentry_dsn: String,
    pub gcp_key_path: String,
    pub server_to_server_token: String,
}

lazy_static! {
    pub static ref CONFIG: Config = Config {
        port: std::env::var("PORT").unwrap().parse().unwrap(),
        market_data_bucket: std::env::var("MARKET_DATA_BUCKET").unwrap(),
        fred_api_key: std::env::var("FRED_API_KEY").unwrap(),
        eod_api_key: std::env::var("EOD_API_KEY").unwrap(),
        sentry_dsn: std::env::var("SENTRY_DSN").unwrap(),
        gcp_key_path: std::env::var("GCP_KEY_PATH").unwrap(),
        server_to_server_token: std::env::var("SERVER_TO_SERVER_TOKEN").unwrap(),
    };
}
