use crate::config::CONFIG;
use crate::estimate_portfolio_balance::estimate_portfolio_balance;
use crate::estimate_portfolio_balance::portfolio_balance_estimation_args::PortfolioBalanceEstimationArgs;
use crate::market_data::downloaded_market_data::{
    dowload_data, get_daily_market_data_for_portfolio_balance_estimation_synthetic_override,
    get_market_data_series_for_portfolio_balance_estimation, get_market_data_series_for_simulation,
    get_market_data_series_for_simulation_synthetic_override, DailyMarketSeriesSrc, DownloadedData,
};
use crate::market_data::get_daily_market_data_series_from_source::get_daily_market_data_series_from_source;
use crate::market_data::upload_daily_market_data_series::upload_daily_market_data_series;
use crate::simulate::simulate;
use crate::wire::{wire_simulation_args, WireSimulationArgs};
use axum::body::Body;
use axum::extract::Request;
use axum::response::{AppendHeaders, IntoResponse, Response};
use axum::{body::Bytes, extract::Path, routing, Router};
use http::{header, HeaderValue, StatusCode};
use prost::Message;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::decompression::RequestDecompressionLayer;
use tower_http::trace::TraceLayer;

const VERSION: u64 = 2;

pub async fn serve() {
    // Pre-load market data.

    let downloaded_data = Arc::new(RwLock::new(dowload_data().await));
    let downloaded_data1 = downloaded_data.clone();
    let downloaded_data2 = downloaded_data.clone();

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    // Note, this is in reverse order of precedence:  https://github.com/tokio-rs/axum/issues/259
    // Note, routes are sensitive to the trailing "/".
    let app = Router::new()
        .route(
            "/",
            routing::get(|| async { format!("I am v{} root!", VERSION) }),
        )
        .route(
            "/update_daily_market_data_series",
            routing::post(move |request: Request<Body>| {
                handle_update_daily_market_data_series(downloaded_data2, request)
            }),
        )
        .route(
            "/:version/simulate",
            routing::post(
                move |Path(version_str): Path<String>, request_body: Bytes| {
                    handle_simulate(downloaded_data1, version_str, request_body)
                },
            ),
        )
        .layer(
            CorsLayer::new()
                .allow_methods(tower_http::cors::Any)
                // Authorization needs to be added in addition to * because
                // https://stackoverflow.com/a/68649111/2771609
                .allow_headers([header::AUTHORIZATION, "*".parse().unwrap()])
                .allow_origin(CONFIG.cors_allow_origin.parse::<HeaderValue>().unwrap())
                .max_age(Duration::from_secs(60 * 60 * 24 * 365)),
        )
        .layer(CompressionLayer::new())
        .layer(RequestDecompressionLayer::new())
        .layer(TraceLayer::new_for_http())
        .layer(sentry_tower::NewSentryLayer::<Request>::new_from_top())
        .layer(sentry_tower::SentryHttpLayer::with_transaction());

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", CONFIG.port))
        .await
        .unwrap();
    println!("Server starting on port {}", CONFIG.port);
    axum::serve(listener, app).await.unwrap();
}

async fn handle_update_daily_market_data_series(
    downloaded_data: Arc<RwLock<DownloadedData>>,
    request: Request<Body>,
) -> Response {
    let token = request.headers().get("x-server-to-server-token");
    if token.is_none() || token.unwrap().to_str().unwrap() != CONFIG.server_to_server_token {
        return (StatusCode::UNAUTHORIZED, "Invalid token").into_response();
    }
    let (for_presets_series, vt_and_bnd_series) = get_daily_market_data_series_from_source().await;
    upload_daily_market_data_series(&for_presets_series, &vt_and_bnd_series).await;
    let mut downloaded_data_ref = downloaded_data.write().await;
    *downloaded_data_ref = DownloadedData {
        daily_market_data_for_presets_series: for_presets_series,
        vt_and_bnd_series: vt_and_bnd_series,
    };
    (StatusCode::OK, "").into_response()
}

async fn handle_simulate(
    downloaded_data: Arc<RwLock<DownloadedData>>,
    version_str: String,
    request_body: Bytes,
) -> Response {
    let version = version_str.parse::<u64>();
    if let Ok(version) = version {
        assert!(version <= VERSION);
        if version == VERSION {
            return handle_simulate_current(downloaded_data, request_body).await;
        } else if version < VERSION {
            let headers = AppendHeaders([("x-app-error-code", "clientNeedsUpdate")]);
            return (StatusCode::BAD_REQUEST, headers, Body::empty()).into_response();
        } else {
            return (StatusCode::BAD_REQUEST, "Invalid version").into_response();
        }
    } else {
        return (StatusCode::BAD_REQUEST, "Invalid version").into_response();
    }
}

// Note: If handler panics, the server will respond with a 500 error but will not crash. It will
// continue to accept requests.
async fn handle_simulate_current(
    downloaded_data: Arc<RwLock<DownloadedData>>,
    request_body: Bytes,
) -> Response {
    let start = Instant::now();
    let downloaded_data_ref = downloaded_data.read().await;
    println!("downloaded_data_ref read took {:?}", start.elapsed());

    let simulation_args = WireSimulationArgs::decode(&request_body.to_vec()[..]).unwrap();

    let market_daily_series_src: DailyMarketSeriesSrc =
        simulation_args.market_daily_series_src.unwrap().into();
    let market_data_series_for_simulation_synthetic_override =
        get_market_data_series_for_simulation_synthetic_override(
            &market_daily_series_src,
            simulation_args.timestamp_for_market_data_ms,
            &downloaded_data_ref.daily_market_data_for_presets_series,
        )
        .await;
    let downloaded_data_ref = downloaded_data.read().await;
    let market_data_series_for_simulation = get_market_data_series_for_simulation(
        market_data_series_for_simulation_synthetic_override.as_ref(),
        &downloaded_data_ref,
    )
    .await;

    let market_data_series_for_portfolio_balance_estimation_synthetic_override =
        get_daily_market_data_for_portfolio_balance_estimation_synthetic_override(
            &market_daily_series_src,
            simulation_args.timestamp_for_market_data_ms,
            &downloaded_data_ref.vt_and_bnd_series,
        )
        .await;
    let market_data_series_for_portfolio_balance_estimation =
        get_market_data_series_for_portfolio_balance_estimation(
            market_data_series_for_portfolio_balance_estimation_synthetic_override.as_ref(),
            &downloaded_data_ref,
        )
        .await;

    let portfolio_balance_estimation_start = Instant::now();
    let (portfolio_balance, portfolio_balance_estimation_result) =
        match simulation_args.current_portfolio_balance.unwrap() {
            wire_simulation_args::CurrentPortfolioBalance::Estimate(estimate) => {
                let args: PortfolioBalanceEstimationArgs = estimate.into();
                let portfolio_balance_estimation_result = estimate_portfolio_balance(
                    args.start_balance as f64,
                    args.plan_params_for_non_market_actions_unsorted.into(),
                    simulation_args.plan_params.evaluation_timestamp_ms,
                    &market_data_series_for_portfolio_balance_estimation,
                    &market_data_series_for_simulation,
                );
                let estimate = portfolio_balance_estimation_result
                    .actions
                    .last()
                    .map(|x| x.state_change.end.estimate)
                    .unwrap_or(portfolio_balance_estimation_result.start_state.estimate);
                (estimate, Some(portfolio_balance_estimation_result))
            }
            wire_simulation_args::CurrentPortfolioBalance::NoEstimate(no_estimate) => {
                (no_estimate, None)
            }
        };
    let portfolio_balance_estimation_in_ms =
        portfolio_balance_estimation_start.elapsed().as_millis() as i64;

    let simulation_start = Instant::now();
    let (result, plan_params_processed) = simulate(
        portfolio_balance,
        &simulation_args.percentiles,
        simulation_args
            .plan_params
            .ages
            .simulation_months
            .num_months,
        &simulation_args.plan_params.into(),
        simulation_args.timestamp_for_market_data_ms,
        &market_data_series_for_simulation,
    );
    let simulation_in_ms = simulation_start.elapsed().as_millis() as i64;
    let total_in_ms = portfolio_balance_estimation_in_ms + simulation_in_ms;
    let wire_result = result.into_wire(
        portfolio_balance_estimation_result.map(|x| x.into()),
        plan_params_processed,
        portfolio_balance_estimation_in_ms,
        simulation_in_ms,
        total_in_ms,
    );

    let response_body = Body::from(wire_result.encode_to_vec());

    let headers = AppendHeaders([(header::CONTENT_TYPE, "application/octet-stream")]);
    (headers, response_body).into_response()
}
