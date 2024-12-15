use google_cloud_storage::http::objects::{
    download::Range, get::GetObjectRequest, list::ListObjectsRequest,
};
use serde::Deserialize;

use crate::{
    clients::get_gcs_client,
    config::CONFIG,
    market_data::market_data_defs::{
        VTAndBNDData, MARKET_DATA_FOR_PRESETS_DIRECTORY, MARKET_DATA_VT_AND_BND_DIRECTORY,
    },
};

use super::market_data_defs::DailyMarketDataForPresets;

pub async fn download_daily_market_data_series() -> (Vec<DailyMarketDataForPresets>, Vec<VTAndBNDData>) {
    let (for_presets_series, vt_and_bnd_series) = futures::join!(
        download::<DailyMarketDataForPresets>(MARKET_DATA_FOR_PRESETS_DIRECTORY),
        download::<VTAndBNDData>(MARKET_DATA_VT_AND_BND_DIRECTORY)
    );
    (for_presets_series, vt_and_bnd_series)
}

async fn download<T>(directory: &str) -> Vec<T>
where
    T: for<'de> Deserialize<'de>,
{
    let gcs_client = get_gcs_client().await;

    let list_result = gcs_client
        .list_objects(&ListObjectsRequest {
            bucket: CONFIG.market_data_bucket.to_owned(),
            prefix: Some(format!("{}/latest/", directory).to_owned()),
            ..Default::default()
        })
        .await
        .unwrap()
        .items
        .unwrap();

    assert_eq!(list_result.len(), 1);
    let latest_series_file_info = list_result.first().unwrap();

    let series_bytes = gcs_client
        .download_object(
            &GetObjectRequest {
                bucket: CONFIG.market_data_bucket.to_owned(),
                object: latest_series_file_info.name.to_owned(),
                ..Default::default()
            },
            &Range::default(),
        )
        .await
        .unwrap();

    let series_str = String::from_utf8(series_bytes).unwrap();
    serde_json::from_str(&series_str).unwrap()
}
