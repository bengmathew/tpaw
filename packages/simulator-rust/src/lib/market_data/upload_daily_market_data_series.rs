use futures::future::join_all;
use google_cloud_storage::http::objects::{
    delete::DeleteObjectRequest,
    list::ListObjectsRequest,
    upload::{Media, UploadObjectRequest, UploadType},
};
use serde::Serialize;

use crate::{clients::get_gcs_client, config::CONFIG, utils::ny_tz::get_now_in_ny_tz};

use super::market_data_defs::{
    DailyMarketDataForPresets, VTAndBNDData, MARKET_DATA_FOR_PRESETS_DIRECTORY,
    MARKET_DATA_VT_AND_BND_DIRECTORY,
};

pub async fn upload_daily_market_data_series(
    for_presets_series: &Vec<DailyMarketDataForPresets>,
    vt_and_bnd_series: &Vec<VTAndBNDData>,
) {
    upload(&for_presets_series, MARKET_DATA_FOR_PRESETS_DIRECTORY).await;
    upload(&vt_and_bnd_series, MARKET_DATA_VT_AND_BND_DIRECTORY).await;
}

async fn upload<Series: Serialize>(series: &Vec<Series>, directory: &str) {
    let gcs_client = get_gcs_client().await;

    let list_result = gcs_client
        .list_objects(&ListObjectsRequest {
            bucket: CONFIG.market_data_bucket.to_owned(),
            prefix: Some(format!("{}/latest/", directory).to_owned()),
            ..Default::default()
        })
        .await
        .unwrap()
        .items;

    if let Some(list_result) = list_result {
        assert_eq!(list_result.len(), 1);
        let market_data_file_info = list_result.first().unwrap();

        gcs_client
            .delete_object(&DeleteObjectRequest {
                bucket: CONFIG.market_data_bucket.to_owned(),
                object: market_data_file_info.name.to_owned(),
                ..Default::default()
            })
            .await
            .unwrap();
    };

    let upload_req = UploadObjectRequest {
        bucket: CONFIG.market_data_bucket.to_owned(),
        ..Default::default()
    };
    let upload_types = {
        let base = get_now_in_ny_tz()
            .format("%Y-%m-%d-%H-%M-%S-%Z")
            .to_string();
        vec![
            format!("{}/{}.json", directory, base),
            format!("{}/latest/{}.json", directory, base),
        ]
        .into_iter()
        .map(|filename| UploadType::Simple(Media::new(filename)))
        .collect::<Vec<UploadType>>()
    };

    join_all(upload_types.iter().map(|upload_type| {
        gcs_client.upload_object(
            &upload_req,
            serde_json::to_string(&series).unwrap().into_bytes(),
            upload_type,
        )
    }))
    .await;
}
