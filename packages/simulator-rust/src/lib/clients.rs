use google_cloud_storage::client::google_cloud_auth::credentials::CredentialsFile;

use crate::config::CONFIG;

pub async fn get_gcs_client() -> google_cloud_storage::client::Client {
    let credentials = CredentialsFile::new_from_file(CONFIG.gcp_key_path.to_owned())
        .await
        .unwrap();
    let config = google_cloud_storage::client::ClientConfig::default()
        .with_credentials(credentials)
        .await
        .unwrap();
    google_cloud_storage::client::Client::new(config)
}
