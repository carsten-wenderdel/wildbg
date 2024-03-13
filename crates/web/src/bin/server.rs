use clap::Parser;
use std::sync::Arc;
use tokio::net::TcpListener;
use web::axum::router;
use web::startup::{self, Args};
use web::web_api::WebApi;

#[tokio::main]
async fn main() {
    let web_address = startup::get_web_address(&Args::parse());

    log_server_links(&web_address);

    let listener = TcpListener::bind(&web_address)
        .await
        .unwrap_or_else(|_| panic!("Could not bind to the web address: '{web_address}'"));

    let web_api = Arc::new(WebApi::try_default());
    let app = router(web_api);
    axum::serve(listener, app).await.unwrap();
}

/// Log the web address and helpful links to the command line.
///
/// # Arguments
///
/// * `web_address` - The address and port number to use for generating links and logging.
fn log_server_links(web_address: &str) {
    println!("The server is running at 'http://{web_address}'.\n");

    println!("You can access the server for example via");
    println!(
        "http://{}/move?die1=3&die2=1&p24=2&p19=-5&p17=-3&p13=5&p12=-5&p8=3&p6=5&p1=-2",
        web_address
    );
    println!("http://{}/swagger-ui", web_address);
}
