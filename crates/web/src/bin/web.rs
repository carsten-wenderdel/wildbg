use std::sync::Arc;
use web::axum::router;
use web::web_api::WebApi;

// This binary is for deploying wildbg at shuttle.dev
// Currently it's available at:
// https://wildbg-pwuo.shuttle.app/swagger-ui/
// https://wildbg-pwuo.shuttle.app/move?die1=5&die2=2&p24=2&p19=-5&p17=-3&p13=5&p12=-5&p8=3&p6=5&p1=-2

// Login at https://console.shuttle.dev
// Login in console: shuttle login
// Locally: shuttle run
// Deploying: shuttle deploy

// If you don't want to use shuttle, but run the web server directly, use `server.rs`.
// Both files use the same library functions.
// A `cargo run` will also start server.rs

#[shuttle_runtime::main]
async fn main() -> shuttle_axum::ShuttleAxum {
    let web_api = Arc::new(WebApi::try_default());
    Ok(router(web_api).into())
}
