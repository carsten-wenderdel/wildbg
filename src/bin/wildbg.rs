use axum::{routing::get, Router, Server};
use hyper::Error;
use std::net::{Ipv4Addr, SocketAddr};

#[tokio::main]
async fn main() -> Result<(), Error> {
    println!("You can access the server via:");
    println!("http://localhost:8080/move");

    let app = Router::new().route("/move", get(axum_bg::get_move));
    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8080));
    Server::bind(&address).serve(app.into_make_service()).await
}

mod axum_bg {
    use axum::http::StatusCode;

    pub async fn get_move() -> Result<String, (StatusCode, String)> {
        // Ok("Hello, World".to_string())
        Err((
            StatusCode::NOT_IMPLEMENTED,
            "Will be implemented later on.".to_string(),
        ))
    }
}
