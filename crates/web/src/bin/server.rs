use axum::Server;
use hyper::Error;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use web::axum::router;
use web::web_api::WebApi;

#[tokio::main]
async fn main() -> Result<(), Error> {
    println!("You can access the server for example via");
    println!(
        "http://localhost:8080/move?die1=3&die2=1&p24=2&p19=-5&p17=-3&p13=5&p12=-5&p8=3&p6=5&p1=-2"
    );
    println!("http://localhost:8080/swagger-ui");

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let web_api = Arc::new(WebApi::try_default());
    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8080));
    Server::bind(&address)
        .serve(router(web_api).into_make_service())
        .await
}
