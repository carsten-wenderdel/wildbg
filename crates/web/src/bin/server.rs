use std::sync::Arc;
use tokio::net::TcpListener;
use web::axum::router;
use web::web_api::WebApi;

#[tokio::main]
async fn main() {
    println!("You can access the server for example via");
    println!(
        "http://localhost:8080/move?die1=3&die2=1&p24=2&p19=-5&p17=-3&p13=5&p12=-5&p8=3&p6=5&p1=-2"
    );
    println!("http://localhost:8080/swagger-ui");

    let listener = TcpListener::bind("0.0.0.0:8080").await.unwrap();
    let web_api = Arc::new(WebApi::try_default());
    let app = router(web_api);
    axum::serve(listener, app).await.unwrap();
}
