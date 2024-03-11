use clap::Parser;
use std::sync::Arc;
use tokio::net::TcpListener;
use web::axum::router;
use web::web_api::WebApi;

/// Command line arguments for starting the web application.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The web address to host the server at with a default value of "0.0.0.0" when no input is provided.
    #[arg(short, long, default_value_t = String::from("0.0.0.0"))]
    address: String,

    /// The port to host the server at with a default value of "8080" when no input is provided.
    #[arg(short, long, default_value_t = String::from("8080"))]
    port: String,
}

#[tokio::main]
async fn main() {
    let cli_args: Args = Args::parse();
    let web_address = format!("{}:{}", cli_args.address, cli_args.port);

    println!("You can access the server for example via");
    println!(
        "http://{}/move?die1=3&die2=1&p24=2&p19=-5&p17=-3&p13=5&p12=-5&p8=3&p6=5&p1=-2",
        &web_address
    );
    println!("http://{}/swagger-ui", &web_address);

    let listener = TcpListener::bind(web_address).await.unwrap();
    let web_api = Arc::new(WebApi::try_default());
    let app = router(web_api);
    axum::serve(listener, app).await.unwrap();
}
