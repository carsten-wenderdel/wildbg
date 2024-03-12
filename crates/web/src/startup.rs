use clap::Parser;

/// Command line arguments for starting the web application.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// The web address to host the server at with a default value of "127.0.0.1" when no input is provided.
    #[arg(short, long, default_value_t = String::from("127.0.0.1"))]
    pub address: String,

    /// The port to host the server at with a default value of "8080" when no input is provided.
    #[arg(short, long, default_value_t = String::from("8080"))]
    pub port: String,
}

/// Parse the command line arguments and generate a web address used for starting the application
/// and generating links.
///
/// # Arguments
///
/// * `cli_args` - A reference to the arguments passed on the command line.
///
/// # Examples
///
/// ```
/// use web::startup::{self, Args};
///
/// let args = Args {
///     address: String::from("127.0.0.1"),
///     port: String::from("8080")
/// };
///
/// let web_address = startup::get_web_address(&args);
///
/// assert_eq!(&web_address, "localhost:8080");
///
/// ```
pub fn get_web_address(cli_args: &Args) -> String {
    match cli_args.address.to_ascii_lowercase().as_str() {
        "localhost" | "127.0.0.1" | "0.0.0.0" => format!("localhost:{}", cli_args.port),
        _ => format!("{}:{}", cli_args.address, cli_args.port),
    }
}
