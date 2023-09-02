use axum::http::StatusCode;
use axum::{routing::get, Json, Router, Server};
use hyper::Error;
use serde::Serialize;
use std::net::{Ipv4Addr, SocketAddr};
use wildbg::bg_move::MoveDetail;
use wildbg::web_api::WebApi;

#[tokio::main]
async fn main() -> Result<(), Error> {
    println!("You can access the server via:");
    println!("http://localhost:8080/move");

    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8080));
    Server::bind(&address)
        .serve(router().into_make_service())
        .await
}

fn router() -> Router {
    Router::new()
        .route("/cube", get(get_cube))
        .route("/move", get(get_move))
}

#[derive(Serialize)]
struct ErrorMessage {
    message: String,
}

impl ErrorMessage {
    fn json(message: &str) -> Json<ErrorMessage> {
        Json(ErrorMessage {
            message: message.to_string(),
        })
    }
}

async fn get_cube() -> Result<String, (StatusCode, Json<ErrorMessage>)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        ErrorMessage::json("Will be implemented later on."),
    ))
}

async fn get_move() -> Result<Json<Vec<MoveDetail>>, (StatusCode, Json<ErrorMessage>)> {
    let web_api = WebApi::try_default().unwrap();
    let best_move = web_api.get_move();
    Ok(Json(best_move))
}

#[cfg(test)]
mod tests {
    use crate::router;
    use axum::http::header::CONTENT_TYPE;
    use hyper::{Body, Request, StatusCode};
    use tower::ServiceExt; // for `oneshot

    /// Consumes the response, so use it at the end of the test
    async fn body_string(response: axum::response::Response) -> String {
        let body_bytes = hyper::body::to_bytes(response.into_body()).await.unwrap();
        std::str::from_utf8(&body_bytes).unwrap().to_string()
    }

    #[tokio::test]
    async fn get_cube_error() {
        let response = router()
            .oneshot(Request::builder().uri("/cube").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");

        let body = body_string(response).await;
        assert_eq!(body, "{\"message\":\"Will be implemented later on.\"}");
    }

    #[tokio::test]
    async fn get_move() {
        let response = router()
            .oneshot(Request::builder().uri("/move").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");

        let body = body_string(response).await;
        assert_eq!(body, r#"[{"from":8,"to":5},{"from":6,"to":5}]"#);
    }
}
