use axum::extract::State;
use axum::http::StatusCode;
use axum::{routing::get, Json, Router, Server};
use hyper::Error;
use serde::Serialize;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use wildbg::bg_move::MoveDetail;
use wildbg::web_api::WebApi;

#[tokio::main]
async fn main() -> Result<(), Error> {
    println!("You can access the server via:");
    println!("http://localhost:8080/move");

    let web_api = Arc::new(WebApi::try_default()) as DynWebApi;
    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8080));
    Server::bind(&address)
        .serve(router(web_api).into_make_service())
        .await
}

fn router(web_api: DynWebApi) -> Router {
    Router::new()
        .route("/cube", get(get_cube))
        .route("/move", get(get_move))
        .with_state(web_api)
}

type DynWebApi = Arc<Option<WebApi>>;

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

async fn get_move(
    State(web_api): State<DynWebApi>,
) -> Result<Json<Vec<MoveDetail>>, (StatusCode, Json<ErrorMessage>)> {
    match web_api.as_ref() {
        None => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorMessage::json("Neural net could not be constructed."),
        )),
        Some(web_api) => Ok(Json(web_api.get_move())),
    }
}

#[cfg(test)]
mod tests {
    use crate::{router, DynWebApi};
    use axum::http::header::CONTENT_TYPE;
    use hyper::{Body, Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt; // for `oneshot
    use wildbg::web_api::WebApi;

    /// Consumes the response, so use it at the end of the test
    async fn body_string(response: axum::response::Response) -> String {
        let body_bytes = hyper::body::to_bytes(response.into_body()).await.unwrap();
        std::str::from_utf8(&body_bytes).unwrap().to_string()
    }

    #[tokio::test]
    async fn get_cube_error() {
        let web_api = Arc::new(None) as DynWebApi;
        let response = router(web_api)
            .oneshot(Request::builder().uri("/cube").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");

        let body = body_string(response).await;
        assert_eq!(body, "{\"message\":\"Will be implemented later on.\"}");
    }

    #[tokio::test]
    async fn get_move_missing_neural_net() {
        let web_api = Arc::new(None) as DynWebApi;
        let response = router(web_api)
            .oneshot(Request::builder().uri("/move").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");

        let body = body_string(response).await;
        assert_eq!(
            body,
            "{\"message\":\"Neural net could not be constructed.\"}"
        );
    }

    #[tokio::test]
    async fn get_move() {
        let web_api = Arc::new(WebApi::try_default()) as DynWebApi;
        let response = router(web_api)
            .oneshot(Request::builder().uri("/move").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");

        let body = body_string(response).await;
        assert_eq!(body, r#"[{"from":8,"to":5},{"from":6,"to":5}]"#);
    }
}
