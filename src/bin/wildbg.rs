use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::{routing::get, Json, Router, Server};
use hyper::Error;
use serde::Serialize;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use wildbg::web_api::{DiceParams, MoveResponse, PipParams, WebApi};

#[tokio::main]
async fn main() -> Result<(), Error> {
    println!("You can access the server for example via");
    println!(
        "http://localhost:8080/move?die1=3&die2=1&p24=2&p19=-5&p17=-3&p13=5&p12=-5&p8=3&p6=5&p1=-2"
    );

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
    Query(dice): Query<DiceParams>,
    Query(pips): Query<PipParams>,
    State(web_api): State<DynWebApi>,
) -> Result<Json<MoveResponse>, (StatusCode, Json<ErrorMessage>)> {
    match web_api.as_ref() {
        None => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorMessage::json("Neural net could not be constructed."),
        )),
        Some(web_api) => match web_api.get_move(pips, dice) {
            Err((status_code, message)) => Err((status_code, ErrorMessage::json(message.as_str()))),
            Ok(move_response) => Ok(Json(move_response)),
        },
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
            .oneshot(
                Request::builder()
                    .uri("/move?die1=3&die2=1&p24=2&p19=-5")
                    .body(Body::empty())
                    .unwrap(),
            )
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
    async fn get_move_no_arguments() {
        let web_api = Arc::new(WebApi::try_default()) as DynWebApi;
        let response = router(web_api)
            .oneshot(Request::builder().uri("/move").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers()[CONTENT_TYPE],
            "text/plain; charset=utf-8"
        );

        let body = body_string(response).await;
        // TODO: JSON would be a better response than plain text
        assert_eq!(
            body,
            "Failed to deserialize query string: missing field `die1`"
        );
    }

    #[tokio::test]
    async fn get_move_illegal_dice() {
        let web_api = Arc::new(WebApi::try_default()) as DynWebApi;
        let response = router(web_api)
            .oneshot(
                Request::builder()
                    .uri("/move?die1=2&die2=0&p4=4&p5=-5")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");

        let body = body_string(response).await;
        assert_eq!(
            body,
            r#"{"message":"Dice values must be between 1 and 6."}"#
        );
    }

    #[tokio::test]
    async fn get_move_wrong_checkers_on_bar() {
        let web_api = Arc::new(WebApi::try_default()) as DynWebApi;
        let response = router(web_api)
            .oneshot(
                Request::builder()
                    .uri("/move?die1=2&die2=0&p4=4&p5=-5&p25=-2")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");

        let body = body_string(response).await;
        assert_eq!(
            body,
            r#"{"message":"Index 25 is the bar for player x, number of checkers needs to be positive."}"#
        );
    }

    #[tokio::test]
    async fn get_move_body_forced_move() {
        let web_api = Arc::new(WebApi::try_default()) as DynWebApi;
        let response = router(web_api)
            .oneshot(
                Request::builder()
                    .uri("/move?die1=3&die2=1&p5=1&p24=-1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");

        let body = body_string(response).await;
        // The probabilities need to be adapted when the neural net is changed.
        assert_eq!(
            body,
            r#"{"moves":[{"play":[{"from":5,"to":4},{"from":4,"to":1}],"probabilities":{"win":0.12966351,"winG":0.000012590854,"winBg":0.0000037218406,"lose":0.8703365,"loseG":0.000004657087,"loseBg":8.185937e-8}}]}"#
        );
    }

    #[tokio::test]
    async fn get_move_double_roll() {
        let web_api = Arc::new(WebApi::try_default()) as DynWebApi;
        let response = router(web_api)
            .oneshot(
                Request::builder()
                    .uri("/move?die1=1&die2=1&p5=2&p24=-1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");

        let body = body_string(response).await;
        // serde::json
        // The probabilities need to be adapted when the neural net is changed.
        assert_eq!(
            body,
            r#"{"moves":[{"play":[{"from":5,"to":4},{"from":5,"to":4},{"from":4,"to":3},{"from":3,"to":2}],"probabilities":{"win":0.13635689,"winG":0.000007560333,"winBg":0.0000020810992,"lose":0.8636431,"loseG":0.000005637916,"loseBg":9.32699e-8}},{"play":[{"from":5,"to":4},{"from":4,"to":3},{"from":3,"to":2},{"from":2,"to":1}],"probabilities":{"win":0.100783594,"winG":0.0000063406314,"winBg":0.0000019900037,"lose":0.8992164,"loseG":0.000004664952,"loseBg":1.2349493e-7}},{"play":[{"from":5,"to":4},{"from":5,"to":4},{"from":4,"to":3},{"from":4,"to":3}],"probabilities":{"win":0.07513183,"winG":0.000009337691,"winBg":0.0000035386754,"lose":0.92486817,"loseG":0.0000036242066,"loseBg":6.4324674e-8}}]}"#
        );
    }
}
