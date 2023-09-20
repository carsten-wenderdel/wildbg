use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::{routing::get, Json, Router, Server};
use hyper::Error;
use serde::Serialize;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;
use wildbg::evaluator::Evaluator;
use wildbg::onnx::OnnxEvaluator;
use wildbg::web_api::{DiceParams, MoveResponse, PipParams, WebApi};

/// This file should handle all axum related code and know as little about backgammon as possible.
#[tokio::main]
async fn main() -> Result<(), Error> {
    println!("You can access the server for example via");
    println!(
        "http://localhost:8080/move?die1=3&die2=1&p24=2&p19=-5&p17=-3&p13=5&p12=-5&p8=3&p6=5&p1=-2"
    );
    println!("http://localhost:8080/swagger-ui");

    let web_api = Arc::new(WebApi::try_default()) as DynWebApi<OnnxEvaluator>;
    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, 8080));
    Server::bind(&address)
        .serve(router(web_api).into_make_service())
        .await
}

fn router<T: Evaluator + Send + Sync + 'static>(web_api: DynWebApi<T>) -> Router {
    #[derive(OpenApi)]
    #[openapi(
        paths(
            crate::get_move,
        ),
        components(
            schemas(
                wildbg::bg_move::MoveDetail,
                wildbg::web_api::MoveInfo,
                wildbg::web_api::MoveResponse,
                wildbg::web_api::Probabilities,
            )
        ),
        tags(
            (name = "wildbg", description = "Backgammon Engine based on neural networks")
        )
    )]
    struct ApiDoc;
    Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/cube", get(get_cube))
        .route("/move", get(get_move))
        .with_state(web_api)
}

type DynWebApi<T> = Arc<Option<WebApi<T>>>;

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

#[utoipa::path(
    get,
    path = "/move",
    tag = "endpoints",
    params(
        DiceParams,
        PipParams,
    ),
    responses(
        (status = 200, description = "List of legal moves ordered by match equity. First move is the best one", body = MoveResponse)
    )
)]
async fn get_move<T: Evaluator>(
    Query(dice): Query<DiceParams>,
    Query(pips): Query<PipParams>,
    State(web_api): State<DynWebApi<T>>,
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
    use std::collections::HashMap;
    use std::sync::Arc;
    use tower::ServiceExt; // for `oneshot
    use wildbg::evaluator::{Evaluator, Probabilities};
    use wildbg::onnx::OnnxEvaluator;
    use wildbg::pos;
    use wildbg::position::Position;
    use wildbg::web_api::WebApi;

    struct EvaluatorFake {}
    /// Because of different floating point implementations on different CPUs we don't want to
    /// use a real neural net evaluator in unit tests. Instead we we use a fake evaluator.
    /// This fake evaluator _knows_ evaluations for all given positions.
    /// Based on that we later verify the rendered json.
    /// Note: The probabilities used here are a different type compared to those in the json.
    /// For internal probabilities all six values later add up to 1, for the json `win` and `lose`
    /// will add up to 1.
    /// Also sides are switched, so winning and losing values are switched.
    impl Evaluator for EvaluatorFake {
        fn eval(&self, position: &Position) -> Probabilities {
            let forced_move = pos!(x 1:1; o 24:1).switch_sides();
            let double_roll_1 = pos!(x 4:1, 2:1; o 24:1).switch_sides();
            let double_roll_2 = pos!(x 5:1, 1:1; o 24:1).switch_sides();
            let double_roll_3 = pos!(x 3:2; o 24:1).switch_sides();

            if position == &forced_move {
                Probabilities::new(&[874, 1, 1, 130, 1, 1])
            } else if position == &double_roll_1 {
                Probabilities::new(&[865, 1, 0, 137, 1, 1])
            } else if position == &double_roll_2 {
                Probabilities::new(&[12, 1, 1, 16, 3, 1])
            } else if position == &double_roll_3 {
                Probabilities::new(&[925, 1, 0, 75, 1, 1])
            } else {
                unreachable!("All evaluated positions should be listed here");
            }
        }
    }

    /// Consumes the response, so use it at the end of the test
    async fn body_string(response: axum::response::Response) -> String {
        let body_bytes = hyper::body::to_bytes(response.into_body()).await.unwrap();
        std::str::from_utf8(&body_bytes).unwrap().to_string()
    }

    #[tokio::test]
    async fn get_cube_error() {
        let web_api = Arc::new(None) as DynWebApi<OnnxEvaluator>;
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
        let web_api = Arc::new(None) as DynWebApi<OnnxEvaluator>;
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
        let web_api = Arc::new(WebApi::try_default()) as DynWebApi<OnnxEvaluator>;
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
        let web_api = Arc::new(WebApi::try_default()) as DynWebApi<OnnxEvaluator>;
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
        let web_api = Arc::new(WebApi::try_default()) as DynWebApi<OnnxEvaluator>;
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
        let web_api = Arc::new(Some(WebApi::new(EvaluatorFake {})));
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
            r#"{"moves":[{"play":[{"from":5,"to":4},{"from":4,"to":1}],"probabilities":{"win":0.13095237,"winG":0.001984127,"winBg":0.0009920635,"loseG":0.001984127,"loseBg":0.0009920635}}]}"#
        );
    }

    #[tokio::test]
    async fn get_move_double_roll() {
        let web_api = Arc::new(Some(WebApi::new(EvaluatorFake {})));
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
            r#"{"moves":[{"play":[{"from":5,"to":4},{"from":4,"to":3},{"from":3,"to":2},{"from":2,"to":1}],"probabilities":{"win":0.5882353,"winG":0.11764706,"winBg":0.029411765,"loseG":0.05882353,"loseBg":0.029411765}},{"play":[{"from":5,"to":4},{"from":5,"to":4},{"from":4,"to":3},{"from":3,"to":2}],"probabilities":{"win":0.13830847,"winG":0.0019900498,"winBg":0.0009950249,"loseG":0.0009950249,"loseBg":0.0}},{"play":[{"from":5,"to":4},{"from":5,"to":4},{"from":4,"to":3},{"from":4,"to":3}],"probabilities":{"win":0.07676969,"winG":0.001994018,"winBg":0.000997009,"loseG":0.000997009,"loseBg":0.0}}]}"#
        );
    }
}
