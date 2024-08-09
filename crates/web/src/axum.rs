use crate::web_api::{DiceParams, EvalResponse, MoveResponse, PipParams, WebApi};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::{routing::get, Json, Router};
use engine::evaluator::Evaluator;
use serde::Serialize;
use std::sync::Arc;
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

// This file should handle all axum/tokio related code and know as little about backgammon as possible.

type DynWebApi<T> = Arc<Option<WebApi<T>>>;

pub fn router<T: Evaluator + Send + Sync + 'static>(web_api: DynWebApi<T>) -> Router {
    #[derive(OpenApi)]
    #[openapi(
        paths(crate::axum::get_eval, crate::axum::get_move,),
        components(schemas(
            logic::bg_move::MoveDetail,
            crate::axum::ErrorMessage,
            logic::cube::CubeInfo,
            crate::web_api::EvalResponse,
            crate::web_api::MoveInfo,
            crate::web_api::MoveResponse,
            crate::web_api::ProbabilitiesView,
        )),
        info(
            title = "wildbg",
            description = "Backgammon engine based on neural networks. Source code from [https://github.com/carsten-wenderdel/wildbg](https://github.com/carsten-wenderdel/wildbg)",
        )
    )]
    struct ApiDoc;
    Router::new()
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/eval", get(get_eval))
        .route("/move", get(get_move))
        .with_state(web_api)
}

/// Returned as body along a 4xx or 5xx HTTP status code.
#[derive(Serialize, ToSchema)]
pub struct ErrorMessage {
    #[schema(example = "Player x has more than 15 checkers on the board.")]
    message: String,
}

impl ErrorMessage {
    fn json(message: &str) -> Json<ErrorMessage> {
        Json(ErrorMessage {
            message: message.to_string(),
        })
    }
}

/// Probabilities and cube decision for a position.
///
/// Parameters are a pair of dice and a position.
/// For the position each pip with checkers on it has to be specified via the parameters `p0` through `p25`; pips without checkers can be skipped.
/// We always move from pip 24 to pip 1, so `p1` to `p6` represent the player's (`x`) home board.
///
/// Positive values represent checkers of `x` (you), negative values represent checkers of `o`, the opponent.
/// Checkers already born off don't have to be given, that information is derived from the other arguments.
///
/// As example in the API documentation the starting position is given.
#[utoipa::path(
    get,
    path = "/eval",
    tag = "endpoints",
    params(
        PipParams,
    ),
    responses(
        (status = 200, description = "Successful request. Response includes game outcome probabilities and cube decisions.", body = EvalResponse,
            example = json!({
                "cube": {
                    "double": false,
                    "accept": true
                },
                "probabilities": {
                    "win": 0.62668705,
                    "winG": 0.2308145,
                    "winBg": 0.026135292,
                    "loseG": 0.11035034,
                    "loseBg": 0.015331022
                }
            })
        ),
        (status = 400, description = "Client error, parameters don't represent legal position", body = ErrorMessage,
            example = json!({"message": "Player x has more than 15 checkers on the board."})
        ),
        (status = 500, description = "Server error", body = ErrorMessage,
            example = json!({"message": "Neural net could not be constructed."})
        )
    )
)]
async fn get_eval<T: Evaluator>(
    Query(pips): Query<PipParams>,
    State(web_api): State<DynWebApi<T>>,
) -> Result<Json<EvalResponse>, (StatusCode, Json<ErrorMessage>)> {
    match web_api.as_ref() {
        None => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorMessage::json("Neural net could not be constructed."),
        )),
        Some(web_api) => match web_api.get_eval(pips) {
            Err((status_code, message)) => Err((status_code, ErrorMessage::json(message.as_str()))),
            Ok(eval_response) => Ok(Json(eval_response)),
        },
    }
}

/// Moves for position/dice.
/// Returns a list of legal moves for a certain position and pair of dice, ordered by match equity.
///
/// Parameters are a pair of dice and a position.
/// For the position each pip with checkers on it has to be specified via the parameters `p0` through `p25`; pips without checkers can be skipped.
/// We always move from pip 24 to pip 1, so `p1` to `p6` represent the player's (`x`) home board.
///
/// Positive values represent checkers of `x` (you), negative values represent checkers of `o`, the opponent.
/// Checkers already born off don't have to be given, that information is derived from the other arguments.
///
/// As example in the API documentation the starting position with dice 3 and 1 is given.
#[utoipa::path(
    get,
    path = "/move",
    tag = "endpoints",
    params(
        DiceParams,
        PipParams,
    ),
    responses(
        (status = 200, description = "Successful request. Response includes the best move and other data.", body = MoveResponse,
            example = json!({"moves": [{"play": [{"from": 5, "to": 2}, {"from": 2, "to": 0}], "probabilities": {"win": 0.14432532, "winG": 0.0000012345678, "winBg": 8.311909e-9, "loseG": 0.26282439, "loseBg": 0.024352359}},{"play": [{"from": 5, "to": 2}, {"from": 5, "to": 3}], "probabilities": {"win": 0.74432532, "winG": 0.223456782, "winBg": 0.12345678, "loseG": 0.012345678, "loseBg": 5.311909e-7}}]})
        ),
        (status = 400, description = "Client error, parameters don't represent legal position/dice", body = ErrorMessage,
            example = json!({"message": "Player x has more than 15 checkers on the board."})
        ),
        (status = 500, description = "Server error", body = ErrorMessage,
            example = json!({"message": "Neural net could not be constructed."})
        )
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
            Err(message) => Err((StatusCode::BAD_REQUEST, ErrorMessage::json(message))),
            Ok(move_response) => Ok(Json(move_response)),
        },
    }
}

#[cfg(test)]
mod tests {
    // use crate::{router, DynWebApi};
    use crate::axum::router;
    use crate::axum::DynWebApi;
    use crate::web_api::WebApi;
    use axum::body::Body;
    use axum::http::header::CONTENT_TYPE;
    use engine::evaluator::Evaluator;
    use engine::inputs::ContactInputsGen;
    use engine::onnx::OnnxEvaluator;
    use engine::pos;
    use engine::position::Position;
    use engine::probabilities::{Probabilities, ResultCounter};
    use http_body_util::BodyExt;
    use hyper::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt; // for `oneshot

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
            let forced_move = pos!(x 1:1; o 24:1).sides_switched();
            let double_roll_1 = pos!(x 4:1, 2:1; o 24:1).sides_switched();
            let double_roll_2 = pos!(x 5:1, 1:1; o 24:1).sides_switched();
            let double_roll_3 = pos!(x 3:2; o 24:1).sides_switched();

            if position == &forced_move {
                Probabilities::from(&ResultCounter::new(874, 1, 1, 130, 1, 1))
            } else if position == &double_roll_1 {
                Probabilities::from(&ResultCounter::new(865, 1, 0, 137, 1, 1))
            } else if position == &double_roll_2 {
                Probabilities::from(&ResultCounter::new(12, 1, 1, 16, 3, 1))
            } else if position == &double_roll_3 {
                Probabilities::from(&ResultCounter::new(925, 1, 0, 75, 1, 1))
            } else {
                unreachable!("All evaluated positions should be listed here");
            }
        }
    }

    /// Consumes the response, so use it at the end of the test
    async fn body_string(response: axum::response::Response) -> String {
        let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
        std::str::from_utf8(&body_bytes).unwrap().to_string()
    }

    #[tokio::test]
    async fn get_eval_wrong_checkers_on_bar() {
        let web_api = Arc::new(Some(WebApi::new(EvaluatorFake {})));
        let response = router(web_api)
            .oneshot(
                Request::builder()
                    .uri("/eval?p0=1&p4=4&p5=-5")
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
            r#"{"message":"Index 0 is the bar for player o, number of checkers needs to be negative."}"#
        );
    }

    #[tokio::test]
    async fn get_eval_success() {
        let web_api = Arc::new(Some(WebApi::new(EvaluatorFake {})));
        let response = router(web_api)
            .oneshot(
                Request::builder()
                    .uri("/eval?p1=1&p20=-1&p24=-1") // this position was specified in EvaluatorFake
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");

        let body = body_string(response).await;
        assert_eq!(
            body,
            r#"{"cube":{"double":false,"accept":true},"probabilities":{"win":0.4117647,"winG":0.05882353,"winBg":0.029411765,"loseG":0.11764706,"loseBg":0.029411765}}"#
        );
    }

    #[tokio::test]
    async fn get_move_missing_neural_net() {
        let web_api = Arc::new(None) as DynWebApi<OnnxEvaluator<ContactInputsGen>>;
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
        let web_api = Arc::new(WebApi::try_default());
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
        let web_api = Arc::new(Some(WebApi::new(EvaluatorFake {})));
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
        let web_api = Arc::new(Some(WebApi::new(EvaluatorFake {})));
        let response = router(web_api)
            .oneshot(
                Request::builder()
                    .uri("/move?die1=2&die2=1&p4=4&p5=-5&p25=-2")
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
        assert_eq!(
            body,
            r#"{"moves":[{"play":[{"from":5,"to":4},{"from":4,"to":3},{"from":3,"to":2},{"from":2,"to":1}],"probabilities":{"win":0.5882353,"winG":0.11764706,"winBg":0.029411765,"loseG":0.05882353,"loseBg":0.029411765}},{"play":[{"from":5,"to":4},{"from":5,"to":4},{"from":4,"to":3},{"from":3,"to":2}],"probabilities":{"win":0.13830847,"winG":0.0019900498,"winBg":0.0009950249,"loseG":0.0009950249,"loseBg":0.0}},{"play":[{"from":5,"to":4},{"from":5,"to":4},{"from":4,"to":3},{"from":4,"to":3}],"probabilities":{"win":0.07676969,"winG":0.001994018,"winBg":0.000997009,"loseG":0.000997009,"loseBg":0.0}}]}"#
        );
    }
}
