use crate::state::AppState;
use axum::{
    extract::State,
    http::StatusCode,
    routing::post,
    Json, Router,
};
use rmcp::model::{Role, Content};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use utoipa::ToSchema;

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct SamplingMessage {
    /// The role of the message sender (User or Assistant)
    pub role: Role,
    /// The actual content of the message (text, image, etc.)
    pub content: Content,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct SamplingRequest {
    /// The session ID for the current agent session
    pub session_id: String,
    /// The extension name that is making the sampling request
    pub extension_name: String,
    /// The messages to send to the LLM
    pub messages: Vec<SamplingMessage>,
    /// Optional model preferences
    pub model_preferences: Option<Vec<String>>,
    /// Optional system prompt
    pub system_prompt: Option<String>,
    /// Whether to include context
    pub include_context: Option<String>,
    /// Temperature for the model
    pub temperature: Option<f64>,
    /// Maximum tokens to generate
    pub max_tokens: i32,
    /// Stop sequences
    pub stop_sequences: Option<Vec<String>>,
    /// Additional metadata
    pub metadata: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct SamplingResponse {
    /// The generated message from the model
    pub message: SamplingMessage,
    /// The model used for generation
    pub model: String,
    /// The reason for stopping
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct SamplingApprovalRequest {
    /// The session ID for the current agent session
    pub session_id: String,
    /// The original sampling request
    pub original_request: SamplingRequest,
    /// The user's action: approve, deny, or edit
    pub action: String,
    /// If action is "edit", this contains the edited messages
    pub edited_messages: Option<Vec<SamplingMessage>>,
}

#[utoipa::path(
    post,
    path = "/sampling/request",
    request_body = SamplingRequest,
    responses(
        (status = 200, description = "Sampling request received", body = Value),
        (status = 401, description = "Unauthorized - invalid secret key"),
        (status = 500, description = "Internal server error")
    )
)]
pub async fn handle_sampling_request(
    State(_state): State<Arc<AppState>>,
    Json(_request): Json<SamplingRequest>,
) -> Result<Json<Value>, StatusCode> {
    // This endpoint will be called by the frontend when it receives a sampling request
    // It returns immediately, and the frontend will show the modal
    // The actual processing happens in handle_sampling_approval
    
    Ok(Json(serde_json::json!({
        "status": "pending",
        "message": "Sampling request received. Awaiting user approval."
    })))
}

#[utoipa::path(
    post,
    path = "/sampling/approve",
    request_body = SamplingApprovalRequest,
    responses(
        (status = 200, description = "Sampling approval processed", body = SamplingResponse),
        (status = 401, description = "Unauthorized - invalid secret key"),
        (status = 500, description = "Internal server error")
    )
)]
pub async fn handle_sampling_approval(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SamplingApprovalRequest>,
) -> Result<Json<SamplingResponse>, StatusCode> {
    match request.action.as_str() {
        "approve" => {
            // Use the original messages
            process_sampling(&state, &request.session_id, request.original_request).await
        }
        "edit" => {
            // Use the edited messages
            if let Some(edited_messages) = request.edited_messages {
                let mut edited_request = request.original_request;
                edited_request.messages = edited_messages;
                process_sampling(&state, &request.session_id, edited_request).await
            } else {
                Err(StatusCode::BAD_REQUEST)
            }
        }
        "deny" => {
            // Return a denial response
            Ok(Json(SamplingResponse {
                message: SamplingMessage {
                    role: Role::Assistant,
                    content: Content::text("Sampling request denied by user."),
                },
                model: "none".to_string(),
                stop_reason: Some("user_denied".to_string()),
            }))
        }
        _ => Err(StatusCode::BAD_REQUEST),
    }
}

async fn process_sampling(
    state: &Arc<AppState>,
    session_id: &str,
    request: SamplingRequest,
) -> Result<Json<SamplingResponse>, StatusCode> {
    // Get the agent for this session
    let agent = state.get_agent_for_route(session_id.to_string()).await?;
    
    // Get the provider from the agent's extension manager
    let provider = agent.extension_manager.get_provider().await
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Convert SamplingMessage to Message for the provider
    let messages: Vec<goose::conversation::message::Message> = request.messages
        .iter()
        .map(|msg| {
            let mut message = match msg.role {
                Role::User => goose::conversation::message::Message::user(),
                Role::Assistant => goose::conversation::message::Message::assistant(),
            };
            // Add content
            if let Some(text) = msg.content.as_text() {
                message = message.with_text(&text.text);
            } else {
                message = message.with_content(msg.content.clone().into());
            }
            message
        })
        .collect();
    
    // Use system prompt from request or default
    let system_prompt = request.system_prompt
        .as_deref()
        .unwrap_or("You are a helpful assistant");
    
    // Call the provider's complete method
    let (response, usage) = provider
        .complete(system_prompt, &messages, &[])
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Extract the response content
    let response_content = if let Some(content) = response.content.first() {
        match content {
            goose::conversation::message::MessageContent::Text(text) => {
                Content::text(&text.text)
            }
            goose::conversation::message::MessageContent::Image(img) => {
                Content::image(&img.data, &img.mime_type)
            }
            _ => Content::text(""),
        }
    } else {
        Content::text("")
    };
    
    // Create the response
    Ok(Json(SamplingResponse {
        message: SamplingMessage {
            role: Role::Assistant,
            content: response_content,
        },
        model: usage.model,
        stop_reason: Some("end_turn".to_string()),
    }))
}

pub fn routes(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/sampling/request", post(handle_sampling_request))
        .route("/sampling/approve", post(handle_sampling_approval))
        .with_state(state)
}
