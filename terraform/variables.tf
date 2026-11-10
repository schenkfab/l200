variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run and Vertex AI"
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name for the ADK Blog Writer Agent"
  type        = string
  default     = "blog-writer-adk-agent"
}

variable "container_image" {
  description = "Artifact Registry container image URI"
  type        = string
}
