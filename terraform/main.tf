terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Google Cloud Secret Manager for storing API keys and sensitive runtime secrets
resource "google_secret_manager_secret" "google_api_key" {
  secret_id = "GOOGLE_API_KEY"

  replication {
    auto {}
  }
}

# Dedicated least-privilege Service Account for the ADK Blog Writer Agent
resource "google_service_account" "blog_writer_sa" {
  account_id   = "blog-writer-adk-sa"
  display_name = "Blog Writer ADK Agent Service Account"
}

resource "google_secret_manager_secret_iam_member" "secret_accessor" {
  secret_id = google_secret_manager_secret.google_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.blog_writer_sa.email}"
}

resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.blog_writer_sa.email}"
}

# Cloud Run v2 Service hosting the containerized Google ADK Agent
resource "google_cloud_run_v2_service" "blog_writer_service" {
  name     = var.service_name
  location = var.region

  template {
    service_account = google_service_account.blog_writer_sa.email

    containers {
      image = var.container_image

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }
      env {
        name  = "USE_SECRET_MANAGER"
        value = "true"
      }
      env {
        name = "GOOGLE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.google_api_key.secret_id
            version = "latest"
          }
        }
      }
    }
  }
}
