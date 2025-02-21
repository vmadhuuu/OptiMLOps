
provider "aws" {
  region = "us-west-2"
}

resource "aws_lambda_function" "model_inference" {
  filename         = "lambda_package.zip"
  function_name    = "alzheimer_prediction"
  role            = aws_iam_role.lambda_role.arn
  handler         = "lambda_function.lambda_handler"
  runtime         = "python3.9"
  timeout         = 30
  memory_size     = 1024

  environment {
    variables = {
      MLFLOW_TRACKING_URI = var.mlflow_tracking_uri
      MODEL_VERSION       = var.model_version
    }
  }
}

resource "aws_iam_role" "lambda_role" {
  name = "alzheimer_prediction_lambda_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_policy" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Add CloudWatch policy
resource "aws_iam_role_policy_attachment" "cloudwatch_policy" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchFullAccess"
}

# Add S3 access for model artifacts
resource "aws_iam_role_policy_attachment" "s3_policy" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}