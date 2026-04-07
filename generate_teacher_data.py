"""
Assignment 3 - Phase 1b: Teacher-Generated JSON Instruct Dataset
Uses Llama 3.1 70B Instruct (UTSA hosted) as teacher model to generate
high-quality structured JSON outputs via imitation learning.

Covers all 5 required task types:
1. JSON extraction from unstructured text
2. Schema-constrained generation
3. Exact-label classification with JSON output
4. JSON repair / formatting correction
5. Tool-call argument generation
"""

import json
import os
import time
import random
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
UTSA_API_KEY   = os.environ.get("UTSA_API_KEY_70B")
UTSA_BASE_URL  = os.environ.get("UTSA_BASE_URL_70B")
UTSA_MODEL     = os.environ.get("UTSA_MODEL_70B")

OUTPUT_DIR     = "/work/fpb170/assignment3/data"
OUTPUT_FILE    = f"{OUTPUT_DIR}/json_train.json"
EVAL_FILE      = f"{OUTPUT_DIR}/json_eval.json"
EVAL_SPLIT     = 0.1
MAX_RETRIES    = 3
RANDOM_SEED    = 42
# ───────────────────────────────────────────────────────────────────────────────

# Initialize UTSA OpenAI-compatible client
client = OpenAI(
    api_key=UTSA_API_KEY,
    base_url=UTSA_BASE_URL
)

# ── Task 1: JSON Extraction Prompts ────────────────────────────────────────────
JSON_EXTRACTION_PROMPTS = [
    {
        "instruction": "Extract all person names, dates, and locations mentioned in the text into a JSON object.",
        "input": "On March 15, 2023, Dr. Sarah Johnson and Professor Michael Chen met in Boston to discuss their upcoming research collaboration. They planned to reconvene in San Francisco by April 30th.",
    },
    {
        "instruction": "Extract all product details from the following product description into a structured JSON object.",
        "input": "The Samsung Galaxy S24 Ultra features a 6.8-inch Dynamic AMOLED display, 12GB RAM, 256GB storage, a 5000mAh battery, and a 200MP main camera. It is available in Titanium Black and Titanium Gray at a price of $1299.99.",
    },
    {
        "instruction": "Extract all financial information from the following earnings report excerpt into a JSON object.",
        "input": "Apple Inc. reported Q3 2023 revenue of $81.8 billion, up 2% year-over-year. Net income was $19.9 billion with earnings per share of $1.26. iPhone revenue reached $39.7 billion while Services generated $21.2 billion.",
    },
    {
        "instruction": "Extract all event details from the following announcement into a structured JSON object.",
        "input": "You are cordially invited to the Annual Tech Summit 2024, taking place on September 12-14, 2024 at the Marriott Marquis Hotel in New York City. Registration fee is $599 for early birds until July 31st, and $799 thereafter. Contact events@techsummit.com for more information.",
    },
    {
        "instruction": "Extract all medical information from the patient record excerpt into a JSON object.",
        "input": "Patient John Doe, DOB 05/12/1975, presented with chest pain and shortness of breath. Blood pressure was 145/90 mmHg, heart rate 98 bpm, temperature 98.6F. Prescribed Lisinopril 10mg daily and advised follow-up in 2 weeks.",
    },
    {
        "instruction": "Extract all job posting details from the following listing into a structured JSON object.",
        "input": "Senior Software Engineer at TechCorp Inc. Location: Austin, TX (Hybrid). Salary: $150,000 - $180,000 per year. Requirements: 5+ years Python experience, strong knowledge of AWS, experience with microservices. Benefits include health insurance, 401k matching, and unlimited PTO. Apply by December 31, 2023.",
    },
    {
        "instruction": "Extract all recipe ingredients and their quantities from the following recipe into a JSON object.",
        "input": "Classic Chocolate Cake: You will need 2 cups all-purpose flour, 1.5 cups sugar, 3/4 cup cocoa powder, 2 teaspoons baking soda, 1 teaspoon salt, 2 eggs, 1 cup buttermilk, 1 cup strong black coffee, 1/2 cup vegetable oil, and 2 teaspoons vanilla extract.",
    },
    {
        "instruction": "Extract all flight information from the following itinerary into a structured JSON object.",
        "input": "Flight AA1234 departs Dallas/Fort Worth (DFW) on November 15, 2023 at 08:30 AM and arrives in London Heathrow (LHR) at 10:45 PM local time. Aircraft: Boeing 777-300ER. Seat: 14A (Window). Confirmation code: XK9P2M.",
    },
    {
        "instruction": "Extract all book details from the following library catalog entry into a JSON object.",
        "input": "The Pragmatic Programmer: Your Journey to Mastery by David Thomas and Andrew Hunt. Publisher: Addison-Wesley Professional, 2019. ISBN: 978-0135957059. Pages: 352. Genre: Computer Science, Software Engineering. Available: 3 copies.",
    },
    {
        "instruction": "Extract all real estate property details from the following listing into a structured JSON object.",
        "input": "Beautiful 4-bedroom, 3-bathroom single family home for sale in Austin, TX 78701. Listed at $875,000. Built in 2018, 2,850 sq ft, 0.25 acre lot. Features include: hardwood floors, updated kitchen, 2-car garage, and swimming pool. HOA fees: $250/month.",
    },
    {
        "instruction": "Extract all transaction details from the following bank statement entry into a JSON object.",
        "input": "Transaction Date: 10/28/2023. Description: AMAZON.COM PURCHASE. Amount: -$127.43. Running Balance: $4,521.67. Transaction ID: TXN20231028A4521. Category: Online Shopping. Payment Method: Visa ending 4521.",
    },
    {
        "instruction": "Extract all weather forecast information from the following report into a structured JSON object.",
        "input": "Weather forecast for Chicago, IL for Thursday November 2, 2023: High of 52°F, Low of 38°F. Wind: NW at 15 mph with gusts up to 25 mph. Precipitation: 60% chance of rain in the afternoon. Humidity: 75%. UV Index: 2 (Low). Sunrise: 6:28 AM, Sunset: 4:58 PM.",
    },
    {
        "instruction": "Extract all conference paper details from the following citation into a JSON object.",
        "input": "Smith, J., Johnson, A., & Williams, R. (2023). Attention Mechanisms in Large Language Models: A Comprehensive Survey. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023), pp. 1234-1256. Toronto, Canada.",
    },
    {
        "instruction": "Extract all vehicle information from the following car listing into a structured JSON object.",
        "input": "2022 Tesla Model 3 Long Range AWD. Mileage: 18,500 miles. Color: Pearl White Multi-Coat. Interior: Black. Features: Autopilot, Premium Audio, Glass Roof, 18-inch wheels. Range: 358 miles. Asking price: $38,500. VIN: 5YJ3E1EA8NF123456.",
    },
    {
        "instruction": "Extract all social media post metadata from the following post into a JSON object.",
        "input": "Tweet by @elonmusk posted on October 27, 2023 at 2:45 PM EST: 'Exciting news coming soon! #Tesla #SpaceX'. Engagement: 45,231 likes, 8,902 retweets, 3,421 replies. Impressions: 2.3M. Media: 1 image attached.",
    },
    {
        "instruction": "Extract all insurance policy details from the following document excerpt into a structured JSON object.",
        "input": "Policy Number: AUTO-2023-789456. Policyholder: Maria Garcia. Effective Date: January 1, 2024. Expiration: December 31, 2024. Vehicle: 2021 Honda Civic. Coverage: Liability $100k/$300k, Collision $500 deductible, Comprehensive $250 deductible. Monthly Premium: $127.50.",
    },
    {
        "instruction": "Extract all restaurant details from the following review into a JSON object.",
        "input": "Nobu Dallas, located at 400 Crescent Court, Dallas TX 75201, is open Tuesday through Sunday 5:30PM-10:30PM. Phone: (214) 252-7000. Cuisine: Japanese fusion. Price range: $$$$. Rating: 4.5/5 stars based on 847 reviews. Reservations required. Dress code: Smart casual.",
    },
    {
        "instruction": "Extract all software package details from the following npm package description into a structured JSON object.",
        "input": "Package: react-query v5.0.0. Author: Tanner Linsley. License: MIT. Weekly downloads: 2.3M. Dependencies: 3 packages. Peer dependencies: React 18+. Description: Powerful asynchronous state management for React. Repository: github.com/tanstack/query. Last published: October 15, 2023.",
    },
    {
        "instruction": "Extract all academic course details from the following course catalog entry into a JSON object.",
        "input": "CS 5443: Deep Learning Systems. Credits: 3. Instructor: Dr. Peyman Najafirad. Schedule: Tuesday/Thursday 5:30-6:45 PM. Location: BSE 2.110. Prerequisites: CS 3343, MATH 3103. Description: Advanced topics in deep learning including transformers, generative models, and reinforcement learning from human feedback. Max enrollment: 30.",
    },
    {
        "instruction": "Extract all sports game statistics from the following box score summary into a structured JSON object.",
        "input": "NBA Game: Dallas Mavericks vs Los Angeles Lakers. Date: November 5, 2023. Final Score: Mavericks 118, Lakers 107. Top Scorer: Luka Doncic 38 points, 9 rebounds, 8 assists. LeBron James: 28 points, 10 rebounds, 7 assists. Game duration: 2 hours 18 minutes. Attendance: 19,068.",
    },
    {
        "instruction": "Extract all invoice details from the following billing document into a JSON object.",
        "input": "Invoice #INV-2023-4521 from TechServices LLC to Acme Corp. Invoice Date: November 1, 2023. Due Date: November 30, 2023. Services: Web Development (40 hours at $150/hr = $6,000), Cloud Hosting (1 month = $500). Subtotal: $6,500. Tax (8.25%): $536.25. Total Due: $7,036.25.",
    },
    {
        "instruction": "Extract all podcast episode details from the following description into a structured JSON object.",
        "input": "Lex Fridman Podcast Episode #401: Sam Altman - OpenAI, GPT-5, Sora, Board Saga, Elon Musk, and the Future of AI. Published: March 21, 2024. Duration: 2 hours 43 minutes. Topics: AI safety, AGI timeline, multimodal models. Downloads: 1.2M in first week. Available on: Spotify, Apple Podcasts, YouTube.",
    },
    {
        "instruction": "Extract all patent information from the following patent filing into a JSON object.",
        "input": "Patent Application US20230123456A1. Title: Method and System for Efficient Attention Computation in Transformer Neural Networks. Inventors: John Smith, Jane Doe. Assignee: Tech Innovation Corp. Filing Date: March 15, 2023. Abstract: A novel approach to reducing computational complexity of attention mechanisms from O(n²) to O(n log n).",
    },
    {
        "instruction": "Extract all shipping information from the following tracking update into a structured JSON object.",
        "input": "Tracking Number: 1Z999AA10123456784. Carrier: UPS. Status: Out for Delivery. Estimated Delivery: Today by 8:00 PM. Current Location: Austin, TX distribution center. Origin: Seattle, WA. Shipped Date: November 1, 2023. Package Weight: 3.2 lbs. Dimensions: 12x8x6 inches.",
    },
    {
        "instruction": "Extract all grant funding details from the following award notice into a JSON object.",
        "input": "NSF Award #2312345: Advancing Trustworthy AI Systems through Interpretable Machine Learning. Awarded to: University of Texas at San Antonio. Principal Investigator: Dr. Sarah Williams. Co-PI: Dr. James Chen. Award Amount: $750,000. Duration: September 1, 2023 - August 31, 2026. Program: Trustworthy AI.",
    },
]

# ── Task 2: Schema-Constrained Generation Prompts ──────────────────────────────
SCHEMA_GENERATION_PROMPTS = [
    {
        "instruction": "Generate a valid JSON object that conforms to the given user profile schema.",
        "input": json.dumps({
            "schema": {
                "user_id": "string (UUID)",
                "username": "string (3-20 chars)",
                "email": "string (valid email)",
                "age": "integer (18-100)",
                "subscription": "string (free|pro|enterprise)",
                "created_at": "string (ISO 8601 date)",
                "preferences": {
                    "theme": "string (light|dark)",
                    "notifications": "boolean",
                    "language": "string (ISO 639-1)"
                }
            },
            "context": "Create a profile for a software engineer named Alex"
        })
    },
    {
        "instruction": "Generate a valid JSON object that conforms to the given e-commerce product schema.",
        "input": json.dumps({
            "schema": {
                "product_id": "string",
                "name": "string",
                "price": "number (positive)",
                "currency": "string (ISO 4217)",
                "category": "string",
                "in_stock": "boolean",
                "quantity": "integer (0+)",
                "ratings": {
                    "average": "number (0-5)",
                    "count": "integer"
                },
                "tags": "array of strings"
            },
            "context": "Create a listing for a mechanical keyboard"
        })
    },
    {
        "instruction": "Generate a valid JSON object conforming to the given API error response schema.",
        "input": json.dumps({
            "schema": {
                "error": {
                    "code": "integer (HTTP status code)",
                    "type": "string",
                    "message": "string",
                    "details": "array of objects with field and message",
                    "timestamp": "string (ISO 8601)",
                    "request_id": "string (UUID)"
                }
            },
            "context": "A 422 validation error for a user registration form with invalid email and short password"
        })
    },
    {
        "instruction": "Generate a valid JSON object conforming to the given weather API response schema.",
        "input": json.dumps({
            "schema": {
                "location": {
                    "city": "string",
                    "country": "string (ISO 3166-1 alpha-2)",
                    "lat": "number",
                    "lon": "number"
                },
                "current": {
                    "temp_c": "number",
                    "feels_like_c": "number",
                    "humidity": "integer (0-100)",
                    "wind_kph": "number",
                    "condition": "string"
                },
                "forecast": "array of 3 objects with date, high_c, low_c, condition"
            },
            "context": "Weather for Tokyo, Japan on a sunny autumn day"
        })
    },
    {
        "instruction": "Generate a valid JSON object conforming to the given medical record schema.",
        "input": json.dumps({
            "schema": {
                "patient_id": "string",
                "name": {"first": "string", "last": "string"},
                "dob": "string (YYYY-MM-DD)",
                "blood_type": "string (A+|A-|B+|B-|AB+|AB-|O+|O-)",
                "allergies": "array of strings",
                "medications": "array of objects with name, dosage, frequency",
                "vitals": {
                    "bp_systolic": "integer",
                    "bp_diastolic": "integer",
                    "heart_rate": "integer",
                    "temp_f": "number"
                }
            },
            "context": "A 45-year-old patient with hypertension on medication"
        })
    },
    {
        "instruction": "Generate a valid JSON object conforming to the given GitHub repository schema.",
        "input": json.dumps({
            "schema": {
                "id": "integer",
                "name": "string",
                "full_name": "string (owner/repo)",
                "private": "boolean",
                "description": "string",
                "language": "string",
                "stars": "integer",
                "forks": "integer",
                "open_issues": "integer",
                "license": "string (SPDX identifier)",
                "topics": "array of strings",
                "created_at": "string (ISO 8601)",
                "updated_at": "string (ISO 8601)"
            },
            "context": "A popular open source machine learning library"
        })
    },
    {
        "instruction": "Generate a valid JSON object conforming to the given food order schema.",
        "input": json.dumps({
            "schema": {
                "order_id": "string",
                "customer": {"name": "string", "phone": "string"},
                "items": "array of objects with name, quantity, price, customizations",
                "subtotal": "number",
                "tax": "number",
                "delivery_fee": "number",
                "total": "number",
                "delivery_address": {
                    "street": "string",
                    "city": "string",
                    "state": "string",
                    "zip": "string"
                },
                "estimated_delivery": "string (ISO 8601)"
            },
            "context": "A pizza delivery order for a family of 4"
        })
    },
    {
        "instruction": "Generate a valid JSON object conforming to the given machine learning model metadata schema.",
        "input": json.dumps({
            "schema": {
                "model_id": "string",
                "name": "string",
                "version": "string (semver)",
                "architecture": "string",
                "parameters": "integer",
                "training": {
                    "dataset": "string",
                    "epochs": "integer",
                    "batch_size": "integer",
                    "learning_rate": "number"
                },
                "metrics": {
                    "accuracy": "number (0-1)",
                    "f1_score": "number (0-1)",
                    "loss": "number"
                },
                "created_at": "string (ISO 8601)"
            },
            "context": "A text classification model trained on news articles"
        })
    },
    {
        "instruction": "Generate a valid JSON object conforming to the given hotel booking schema.",
        "input": json.dumps({
            "schema": {
                "booking_id": "string",
                "hotel": {"name": "string", "address": "string", "rating": "number (1-5)"},
                "guest": {"name": "string", "email": "string"},
                "room": {"type": "string", "number": "string", "floor": "integer"},
                "check_in": "string (YYYY-MM-DD)",
                "check_out": "string (YYYY-MM-DD)",
                "nights": "integer",
                "rate_per_night": "number",
                "total": "number",
                "amenities": "array of strings",
                "status": "string (confirmed|pending|cancelled)"
            },
            "context": "A 3-night luxury hotel stay in Paris"
        })
    },
    {
        "instruction": "Generate a valid JSON object conforming to the given CI/CD pipeline configuration schema.",
        "input": json.dumps({
            "schema": {
                "pipeline_name": "string",
                "trigger": "string (push|pull_request|schedule)",
                "stages": "array of objects with name, steps array",
                "environment": {
                    "os": "string",
                    "python_version": "string",
                    "node_version": "string"
                },
                "notifications": {
                    "on_success": "boolean",
                    "on_failure": "boolean",
                    "channels": "array of strings"
                },
                "timeout_minutes": "integer"
            },
            "context": "A Python web application CI/CD pipeline with testing and deployment"
        })
    },
    {
        "instruction": "Generate a valid JSON object conforming to the given cryptocurrency transaction schema.",
        "input": json.dumps({
            "schema": {
                "tx_hash": "string (64 char hex)",
                "block_number": "integer",
                "timestamp": "string (ISO 8601)",
                "from_address": "string",
                "to_address": "string",
                "value_eth": "number",
                "gas_used": "integer",
                "gas_price_gwei": "number",
                "fee_eth": "number",
                "status": "string (success|failed|pending)"
            },
            "context": "An Ethereum transfer of 1.5 ETH between two wallets"
        })
    },
    {
        "instruction": "Generate a valid JSON object conforming to the given university course enrollment schema.",
        "input": json.dumps({
            "schema": {
                "enrollment_id": "string",
                "student": {"id": "string", "name": "string", "major": "string"},
                "course": {
                    "code": "string",
                    "title": "string",
                    "credits": "integer (1-4)",
                    "instructor": "string",
                    "semester": "string"
                },
                "grade": "string or null (A|B|C|D|F|null if in progress)",
                "attendance_rate": "number (0-1) or null",
                "enrolled_at": "string (ISO 8601)"
            },
            "context": "A computer science student enrolled in an AI course currently in progress"
        })
    },
    {
        "instruction": "Generate a valid JSON object conforming to the given IoT sensor reading schema.",
        "input": json.dumps({
            "schema": {
                "device_id": "string",
                "device_type": "string",
                "location": {"building": "string", "room": "string", "floor": "integer"},
                "timestamp": "string (ISO 8601)",
                "readings": {
                    "temperature_c": "number",
                    "humidity_percent": "number (0-100)",
                    "co2_ppm": "integer",
                    "light_lux": "integer"
                },
                "battery_level": "number (0-1)",
                "status": "string (online|offline|warning)"
            },
            "context": "A smart office environmental sensor showing slightly elevated CO2"
        })
    },
    {
        "instruction": "Generate a valid JSON object conforming to the given social media analytics schema.",
        "input": json.dumps({
            "schema": {
                "account_id": "string",
                "platform": "string (twitter|instagram|linkedin|tiktok)",
                "period": {"start": "string (YYYY-MM-DD)", "end": "string (YYYY-MM-DD)"},
                "metrics": {
                    "followers": "integer",
                    "following": "integer",
                    "posts": "integer",
                    "impressions": "integer",
                    "engagements": "integer",
                    "engagement_rate": "number (0-1)"
                },
                "top_post": {"id": "string", "type": "string", "likes": "integer", "shares": "integer"}
            },
            "context": "Analytics for a tech company LinkedIn account for October 2023"
        })
    },
    {
        "instruction": "Generate a valid JSON object conforming to the given legal contract summary schema.",
        "input": json.dumps({
            "schema": {
                "contract_id": "string",
                "type": "string (NDA|SLA|employment|vendor|lease)",
                "parties": "array of objects with name, role, and address",
                "effective_date": "string (YYYY-MM-DD)",
                "expiration_date": "string (YYYY-MM-DD) or null",
                "value": "number or null",
                "currency": "string or null",
                "key_terms": "array of strings (max 5)",
                "status": "string (draft|active|expired|terminated)"
            },
            "context": "An active NDA between a startup and a potential investor"
        })
    },
]

# ── Task 3: Classification Prompts ─────────────────────────────────────────────
CLASSIFICATION_PROMPTS = [
    {
        "instruction": "Classify the sentiment of the following customer review. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "I've been using this laptop for 3 months now and it's been fantastic. The battery lasts all day, it's fast, and the keyboard is comfortable. Worth every penny!",
            "labels": ["positive", "negative", "neutral", "mixed"]
        })
    },
    {
        "instruction": "Classify the urgency level of the following support ticket. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Our entire production database is down. All customers are unable to access the service. This is causing significant revenue loss. Need immediate assistance.",
            "labels": ["critical", "high", "medium", "low"]
        })
    },
    {
        "instruction": "Classify the topic category of the following news headline. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Federal Reserve Raises Interest Rates by 25 Basis Points Amid Inflation Concerns",
            "labels": ["politics", "economics", "technology", "sports", "health", "entertainment", "science"]
        })
    },
    {
        "instruction": "Classify whether the following email is spam or not spam. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Congratulations! You have been selected as our lucky winner! Click here to claim your $1,000,000 prize. Limited time offer! Act now before it expires!!!",
            "labels": ["spam", "not_spam"]
        })
    },
    {
        "instruction": "Classify the programming language of the following code snippet. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nresult = [fibonacci(i) for i in range(10)]",
            "labels": ["python", "javascript", "java", "c++", "ruby", "go", "rust"]
        })
    },
    {
        "instruction": "Classify the emotion expressed in the following social media post. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Just got promoted to Senior Engineer after 4 years of hard work! Can't believe this is finally happening! Dreams do come true!",
            "labels": ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
        })
    },
    {
        "instruction": "Classify the difficulty level of the following LeetCode problem description. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.",
            "labels": ["easy", "medium", "hard"]
        })
    },
    {
        "instruction": "Classify the intent of the following customer service message. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Hi, I ordered a blue shirt size L two weeks ago but received a red shirt size M instead. I need to return this and get the correct item.",
            "labels": ["refund_request", "exchange_request", "complaint", "inquiry", "compliment", "cancellation"]
        })
    },
    {
        "instruction": "Classify whether the following medical symptom description requires emergency care. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Patient reports sudden severe headache described as worst of their life, accompanied by stiff neck, fever of 103°F, and sensitivity to light. Symptoms started 2 hours ago.",
            "labels": ["emergency", "urgent", "routine", "monitor_at_home"]
        })
    },
    {
        "instruction": "Classify the type of the following SQL query. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id HAVING total_spent > 1000 ORDER BY total_spent DESC;",
            "labels": ["select", "insert", "update", "delete", "create", "drop", "alter"]
        })
    },
    {
        "instruction": "Classify the reliability of the following news source excerpt. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Scientists have PROVEN that 5G towers cause cancer! Government hiding the truth! Share before they delete this! My cousin's neighbor's doctor confirmed it!",
            "labels": ["highly_reliable", "reliable", "questionable", "unreliable", "misinformation"]
        })
    },
    {
        "instruction": "Classify the type of machine learning task described. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "We have a dataset of 10,000 customer transactions. Each transaction has 50 features. We want to identify unusual patterns that might indicate fraudulent activity without using any labeled fraud examples.",
            "labels": ["supervised_classification", "supervised_regression", "unsupervised_clustering", "anomaly_detection", "reinforcement_learning", "self_supervised"]
        })
    },
    {
        "instruction": "Classify the legal risk level of the following business action. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Our company plans to collect user browsing history, location data, and purchase behavior without explicitly mentioning it in our privacy policy, then sell this data to third-party advertisers.",
            "labels": ["high_risk", "medium_risk", "low_risk", "no_risk"]
        })
    },
    {
        "instruction": "Classify the security vulnerability type in the following code. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "def get_user(user_id):\n    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n    return db.execute(query)",
            "labels": ["sql_injection", "xss", "csrf", "buffer_overflow", "path_traversal", "insecure_deserialization", "no_vulnerability"]
        })
    },
    {
        "instruction": "Classify the readability level of the following text passage. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "The mitochondria, often referred to as the powerhouse of the cell, are membrane-bound organelles found in the cytoplasm of eukaryotic cells. They generate most of the cell's supply of adenosine triphosphate (ATP) through oxidative phosphorylation.",
            "labels": ["elementary", "middle_school", "high_school", "undergraduate", "graduate", "expert"]
        })
    },
    {
        "instruction": "Classify the type of logical fallacy in the following argument. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "You can't trust John's opinion on climate change because he drives an SUV. His hypocrisy means his arguments have no merit.",
            "labels": ["ad_hominem", "straw_man", "false_dilemma", "slippery_slope", "appeal_to_authority", "circular_reasoning", "no_fallacy"]
        })
    },
    {
        "instruction": "Classify the API HTTP method that should be used for the following operation. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Update only the email address field of an existing user record with ID 12345 in the database, without affecting any other user attributes.",
            "labels": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
        })
    },
    {
        "instruction": "Classify the type of cognitive bias demonstrated in the following scenario. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "After the stock market crashed, an investor said 'I knew this would happen all along. The signs were obvious.' However, before the crash, they had made no such prediction.",
            "labels": ["hindsight_bias", "confirmation_bias", "anchoring_bias", "availability_heuristic", "dunning_kruger", "sunk_cost_fallacy"]
        })
    },
    {
        "instruction": "Classify the writing style of the following paragraph. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Mix the dry ingredients in a large bowl. In a separate bowl, whisk together the wet ingredients. Gradually combine the two mixtures, stirring until just incorporated. Do not overmix. Pour into a greased 9x13 pan and bake at 350°F for 30-35 minutes.",
            "labels": ["narrative", "descriptive", "expository", "persuasive", "instructional", "technical", "conversational"]
        })
    },
    {
        "instruction": "Classify the data structure most appropriate for the described use case. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "We need to store a collection of unique user IDs and frequently check whether a given user ID already exists in our system. The collection can have millions of entries and lookup speed is critical.",
            "labels": ["array", "linked_list", "hash_set", "binary_tree", "heap", "stack", "queue"]
        })
    },
    {
        "instruction": "Classify the tone of the following business email. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Per my last email, I clearly outlined the deadlines that were agreed upon. It is frankly disappointing that this needs to be reiterated. Please ensure all deliverables are submitted by EOD Friday without further delay.",
            "labels": ["professional", "friendly", "passive_aggressive", "aggressive", "formal", "casual", "urgent"]
        })
    },
    {
        "instruction": "Classify the type of cloud architecture pattern described. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Our application is broken down into small, independent services, each running in its own container. Each service handles a specific business function, has its own database, and communicates with other services via REST APIs and message queues.",
            "labels": ["monolithic", "microservices", "serverless", "event_driven", "layered", "hexagonal", "peer_to_peer"]
        })
    },
    {
        "instruction": "Classify the type of chart most appropriate for visualizing the described data. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "We want to show how our company's market share (35%) compares to our three main competitors (28%, 22%, and 15%) for the current fiscal year.",
            "labels": ["bar_chart", "pie_chart", "line_chart", "scatter_plot", "histogram", "heatmap", "box_plot"]
        })
    },
    {
        "instruction": "Classify the severity of the following software bug report. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "When user clicks the 'Export to PDF' button on the reports page, the button text temporarily changes to 'Exporting...' but then the page refreshes and no PDF is downloaded. This occurs 100% of the time in Chrome but works fine in Firefox.",
            "labels": ["blocker", "critical", "major", "minor", "trivial"]
        })
    },
    {
        "instruction": "Classify the type of organizational structure described. Return a JSON object with label, confidence, and reason.",
        "input": json.dumps({
            "text": "Employees report to both a functional manager (e.g., Engineering Manager) and a project manager simultaneously. Resources are shared across multiple projects, and employees may work on several projects at once.",
            "labels": ["hierarchical", "flat", "matrix", "divisional", "network", "team_based", "circular"]
        })
    },
]

# ── Task 4: JSON Repair Prompts ────────────────────────────────────────────────
JSON_REPAIR_PROMPTS = [
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": "{name: 'John Doe', age: 30, email: 'john@example.com'}"
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"user": {"id": 123, "name": "Alice", "roles": ["admin", "user",], "active": True}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"product": {"name": "Laptop", "price": $999.99, "currency": "USD", "available": yes}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": "{'server': {'host': 'localhost', 'port': 8080, 'debug': true, 'workers': 4,}}"
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"items": [{"id": 1, "name": "Apple", "price": 0.99}, {"id": 2, "name": "Banana", "price": 0.59}, {"id": 3 "name": "Cherry" "price": 2.99}]}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"config": {"database": {"host": "db.example.com", "port": "5432", "name": "mydb", "ssl": True, "timeout": None}}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"order": {"id": "ORD-123", "customer": "Bob Smith", "items": [{"sku": "A1", "qty": 2, "price": 15.99], "total": 31.98}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{employee: {firstName: "Jane", lastName: "Smith", department: "Engineering", salary: 95000, fullTime: true}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"api_response": {"status": 200, "data": {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}, ]}, "meta": {"total": 2, "page": 1}}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": "{'coordinates': {'latitude': 29.4241, 'longitude': -98.4936, 'altitude': 650, 'accuracy': 10.5, 'timestamp': '2023-11-01T10:30:00Z'}}"
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"movie": {"title": "Inception", "year": 2010, "director": "Christopher Nolan", "rating": 8.8, "genres": ["Action", "Sci-Fi", "Thriller", "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt"]}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"settings": {"theme": dark, "fontSize": 14, "autoSave": True, "backupInterval": 300, "language": "en-US", "shortcuts": {"save": "Ctrl+S", "undo": "Ctrl+Z",}}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"student": {"name": "Carlos Rivera", "gpa": 3.85, "courses": [{"code": "CS101", "grade": "A"}, {"code": "MATH201", "grade": "B+"}, {"code": "ENG301", "grade": "A-"}], "graduated": False, "graduationYear": null}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"recipe": {"name": "Pancakes", "servings": 4, "ingredients": [{"item": "flour", "amount": "2 cups"}, {item: "eggs", "amount": "2"}, {"item": "milk", "amount": "1.5 cups"}], "time_minutes": 20}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"deployment": {"version": "2.1.0", "environment": "production", "timestamp": 2023-11-01T15:30:00Z, "services": ["api", "worker", "scheduler"], "healthcheck": {"enabled": true, "interval": 30, "timeout": 5}}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": "{'analytics': {'pageViews': 12543, 'uniqueVisitors': 8921, 'bounceRate': 0.42, 'avgSessionDuration': '3:45', 'topPages': ['/home', '/products', '/about'], 'conversionRate': 0.035,}}"
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"sensor_data": {"device_id": "SENS-001", "readings": {"temperature": 22.5°C, "humidity": 65%, "pressure": 1013.25 hPa}, "status": "normal", "battery": 0.87}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"subscription": {"plan": "premium", "price": 29.99, "billing_cycle": "monthly", "features": ["unlimited_storage", "priority_support", "api_access",], "auto_renew": True, "next_billing": "2024-01-01"}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"git_commit": {"hash": "a1b2c3d4e5f6", "author": {"name": John Smith, "email": "john@dev.com"}, "message": "Fix critical bug in payment processor", "timestamp": "2023-10-28T09:15:00Z", "files_changed": 3}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"inventory": {"warehouse": "WH-001", "items": [{"sku": "ITEM-A", "quantity": 150, "reorder_point": 50}, {"sku": "ITEM-B", "quantity": 0, "reorder_point": 25}, {"sku": "ITEM-C" "quantity": 75 "reorder_point": 30}], "last_updated": "2023-11-01"}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": "{'payment': {'transaction_id': 'TXN-98765', 'amount': 249.99, 'currency': 'USD', 'method': 'credit_card', 'card': {'last4': '4242', 'brand': 'Visa', 'exp_month': 12, 'exp_year': 2025}, 'status': 'completed',}}"
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"notification": {"id": "NOTIF-555", "type": "alert", "title": "System Alert", "message": "CPU usage exceeded 90% threshold", "severity": high, "timestamp": "2023-11-01T08:00:00Z", "acknowledged": false, "recipients": ["admin@company.com"]}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"model_config": {"architecture": "transformer", "layers": 12, "heads": 8, "hidden_size": 768, "vocab_size": 50257, "max_position_embeddings": 1024, "dropout": 0.1, "activation": "gelu", "parameters": 117M}}'
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": "{'task': {'id': 'TASK-2023-456', 'title': 'Implement authentication', 'assignee': 'Sarah Chen', 'priority': 'high', 'status': 'in_progress', 'due_date': '2023-11-15', 'subtasks': ['Design OAuth flow', 'Implement JWT', 'Write tests',], 'story_points': 8}}"
    },
    {
        "instruction": "Fix the following malformed JSON and return only the corrected valid JSON.",
        "input": '{"survey_response": {"respondent_id": "R-12345", "submitted_at": "2023-10-30T14:22:00Z", "answers": [{"question_id": 1, "answer": "Very satisfied"}, {"question_id": 2, "answer": 9}, {"question_id": 3 "answer": ["Ease of use", "Customer support"]}], "nps_score": 9, "completed": True}}'
    },
]

# ── Task 5: Tool-Call Argument Generation Prompts ──────────────────────────────
TOOL_CALL_PROMPTS = [
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "send_email",
            "description": "Send an email to one or more recipients",
            "parameters": {
                "to": "array of email addresses",
                "subject": "string",
                "body": "string",
                "cc": "array of email addresses (optional)",
                "priority": "string: low|normal|high (optional, default: normal)"
            },
            "context": "Send a high priority meeting reminder to john@company.com and sarah@company.com about the board meeting tomorrow at 9 AM, with a copy to the CEO at ceo@company.com"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "search_products",
            "description": "Search for products in an e-commerce catalog",
            "parameters": {
                "query": "string",
                "category": "string (optional)",
                "min_price": "number (optional)",
                "max_price": "number (optional)",
                "sort_by": "string: price_asc|price_desc|rating|newest (optional)",
                "page": "integer (optional, default: 1)",
                "limit": "integer (optional, default: 20, max: 100)"
            },
            "context": "Search for wireless noise-canceling headphones under $300, sorted by rating, showing 10 results"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "create_calendar_event",
            "description": "Create a new event in the user's calendar",
            "parameters": {
                "title": "string",
                "start_datetime": "string (ISO 8601)",
                "end_datetime": "string (ISO 8601)",
                "location": "string (optional)",
                "description": "string (optional)",
                "attendees": "array of email addresses (optional)",
                "reminder_minutes": "integer (optional, default: 30)",
                "recurring": "string: none|daily|weekly|monthly (optional, default: none)"
            },
            "context": "Schedule a weekly team standup every Monday at 9:30 AM for 30 minutes in Conference Room B, invite team@company.com, with a 15-minute reminder"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "translate_text",
            "description": "Translate text from one language to another",
            "parameters": {
                "text": "string",
                "source_language": "string (ISO 639-1, optional: auto-detect if not provided)",
                "target_language": "string (ISO 639-1)",
                "formality": "string: formal|informal (optional, default: formal)",
                "preserve_formatting": "boolean (optional, default: true)"
            },
            "context": "Translate 'Hello, how are you today? We hope you enjoy our service.' to Spanish, informally"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "get_weather_forecast",
            "description": "Retrieve weather forecast for a location",
            "parameters": {
                "location": "string (city name or coordinates)",
                "days": "integer (1-14, default: 7)",
                "units": "string: metric|imperial|kelvin (default: metric)",
                "include_hourly": "boolean (default: false)",
                "include_alerts": "boolean (default: true)"
            },
            "context": "Get a 5-day weather forecast for San Francisco in Fahrenheit with hourly breakdowns"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "process_payment",
            "description": "Process a payment transaction",
            "parameters": {
                "amount": "number",
                "currency": "string (ISO 4217)",
                "payment_method_id": "string",
                "customer_id": "string",
                "description": "string (optional)",
                "metadata": "object (optional)",
                "capture_immediately": "boolean (default: true)",
                "idempotency_key": "string (optional)"
            },
            "context": "Charge customer cust_123 $49.99 USD using their saved payment method pm_456 for a monthly subscription renewal"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "query_database",
            "description": "Execute a parameterized database query",
            "parameters": {
                "table": "string",
                "operation": "string: select|insert|update|delete",
                "conditions": "object (field-value pairs for WHERE clause)",
                "fields": "array of strings (for SELECT, optional: all fields if not specified)",
                "order_by": "string (optional)",
                "limit": "integer (optional)",
                "offset": "integer (optional)"
            },
            "context": "Get the top 10 most recent orders for customer ID 789 that have status 'shipped', showing only order_id, created_at, and total"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "generate_report",
            "description": "Generate a business analytics report",
            "parameters": {
                "report_type": "string: sales|inventory|customer|financial|performance",
                "start_date": "string (YYYY-MM-DD)",
                "end_date": "string (YYYY-MM-DD)",
                "filters": "object (optional)",
                "group_by": "string: day|week|month|quarter|year (optional)",
                "format": "string: pdf|excel|csv|json (default: pdf)",
                "include_charts": "boolean (default: true)",
                "email_to": "array of email addresses (optional)"
            },
            "context": "Generate a monthly sales report for Q3 2023 (July-September), grouped by month, in Excel format with charts, emailed to management@company.com"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "upload_file",
            "description": "Upload a file to cloud storage",
            "parameters": {
                "file_path": "string (local path)",
                "destination": "string (cloud storage path)",
                "bucket": "string",
                "content_type": "string (MIME type, optional: auto-detect)",
                "access": "string: private|public_read|authenticated (default: private)",
                "metadata": "object (optional)",
                "overwrite": "boolean (default: false)",
                "compress": "boolean (default: false)"
            },
            "context": "Upload a profile photo from /tmp/photo.jpg to the users/avatars/ folder in the media-bucket, making it publicly readable, overwriting if exists"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "create_user",
            "description": "Create a new user account in the system",
            "parameters": {
                "email": "string (valid email, unique)",
                "password": "string (min 8 chars)",
                "first_name": "string",
                "last_name": "string",
                "role": "string: admin|manager|user|viewer (default: user)",
                "organization_id": "string (optional)",
                "send_welcome_email": "boolean (default: true)",
                "require_password_change": "boolean (default: false)"
            },
            "context": "Create an account for Dr. Emily Watson, email emily.watson@hospital.org, temporary password 'Welcome123!', as a manager role, requiring her to change password on first login, in organization org_hospital_456"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "run_ml_inference",
            "description": "Run inference on a deployed machine learning model",
            "parameters": {
                "model_id": "string",
                "model_version": "string (optional, default: latest)",
                "inputs": "object or array",
                "batch_size": "integer (optional, default: 1)",
                "return_probabilities": "boolean (optional, default: false)",
                "timeout_ms": "integer (optional, default: 5000)"
            },
            "context": "Run sentiment analysis using model 'sentiment-v2' on the text 'This product exceeded all my expectations!' with probability scores"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "send_notification",
            "description": "Send push notification to mobile app users",
            "parameters": {
                "user_ids": "array of strings or 'all' for broadcast",
                "title": "string (max 50 chars)",
                "body": "string (max 200 chars)",
                "data": "object (optional, custom payload)",
                "platform": "string: ios|android|both (default: both)",
                "badge_count": "integer (optional)",
                "sound": "string (optional, default: default)",
                "scheduled_at": "string (ISO 8601, optional: send immediately if not specified)"
            },
            "context": "Send an order shipped notification to user user_789 for iOS and Android with their order number ORD-12345 in the data payload"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "apply_discount",
            "description": "Apply a discount code to a shopping cart",
            "parameters": {
                "cart_id": "string",
                "discount_code": "string",
                "customer_id": "string (optional, for validation)",
                "apply_to": "string: entire_order|specific_items|shipping (default: entire_order)",
                "override_existing": "boolean (default: false)"
            },
            "context": "Apply discount code SAVE20 to cart cart_abc123 for customer cust_xyz789"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "backup_database",
            "description": "Create a backup of a database",
            "parameters": {
                "database_name": "string",
                "backup_type": "string: full|incremental|differential",
                "destination": "string (storage path or S3 URI)",
                "compression": "boolean (default: true)",
                "encryption": "boolean (default: true)",
                "retention_days": "integer (optional, default: 30)",
                "notify_on_complete": "string (email, optional)"
            },
            "context": "Create a full encrypted compressed backup of the production database 'prod_db' to S3 bucket s3://backups/prod/, keep for 90 days, notify devops@company.com"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "resize_image",
            "description": "Resize and optionally convert an image",
            "parameters": {
                "input_path": "string",
                "output_path": "string",
                "width": "integer (pixels, optional)",
                "height": "integer (pixels, optional)",
                "maintain_aspect_ratio": "boolean (default: true)",
                "output_format": "string: jpeg|png|webp|gif (optional: same as input)",
                "quality": "integer (1-100, default: 85, for lossy formats)",
                "max_file_size_kb": "integer (optional)"
            },
            "context": "Resize banner.png to exactly 1200x628 pixels for social media sharing, save as banner_social.jpg at 90% quality"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "scrape_webpage",
            "description": "Extract structured data from a webpage",
            "parameters": {
                "url": "string",
                "selectors": "object (CSS selector to field name mapping)",
                "wait_for": "string (CSS selector to wait for, optional)",
                "javascript_enabled": "boolean (default: true)",
                "timeout_seconds": "integer (default: 30)",
                "retry_count": "integer (default: 3)",
                "proxy": "string (optional)"
            },
            "context": "Scrape product name, price, and availability from https://shop.example.com/product/123, waiting for the price element .product-price to load"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "schedule_job",
            "description": "Schedule a background job or cron task",
            "parameters": {
                "job_name": "string",
                "job_type": "string: script|function|api_call",
                "schedule": "string (cron expression or interval like '1h', '30m')",
                "payload": "object (optional)",
                "max_retries": "integer (default: 3)",
                "timeout_minutes": "integer (default: 60)",
                "enabled": "boolean (default: true)",
                "notify_on_failure": "string (email, optional)"
            },
            "context": "Schedule a daily data sync job 'sync_analytics' that runs at 2 AM UTC every day, with 2 retry attempts, 30-minute timeout, notifying alerts@company.com on failure"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "calculate_shipping",
            "description": "Calculate shipping cost and estimated delivery time",
            "parameters": {
                "origin_zip": "string",
                "destination_zip": "string",
                "weight_lbs": "number",
                "dimensions": "object with length, width, height in inches",
                "service_level": "string: standard|express|overnight|economy",
                "insurance_value": "number (optional)",
                "signature_required": "boolean (default: false)"
            },
            "context": "Calculate express shipping from Austin TX (78701) to New York NY (10001) for a 5lb package that is 12x10x8 inches, with $200 insurance"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "analyze_sentiment_batch",
            "description": "Analyze sentiment for multiple texts at once",
            "parameters": {
                "texts": "array of strings",
                "model": "string: basic|advanced|domain_specific (default: advanced)",
                "language": "string (ISO 639-1, default: en)",
                "aspects": "array of strings (optional, for aspect-based sentiment)",
                "return_keywords": "boolean (default: false)",
                "confidence_threshold": "number (0-1, default: 0.5)"
            },
            "context": "Analyze sentiment of 3 customer reviews about food delivery speed and driver friendliness, returning keywords"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "generate_invoice",
            "description": "Generate a PDF invoice for billing",
            "parameters": {
                "invoice_number": "string",
                "issue_date": "string (YYYY-MM-DD)",
                "due_date": "string (YYYY-MM-DD)",
                "seller": "object with name, address, email, tax_id",
                "buyer": "object with name, address, email",
                "line_items": "array of objects with description, quantity, unit_price",
                "tax_rate": "number (0-1, optional)",
                "discount": "number (optional, amount in currency)",
                "currency": "string (ISO 4217, default: USD)",
                "notes": "string (optional)",
                "template": "string: standard|professional|minimal (default: standard)"
            },
            "context": "Generate an invoice INV-2023-100 from TechCorp LLC to ClientCo Inc for 10 hours of consulting at $200/hr and software license for $500, with 8.25% tax, due in 30 days"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "create_git_branch",
            "description": "Create a new branch in a Git repository",
            "parameters": {
                "repository": "string (owner/repo format)",
                "branch_name": "string",
                "source_branch": "string (default: main)",
                "source_commit": "string (optional, SHA hash)",
                "set_as_default": "boolean (default: false)",
                "protection_rules": "object (optional)"
            },
            "context": "Create a feature branch 'feature/user-authentication' in myorg/webapp repository from the main branch"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "aggregate_metrics",
            "description": "Aggregate and summarize time-series metrics",
            "parameters": {
                "metric_names": "array of strings",
                "start_time": "string (ISO 8601)",
                "end_time": "string (ISO 8601)",
                "aggregation": "string: avg|sum|min|max|count|p95|p99",
                "interval": "string: 1m|5m|15m|1h|1d",
                "filters": "object (optional, tag key-value pairs)",
                "fill_missing": "string: none|zero|previous|interpolate (default: none)"
            },
            "context": "Get average CPU and memory usage for all production servers in us-east-1 from November 1-7 2023, aggregated in 1-hour intervals, filling missing values with previous"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "deploy_application",
            "description": "Deploy an application to a cloud environment",
            "parameters": {
                "app_name": "string",
                "version": "string (semver or git tag)",
                "environment": "string: development|staging|production",
                "region": "string (cloud region)",
                "replicas": "integer (default: 1)",
                "health_check_path": "string (default: /health)",
                "rollback_on_failure": "boolean (default: true)",
                "notify_slack_channel": "string (optional)",
                "deployment_strategy": "string: rolling|blue_green|canary (default: rolling)"
            },
            "context": "Deploy version 2.5.1 of the payments-service to production in us-east-1 with 3 replicas using blue-green deployment, notify #deployments Slack channel"
        })
    },
    {
        "instruction": "Generate a JSON object representing a function call with the correct named parameters.",
        "input": json.dumps({
            "function_name": "export_data",
            "description": "Export data from the system in various formats",
            "parameters": {
                "resource_type": "string: users|orders|products|analytics|logs",
                "format": "string: csv|json|xlsx|parquet",
                "filters": "object (optional)",
                "fields": "array of strings (optional, all fields if not specified)",
                "date_range": "object with start and end (optional)",
                "limit": "integer (optional, all records if not specified)",
                "include_headers": "boolean (default: true, for CSV)",
                "compress": "boolean (default: false)",
                "destination_email": "string (optional, send link when ready)"
            },
            "context": "Export all orders from October 2023 as a compressed Excel file with only order_id, customer_email, total, and status fields, emailed to analyst@company.com when ready"
        })
    },
]


def call_teacher_model(instruction, input_text, max_retries=MAX_RETRIES):
    """
    Call Llama 3.1 70B (UTSA hosted) to generate a JSON response.
    Validates that the response is valid JSON.
    Retries up to max_retries times if invalid.
    """
    # Build the full prompt following assignment template
    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    system_message = (
        "You are a precise JSON generation assistant. "
        "You ALWAYS respond with valid JSON only. "
        "No explanations, no markdown code blocks, no preamble. "
        "Just the raw JSON object."
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=UTSA_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            raw_output = response.choices[0].message.content.strip()

            # Strip markdown code blocks if model added them
            if raw_output.startswith("```"):
                lines = raw_output.split("\n")
                raw_output = "\n".join(lines[1:-1])

            # Validate JSON
            json.loads(raw_output)
            return raw_output

        except json.JSONDecodeError:
            print(f"    ⚠ Attempt {attempt+1}/{max_retries}: Invalid JSON, retrying...")
            time.sleep(1)
        except Exception as e:
            print(f"    ⚠ Attempt {attempt+1}/{max_retries}: API error: {e}, retrying...")
            time.sleep(2)

    return None  # Failed after all retries


def generate_dataset():
    """
    Generate the full teacher JSON dataset covering all 5 task types.
    """
    print("=" * 60)
    print("Assignment 3 - Teacher-Generated JSON Dataset")
    print("Using Llama 3.1 70B Instruct (UTSA hosted)")
    print("=" * 60)

    # Verify API config
    if not UTSA_API_KEY or not UTSA_BASE_URL or not UTSA_MODEL:
        print("❌ ERROR: UTSA API environment variables not set!")
        print("   Run: source ~/.bashrc")
        return

    print(f"\nModel:    {UTSA_MODEL}")
    print(f"Base URL: {UTSA_BASE_URL}")

    all_tasks = [
        ("1. JSON Extraction",              JSON_EXTRACTION_PROMPTS),
        ("2. Schema-Constrained Generation", SCHEMA_GENERATION_PROMPTS),
        ("3. Classification with JSON",      CLASSIFICATION_PROMPTS),
        ("4. JSON Repair",                   JSON_REPAIR_PROMPTS),
        ("5. Tool-Call Generation",          TOOL_CALL_PROMPTS),
    ]

    all_examples = []
    task_counts  = {}

    for task_name, prompts in all_tasks:
        print(f"\n{'─'*60}")
        print(f"Generating: {task_name}")
        print(f"Prompts:    {len(prompts)}")
        print(f"{'─'*60}")

        task_examples = []
        for i, prompt in enumerate(prompts):
            instruction = prompt["instruction"]
            input_text  = prompt.get("input", "")

            print(f"  [{i+1:2d}/{len(prompts)}] ", end="", flush=True)

            output = call_teacher_model(instruction, input_text)

            if output is not None:
                task_examples.append({
                    "instruction": instruction,
                    "input":       input_text,
                    "output":      output,
                    "task_type":   task_name
                })
                print("✅")
            else:
                print("❌ SKIPPED (invalid JSON after all retries)")

        task_counts[task_name] = len(task_examples)
        all_examples.extend(task_examples)
        print(f"  Collected: {len(task_examples)}/{len(prompts)} examples")

    # ── Shuffle and split ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Total valid examples: {len(all_examples)}")

    random.seed(RANDOM_SEED)
    random.shuffle(all_examples)

    split_idx  = int(len(all_examples) * (1 - EVAL_SPLIT))
    train_data = all_examples[:split_idx]
    eval_data  = all_examples[split_idx:]

    # ── Save ───────────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        json.dump(train_data, f, indent=2)

    with open(EVAL_FILE, "w") as f:
        json.dump(eval_data, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nTask breakdown:")
    for task_name, count in task_counts.items():
        print(f"  {task_name}: {count} examples")
    print(f"\nTrain set: {len(train_data)} examples → {OUTPUT_FILE}")
    print(f"Eval set:  {len(eval_data)} examples  → {EVAL_FILE}")
    print(f"\n✅ Teacher JSON dataset ready!")


if __name__ == "__main__":
    generate_dataset()