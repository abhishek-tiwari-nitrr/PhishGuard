import os
from dotenv import load_dotenv

load_dotenv()


API_URL = os.getenv("API_URL")
PIPELINE_STEPS = [
    ("Data Ingestion", "Connecting to MongoDB and fetching raw data..."),
    ("Data Validation", "Validating schema, columns and checking for data drift..."),
    ("Data Transformation", "Applying KNN imputation and saving numpy arrays..."),
    (
        "Model Training",
        "Training 5 classifiers with GridSearchCV, logging to MLflow...",
    ),
    ("Model Evaluation", "Comparing new model against production model..."),
    ("Model Pusher", "Promoting accepted model to production path..."),
]
FEATURE_META = [
    (
        "having_IP_Address",
        "IP address in URL?",
        {1: "No (Safe)", -1: "Yes (Suspicious)"},
    ),
    ("URL_Length", "URL length", {1: "Short", 0: "Medium", -1: "Long"}),
    ("Shortining_Service", "URL shortening service?", {1: "No", -1: "Yes"}),
    ("having_At_Symbol", "'@' symbol in URL?", {1: "No", -1: "Yes"}),
    ("double_slash_redirecting", "'//' redirect in URL?", {1: "No", -1: "Yes"}),
    ("Prefix_Suffix", "'-' prefix/suffix in domain?", {1: "No", -1: "Yes"}),
    ("having_Sub_Domain", "Sub-domain count", {1: "One", 0: "Two", -1: "Many"}),
    (
        "SSLfinal_State",
        "SSL certificate state",
        {1: "Valid HTTPS", 0: "No HTTPS", -1: "Suspicious"},
    ),
    (
        "Domain_registeration_length",
        "Domain registration length",
        {1: "≥ 1 year (Safe)", -1: "< 1 year (Suspicious)"},
    ),
    ("Favicon", "Favicon from same domain?", {1: "Yes", -1: "No"}),
    ("port", "Non-standard port?", {1: "No", -1: "Yes"}),
    ("HTTPS_token", "'https' in domain part?", {1: "No", -1: "Yes (Suspicious)"}),
    ("Request_URL", "% external objects", {1: "Low", 0: "Medium", -1: "High"}),
    (
        "URL_of_Anchor",
        "% anchor links (external)",
        {1: "Low", 0: "Medium", -1: "High"},
    ),
    (
        "Links_in_tags",
        "Links in meta/script tags",
        {1: "Low", 0: "Medium", -1: "High"},
    ),
    (
        "SFH",
        "Server form handler",
        {1: "Same domain", 0: "External", -1: "Empty/Suspicious"},
    ),
    ("Submitting_to_email", "Form submits to email?", {1: "No", -1: "Yes"}),
    (
        "Abnormal_URL",
        "URL matches WHOIS?",
        {1: "Yes (Safe)", -1: "No (Suspicious)"},
    ),
    ("Redirect", "Number of redirects", {0: "0–1 (Safe)", 1: "2+"}),
    ("on_mouseover", "Status bar changes on hover?", {1: "No", -1: "Yes"}),
    ("RightClick", "Right-click disabled?", {1: "No", -1: "Yes"}),
    ("popUpWidnow", "Pop-up with text field?", {1: "No", -1: "Yes"}),
    ("Iframe", "Uses iFrame?", {1: "No", -1: "Yes"}),
    (
        "age_of_domain",
        "Domain age",
        {1: "≥ 6 months (Safe)", -1: "< 6 months (Suspicious)"},
    ),
    ("DNSRecord", "DNS record found?", {1: "Yes", -1: "No"}),
    (
        "web_traffic",
        "Alexa web traffic rank",
        {1: "Top 100k", 0: "Low", -1: "Not ranked"},
    ),
    ("Page_Rank", "Page rank", {1: "High", 0: "Medium", -1: "Low"}),
    ("Google_Index", "Indexed by Google?", {1: "Yes", -1: "No"}),
    (
        "Links_pointing_to_page",
        "Inbound links count",
        {1: "Many", 0: "Few", -1: "None"},
    ),
    (
        "Statistical_report",
        "In phishing reports?",
        {1: "No (Safe)", -1: "Yes (Suspicious)"},
    ),
]
FEATURE_COLS = [f[0] for f in FEATURE_META]
