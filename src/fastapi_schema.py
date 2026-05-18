from pydantic import BaseModel, Field
from typing import List


class URLFeatures(BaseModel):
    """
    Input schema for phishing website prediction.
    """

    having_IP_Address: int = Field(..., example=1)
    URL_Length: int = Field(..., example=-1)
    Shortining_Service: int = Field(..., example=1)
    having_At_Symbol: int = Field(..., example=1)
    double_slash_redirecting: int = Field(..., example=-1)
    Prefix_Suffix: int = Field(..., example=-1)
    having_Sub_Domain: int = Field(..., example=0)
    SSLfinal_State: int = Field(..., example=1)
    Domain_registeration_length: int = Field(..., example=-1)
    Favicon: int = Field(..., example=1)
    port: int = Field(..., example=1)
    HTTPS_token: int = Field(..., example=-1)
    Request_URL: int = Field(..., example=1)
    URL_of_Anchor: int = Field(..., example=0)
    Links_in_tags: int = Field(..., example=1)
    SFH: int = Field(..., example=1)
    Submitting_to_email: int = Field(..., example=-1)
    Abnormal_URL: int = Field(..., example=1)
    Redirect: int = Field(..., example=0)
    on_mouseover: int = Field(..., example=1)
    RightClick: int = Field(..., example=1)
    popUpWidnow: int = Field(..., example=1)
    Iframe: int = Field(..., example=1)
    age_of_domain: int = Field(..., example=-1)
    DNSRecord: int = Field(..., example=-1)
    web_traffic: int = Field(..., example=0)
    Page_Rank: int = Field(..., example=-1)
    Google_Index: int = Field(..., example=1)
    Links_pointing_to_page: int = Field(..., example=0)
    Statistical_report: int = Field(..., example=-1)


class PredictionResponse(BaseModel):
    """
    Response schema for phishing website prediction.
    """

    prediction: int = Field(..., description="1 = Legitimate, 0 = Phishing")
    label: str = Field(..., description="Human readable verdict")
    confidence: float | None = Field(None, description="Probability of predicted class")


class BatchRequest(BaseModel):
    """
    Request schema for batch phishing website prediction.
    """

    records: List[URLFeatures]


class BatchResponse(BaseModel):
    """
    Response schema for batch phishing website prediction.
    """

    total: int
    phishing_count: int
    legitimate_count: int
    results: List[PredictionResponse]
