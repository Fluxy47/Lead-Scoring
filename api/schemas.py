# api/schemas.py

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class LeadFeatures(BaseModel):
    """
    Single lead input via JSON.
    Field aliases match the original CSV column names exactly.
    Client sends: {"Lead Source": "Google", "TotalVisits": 3.0}
    No transformation needed — these go straight into a DataFrame
    with the exact column names the pipeline expects.
    """

    model_config = ConfigDict(populate_by_name=True)

    # Using alias= means the client sends the original CSV name
    # The pipeline receives a DataFrame with those same names
    # DataCleaner handles renaming internally — nothing changes

    Lead_Origin: Optional[str] = Field(None, alias="Lead Origin")
    Lead_Source: Optional[str] = Field(None, alias="Lead Source")
    Do_Not_Email: Optional[str] = Field(None, alias="Do Not Email")
    Do_Not_Call: Optional[str] = Field(None, alias="Do Not Call")
    TotalVisits: Optional[float] = Field(None, alias="TotalVisits")
    Total_Time_Spent_on_Website: Optional[int] = Field(
        None, alias="Total Time Spent on Website"
    )
    Page_Views_Per_Visit: Optional[float] = Field(
        None, alias="Page Views Per Visit"
    )
    Last_Activity: Optional[str] = Field(None, alias="Last Activity")
    Country: Optional[str] = Field(None, alias="Country")
    Specialization: Optional[str] = Field(None, alias="Specialization")
    How_did_you_hear: Optional[str] = Field(
        None, alias="How did you hear about X Education"
    )
    Occupation: Optional[str] = Field(
        None, alias="What is your current occupation"
    )
    Course_Priority: Optional[str] = Field(
        None, alias="What matters most to you in choosing a course"
    )
    Search: Optional[str] = Field(None, alias="Search")
    Magazine: Optional[str] = Field(None, alias="Magazine")
    Newspaper_Article: Optional[str] = Field(
        None, alias="Newspaper Article"
    )
    X_Education_Forums: Optional[str] = Field(
        None, alias="X Education Forums"
    )
    Newspaper: Optional[str] = Field(None, alias="Newspaper")
    Digital_Advertisement: Optional[str] = Field(
        None, alias="Digital Advertisement"
    )
    Through_Recommendations: Optional[str] = Field(
        None, alias="Through Recommendations"
    )
    Receive_Course_Updates: Optional[str] = Field(
        None, alias="Receive More Updates About Our Courses"
    )
    Tags: Optional[str] = Field(None, alias="Tags")
    Lead_Quality: Optional[str] = Field(None, alias="Lead Quality")
    Supply_Chain_Updates: Optional[str] = Field(
        None, alias="Update me on Supply Chain Content"
    )
    DM_Content_Updates: Optional[str] = Field(
        None, alias="Get updates on DM Content"
    )
    Lead_Profile: Optional[str] = Field(None, alias="Lead Profile")
    City: Optional[str] = Field(None, alias="City")
    Asym_Activity_Index: Optional[str] = Field(
        None, alias="Asymmetrique Activity Index"
    )
    Asym_Profile_Index: Optional[str] = Field(
        None, alias="Asymmetrique Profile Index"
    )
    Asym_Activity_Score: Optional[float] = Field(
        None, alias="Asymmetrique Activity Score"
    )
    Asym_Profile_Score: Optional[float] = Field(
        None, alias="Asymmetrique Profile Score"
    )
    Agreed_To_Pay_Cheque: Optional[str] = Field(
        None, alias="I agree to pay the amount through cheque"
    )
    Free_Interview_Guide: Optional[str] = Field(
        None, alias="A free copy of Mastering The Interview"
    )
    Last_Notable_Activity: Optional[str] = Field(
        None, alias="Last Notable Activity"
    )

    def to_dataframe(self):
        """
        Convert to DataFrame using alias names (original CSV column names).
        These go directly into the pipeline — no mapping needed.
        """
        import pandas as pd

        # by_alias=True means keys are "Lead Source" not "Lead_Source"
        return pd.DataFrame([self.model_dump(by_alias=True)])


class PredictionResponse(BaseModel):
    conversion_probability: float
    label: str
    threshold_used: float


class BatchPredictionResponse(BaseModel):
    total_leads: int
    results: list[PredictionResponse]


class CSVPredictionResponse(BaseModel):
    total_leads: int
    results: list[dict]  # includes original row data + score columns