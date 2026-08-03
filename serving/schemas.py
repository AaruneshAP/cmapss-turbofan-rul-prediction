"""
Pydantic schemas for the CMAPSS RUL prediction API.

Defines the request/response models with validation to ensure
sensor payloads are well-formed before reaching the prediction pipeline.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class SensorReading(BaseModel):
    """
    A single cycle's worth of sensor readings.
    
    Each reading contains the 3 operational settings and 21 sensor values
    as recorded by the engine monitoring system.
    """
    setting_1: float = Field(..., description="Operational setting 1")
    setting_2: float = Field(..., description="Operational setting 2")
    setting_3: float = Field(..., description="Operational setting 3")
    sensor_1: float = Field(..., description="Sensor 1 reading")
    sensor_2: float = Field(..., description="Sensor 2 reading")
    sensor_3: float = Field(..., description="Sensor 3 reading")
    sensor_4: float = Field(..., description="Sensor 4 reading")
    sensor_5: float = Field(..., description="Sensor 5 reading")
    sensor_6: float = Field(..., description="Sensor 6 reading")
    sensor_7: float = Field(..., description="Sensor 7 reading")
    sensor_8: float = Field(..., description="Sensor 8 reading")
    sensor_9: float = Field(..., description="Sensor 9 reading")
    sensor_10: float = Field(..., description="Sensor 10 reading")
    sensor_11: float = Field(..., description="Sensor 11 reading")
    sensor_12: float = Field(..., description="Sensor 12 reading")
    sensor_13: float = Field(..., description="Sensor 13 reading")
    sensor_14: float = Field(..., description="Sensor 14 reading")
    sensor_15: float = Field(..., description="Sensor 15 reading")
    sensor_16: float = Field(..., description="Sensor 16 reading")
    sensor_17: float = Field(..., description="Sensor 17 reading")
    sensor_18: float = Field(..., description="Sensor 18 reading")
    sensor_19: float = Field(..., description="Sensor 19 reading")
    sensor_20: float = Field(..., description="Sensor 20 reading")
    sensor_21: float = Field(..., description="Sensor 21 reading")


class PredictionRequest(BaseModel):
    """
    Prediction request: a sequence of recent sensor readings for one engine.
    
    The API expects the last N cycles of sensor data. More cycles generally
    yield better predictions (the feature pipeline uses rolling windows of
    up to 20 cycles).
    
    Minimum: 1 cycle (rolling features will be computed with limited history).
    Recommended: 20+ cycles for best accuracy.
    """
    readings: list[SensorReading] = Field(
        ...,
        min_length=1,
        description="List of sensor readings, ordered chronologically (oldest first).",
    )

    @field_validator("readings")
    @classmethod
    def validate_readings_length(cls, v):
        if len(v) > 500:
            raise ValueError("Maximum 500 cycles per request to prevent abuse.")
        return v


class PredictionResponse(BaseModel):
    """Response containing the predicted Remaining Useful Life."""
    predicted_rul: float = Field(
        ..., description="Predicted Remaining Useful Life in cycles."
    )
    model_version: str = Field(
        ..., description="Identifier of the model used for prediction."
    )
    confidence_note: str = Field(
        default="",
        description="Additional context about the prediction quality.",
    )


class HealthResponse(BaseModel):
    """Response from the health check endpoint."""
    status: str = Field(..., description="Service status: 'healthy' or 'degraded'.")
    model_loaded: bool = Field(..., description="Whether the model is loaded.")
    model_version: str = Field(default="", description="Loaded model identifier.")
