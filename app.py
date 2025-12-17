from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.sla_apis import sla_router
from api.sla_tabs_api import sla_tabs_router
from api.ml_prediction_api import ml_prediction_router


def create_app() -> FastAPI:
    app = FastAPI(title="SLA APIs", version="1.0.0")

    # Allow the S3 static site and local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://bainocular.ai.s3-website-ap-southeast-1.amazonaws.com",
            "http://localhost",
            "http://localhost:3000",
            "*",  # optional: keep permissive if needed by your setup
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(sla_router)
    app.include_router(sla_tabs_router)
    app.include_router(ml_prediction_router)

    return app


app = create_app()


