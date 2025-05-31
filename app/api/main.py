from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import predict

app = FastAPI()

# ✅ Abilita CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # oppure ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Registra router
app.include_router(predict.router)

@app.get("/")
def root():
    return {"message": "GoEmotions API is running"}
