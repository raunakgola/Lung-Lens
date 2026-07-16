import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # 7860 = Hugging Face Spaces default
    uvicorn.run("App:app", host="0.0.0.0", port=port)
