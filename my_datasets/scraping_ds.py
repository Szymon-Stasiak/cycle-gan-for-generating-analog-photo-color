import asyncio
import aiohttp
import aiofiles
import os
import json
from pathlib import Path


def load_unsplash_key():
    secrets_path = Path(__file__).resolve().parent.parent / "secrets.json"
    with secrets_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("UNSPLASH_ACCESS_KEY")


UNSPLASH_ACCESS_KEY = load_unsplash_key()



# ---------------------------------------------------
# 1) Pobieranie metadanych zdjęć (LISTA URL)
# ---------------------------------------------------
async def fetch_random_photos(session, count, query=None):
    url = "https://api.unsplash.com/photos/random"
    params = {
        "client_id": UNSPLASH_ACCESS_KEY,
        "count": count
    }
    if query:
        params["query"] = query

    async with session.get(url, params=params) as resp:
        return await resp.json()


# ---------------------------------------------------
# 2) Pobieranie jednego obrazu
# ---------------------------------------------------
async def download_image(session, url, save_path):
    async with session.get(url) as resp:
        data = await resp.read()

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(data)


# ---------------------------------------------------
# 3) Główna funkcja – współbieżne pobieranie
# ---------------------------------------------------
async def download_photos_async(count: int, target_dir: str, query: str | None = None):
    os.makedirs(target_dir, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        photos = await fetch_random_photos(session, count, query)

        tasks = []
        for photo in photos:
            img_url = photo["urls"]["full"]
            filename = f"{query or 'random'}_{photo['id']}.jpg"
            save_path = os.path.join(target_dir, filename)

            tasks.append(download_image(session, img_url, save_path))

        print(f"🚀 Pobieram {len(tasks)} zdjęć równolegle…")

        await asyncio.gather(*tasks)

        print("✅ Wszystkie zdjęcia pobrane!")


# ---------------------------------------------------
# 4) Użycie w __main__
# ---------------------------------------------------
if __name__ == "__main__":
    for _ in range(30):
        asyncio.run(download_photos_async(30, "datasets/analog_photos", query="analog"))
        asyncio.run(download_photos_async(30, "datasets/random_photos"))