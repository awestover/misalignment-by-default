import requests
import json
from tqdm import tqdm
import time

API_URL = "https://www.alignmentforum.org/graphql"
HEADERS = {
    "Content-Type": "application/json"
}

def graphql_query(query, variables=None):
    payload = {
        "query": query,
        "variables": variables or {}
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        raise Exception(f"Query failed: {response.text}")
    return response.json()

def get_posts(limit=1000):
    posts = []
    has_more = True
    after = None
    batch_size = 50

    query = """
    query AlignmentPosts($limit: Int, $after: String) {
      posts(input: {
        terms: {
          view: "alignmentForum"
          limit: $limit
          after: $after
        }
      }) {
        results {
          _id
          title
          slug
          content
          url
          postedAt
          user {
            displayName
          }
        }
        hasMore
      }
    }
    """

    while has_more and len(posts) < limit:
        result = graphql_query(query, variables={"limit": batch_size, "after": after})
        batch = result["data"]["posts"]["results"]
        posts.extend(batch)
        has_more = result["data"]["posts"]["hasMore"]
        if batch:
            after = batch[-1]["_id"]
        time.sleep(1)

    return posts

def clean_post(post):
    return {
        "id": post["_id"],
        "title": post["title"],
        "slug": post["slug"],
        "content": post["content"],
        "url": post["url"],
        "posted_at": post["postedAt"],
        "author": post["user"]["displayName"]
    }

def main():
    print("Scraping Alignment Forum posts...")
    raw_posts = get_posts(limit=500)  # or however many you'd like
    print(f"Retrieved {len(raw_posts)} posts.")
    cleaned = [clean_post(p) for p in raw_posts]

    with open("alignment_forum_posts.jsonl", "w", encoding="utf-8") as f:
        for item in cleaned:
            f.write(json.dumps(item) + "\n")

    print("Saved to alignment_forum_posts.jsonl")

if __name__ == "__main__":
    main()
