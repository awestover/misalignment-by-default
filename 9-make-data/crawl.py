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
    offset = 0
    batch_size = 50

    query = """
    query AlignmentPosts($limit: Int, $offset: Int) {
      posts(input: {
        terms: {
          view: "alignmentForum"
          limit: $limit
          offset: $offset
        }
      }) {
        results {
          _id
          title
          slug
          contents
          url
          postedAt
          user {
            displayName
          }
        }
      }
    }
    """

    while has_more and len(posts) < limit:
        print(f"Fetching batch at offset {offset}")
        result = graphql_query(query, variables={"limit": batch_size, "offset": offset})
        batch = result["data"]["posts"]["results"]
        posts.extend(batch)
        offset += batch_size
        has_more = len(batch) == batch_size
        time.sleep(1)

    return posts

def clean_post(post):
    return {
        "id": post["_id"],
        "title": post["title"],
        "slug": post["slug"],
        "content": post["contents"],
        "url": post["url"],
        "posted_at": post["postedAt"],
        "author": post["user"]["displayName"]
    }

def main():
    print("Scraping Alignment Forum posts...")
    raw_posts = get_posts(limit=500)  # Change limit as needed
    print(f"Retrieved {len(raw_posts)} posts.")
    cleaned = [clean_post(p) for p in raw_posts]

    output_file = "alignment_forum_posts.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in cleaned:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(cleaned)} posts to {output_file}")

if __name__ == "__main__":
    main()
