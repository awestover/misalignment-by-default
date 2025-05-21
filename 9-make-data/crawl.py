import requests
import json
import time

TESTING = True
batch_size = 50
limit = 1000
if TESTING:
  batch_size = 2
  limit = 2

API_URL = "https://www.alignmentforum.org/graphql"
HEADERS = {
    "Content-Type": "application/json"
}

def graphql_query(query, variables=None, max_retries=5, backoff=2):
    payload = {
        "query": query,
        "variables": variables or {}
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code >= 500:
                print(f"Server error {response.status_code}, retrying...")
            else:
                raise Exception(f"Query failed: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}, retrying...")

        time.sleep(backoff * (attempt + 1))

    raise Exception(f"Query failed after {max_retries} retries.")

def get_posts():
    posts = []
    offset = 0

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
          contents {
            html
          }
          url
          postedAt
          user {
            displayName
          }
        }
      }
    }
    """

    while len(posts) < limit:
        print(f"Fetching batch at offset {offset}")
        result = graphql_query(query, variables={"limit": batch_size, "offset": offset})
        batch = result["data"]["posts"]["results"]
        if not batch:
            break
        posts.extend(batch)
        offset += batch_size
        if len(batch) < batch_size:
            break
        time.sleep(1)

    return posts

def clean_post(post):
    contents = post.get("contents")
    if not contents or not contents.get("html"):
        return None  # Skip posts with no content

    return {
        "id": post["_id"],
        "title": post["title"],
        "slug": post["slug"],
        "content_html": contents["html"],
        "url": post["url"],
        "posted_at": post["postedAt"],
        "author": post["user"]["displayName"]
    }

def main():
    print("Scraping Alignment Forum posts...")
    raw_posts = get_posts()
    print(f"Retrieved {len(raw_posts)} posts.")
    cleaned = [p for p in (clean_post(p) for p in raw_posts) if p is not None]
    output_file = "alignment_forum_posts.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in cleaned:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(cleaned)} cleaned posts to {output_file}")

if __name__ == "__main__":
    main()
