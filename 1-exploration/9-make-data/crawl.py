import requests
import json
import time
from bs4 import BeautifulSoup

TESTING = False # Set to False for full scrape
batch_size = 50
limit = 4000
if TESTING:
    batch_size = 5
    limit = 10

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
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=20)
            response.raise_for_status()
            response_json = response.json()
            if "errors" in response_json:
                print(f"GraphQL API returned errors (Attempt {attempt + 1}/{max_retries}): {response_json['errors']}")
            return response_json
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e} (Attempt {attempt + 1}/{max_retries}). Status: {e.response.status_code}. Response: {e.response.text[:200]}...")
            if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                print(f"Critical client error {e.response.status_code}. Aborting retries for this query.")
                break
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e} (Attempt {attempt + 1}/{max_retries}). Retrying in {backoff ** attempt}s...")
        if attempt < max_retries - 1:
            time.sleep(backoff ** attempt)
    print(f"Query failed after {max_retries} retries for query: {query[:100]}... with variables: {variables}")
    return None

def get_posts():
    posts = []
    offset = 0
    query_string = """
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
    while True:
        if len(posts) >= limit:
            print(f"Reached or exceeded desired limit of {limit} posts. Current count: {len(posts)}.")
            break
        remaining_limit = limit - len(posts)
        current_batch_size = min(batch_size, remaining_limit)
        if current_batch_size <= 0:
            break
        print(f"Fetching batch of {current_batch_size} posts at offset {offset}. Total posts collected: {len(posts)}/{limit}")
        result_json = graphql_query(query_string, variables={"limit": current_batch_size, "offset": offset})
        if result_json is None:
            print("GraphQL query returned None after retries. Stopping.")
            break
        if TESTING:
            print(f"Raw result from API: {json.dumps(result_json, indent=2)}")
        if "errors" in result_json and result_json["errors"]:
            print(f"GraphQL query failed with errors: {result_json['errors']}. Halting.")
            break
        data = result_json.get("data")
        if not data:
            print(f"No 'data' key in GraphQL response. Response: {result_json}")
            break
        posts_data = data.get("posts")
        if not posts_data:
            print(f"No 'posts' key in 'data'. Data: {data}")
            break
        batch = posts_data.get("results")
        if batch is None:
            print(f"No 'results' key in 'posts_data'. Posts data: {posts_data}")
            break
        if not batch:
            print("Received an empty batch. Assuming no more posts.")
            break
        print(f"Fetched {len(batch)} posts in this batch.")
        posts.extend(batch)
        offset += len(batch)
        if len(batch) < current_batch_size:
            print("Received fewer posts than requested in batch. Assuming end of available data.")
            break
        time.sleep(1)
    return posts[:limit]

def clean_post(post_data):
    if not isinstance(post_data, dict):
        print(f"Warning: clean_post received non-dict data: {post_data}")
        return None
    contents = post_data.get("contents")
    html_content = None
    if contents and contents.get("html"):
        html_content = contents["html"]
    else:
        if TESTING:
            print(f"Skipping post due to missing content/HTML: {post_data.get('_id', 'Unknown ID')}")
        return None
    user_data = post_data.get("user")
    author_name = "Unknown Author"
    if user_data and isinstance(user_data, dict) and user_data.get("displayName"):
        author_name = user_data["displayName"]
    elif user_data and TESTING:
        print(f"Post {post_data.get('_id','ID missing')} has user data but no displayName: {user_data}")

    plain_text_content = ""
    if html_content:
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            plain_text_content = soup.get_text(separator=" ", strip=True)
        except Exception as e:
            if TESTING:
                print(f"Error parsing HTML for post {post_data.get('_id', 'Unknown ID')}: {e}")
            plain_text_content = "[Error parsing HTML]" # Indicate parsing error in text
    cleaned = {
        "id": post_data.get("_id"),
        "title": post_data.get("title"),
        "slug": post_data.get("slug"),
        "content_html": html_content,
        "content_text": plain_text_content,
        "url": post_data.get("url"),
        "posted_at": post_data.get("postedAt"),
        "author": author_name
    }
    if not cleaned["id"] and TESTING:
        print(f"Warning: Post missing '_id'. Original data: {post_data}")
    return cleaned

def main():
    print("Scraping Alignment Forum posts...")
    raw_posts = get_posts()
    print(f"Retrieved {len(raw_posts)} raw posts.")

    if not raw_posts:
        print("No posts were retrieved. Exiting.")
        return

    cleaned_posts = []
    for p in raw_posts:
        cleaned_p = clean_post(p)
        if cleaned_p:
            cleaned_posts.append(cleaned_p)

    print(f"Successfully cleaned {len(cleaned_posts)} posts.")

    if not cleaned_posts:
        print("No posts were successfully cleaned. Not writing to file(s).")
        return

    # --- Save all cleaned data to a single JSON file as a list ---
    output_json_file = "alignment_forum_posts.json"
    try:
        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_posts, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(cleaned_posts)} cleaned posts (as a list) to {output_json_file}")
    except IOError as e:
        print(f"Error writing to JSON file {output_json_file}: {e}")

    # --- Save only the extracted text to a new .txt file ---
    output_text_file = "alignment_forum_extracted_texts.txt"
    texts_saved_count = 0
    try:
        with open(output_text_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(cleaned_posts):
                text_content = item.get("content_text", "").strip() # Get text, default to empty string, strip
                if text_content: # Only write if there's actual text
                    f.write(text_content)
                    f.write("\n\n") # Add two newlines as a separator between posts
                    f.write("=" * 80) # Add a visual separator line
                    f.write("\n\n") # Add two more newlines before the next post
                    texts_saved_count +=1
                elif TESTING: # If testing, you might want to know about empty texts
                    print(f"Post ID {item.get('id', 'N/A')} had no text content to save.")
        if texts_saved_count > 0:
            print(f"Saved extracted text from {texts_saved_count} posts to {output_text_file}")
        else:
            print(f"No text content found in cleaned posts to save to {output_text_file}")
    except IOError as e:
        print(f"Error writing to text file {output_text_file}: {e}")


if __name__ == "__main__":
    main()