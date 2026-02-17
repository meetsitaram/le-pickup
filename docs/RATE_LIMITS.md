# Hugging Face Hub rate limits

When calling the Hub (e.g. `discover_lerobot_datasets.py`), requests count against **Hub API** limits.

## Limits (per 5-minute window)

| Plan              | Hub API requests |
|-------------------|-------------------|
| Anonymous (per IP)| 500               |
| Free (logged in)  | 1,000             |
| PRO / Team / etc. | 2,500+            |

Exact numbers: [Hub rate limits](https://huggingface.co/docs/hub/en/rate-limits). Quotas are over a **5-minute** window, not per second.

## Do we need to rate-limit?

- **Current discovery script:** One `list_datasets(filter=..., limit=500)` call. The client typically uses a few paginated requests (often &lt;20) to stream results. That stays well under 500–1000, so **no extra rate limiting is required** for normal runs.
- **If you add scripts that call the API per dataset** (e.g. fetching each dataset card or metadata in a loop), then many hundreds of requests in a short time could hit the limit. In that case: add a small delay between requests (e.g. 0.5–1 s) and/or batch work.

## Avoid getting blocked

1. **Use a token (recommended)**  
   Set `HF_TOKEN` (or log in with `hf auth login`). Logged-in users get the Free (or higher) quota and are less likely to be blocked than anonymous IPs.

2. **Rely on `huggingface_hub`**  
   The library (v1.2.0+) handles **429 Too Many Requests** for you: it reads the `RateLimit` header, waits until the window resets, then retries. No need to implement your own backoff for 429s.

3. **Check usage**  
   See current usage and limits: [Billing / rate limit dashboard](https://huggingface.co/settings/billing).

## Summary

- No extra rate limiting is needed for the current discovery script.
- Set `HF_TOKEN` (or log in) to get a higher quota and better behavior.
- If you add scripts that make many Hub API calls in a loop, add a short delay between requests and/or batch work.
