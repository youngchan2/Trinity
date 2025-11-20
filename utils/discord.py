import requests
import time

def send_discord_notification(url: str, case: int, top_result: list, top_k: int, total: int, error: int):
    """Send benchmark results to Discord via webhook.

    Args:
        webhook_url: Discord webhook URL
        case_num: Case number
        top_results: List of top k BenchmarkResult objects
        total_results: Total number of results
        errors: Number of errors
    """

    if not url:
        return

    # Format top results
    top_k_text = ""
    for i, result in enumerate(top_result[:top_k], 1):
        top_k_text += f"{i}. IR{result.ir_id}: {result.execution_time:.4f} ms\n"

    # Calculate success rate
    success_rate = ((total - error) / total * 100) if total > 0 else 0

    # Build message
    message = f"**BWD Case {case} Completed**\n\n"
    message += f"Total: {total} | Success: {total - error} | Errors: {error}\n"
    message += f"Success Rate: {success_rate:.1f}%\n\n"
    message += f"**Top 5 Results:**\n```\n{top_k_text}```"

    # Discord embed
    data = {
        "embeds": [{
            "title": f"Backward Benchmark Case {case} Complete",
            "description": message,
            "color": 5814783 if error == 0 else 15158332,  # Blue if no errors, red otherwise
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }]
    }

    try:
        response = requests.post(url, json=data, timeout=10)
        if response.status_code != 204:
            print(f"Warning: Failed to send Discord notification: {response.status_code}")
    except Exception as e:
        print(f"Warning: Error sending Discord notification: {e}")

    return