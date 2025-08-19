from typing import Dict, Any

def diff(prev: Dict[str, Any], curr: Dict[str, Any]) -> Dict[str, Any]:
    changes = {
        "score_change": None,
        "new_issues": [],
        "resolved_issues": [],
        "meta_changes": {},
        "performance_changes": {},
        "content_changes": {},
    }

    # Very simple "score": based on presence of title/desc/canonical and load time
    def score(d):
        s = 0
        s += 20 if d.get("title") else 0
        s += 20 if d.get("description") else 0
        s += 20 if d.get("canonical") else 0
        lt = d.get("load_time_ms", 0)
        if lt:
            s += max(0, 40 - min(40, lt // 250))  # faster is better
        return s

    ps, cs = score(prev), score(curr)
    changes["score_change"] = {"previous": ps, "current": cs, "difference": cs-ps}

    # meta
    for k in ("title","description","canonical"):
        if prev.get(k) != curr.get(k):
            changes["meta_changes"][k] = {"old": prev.get(k), "new": curr.get(k)}

    # perf
    if prev.get("load_time_ms") != curr.get("load_time_ms"):
        changes["performance_changes"]["load_time_ms"] = {"old": prev.get("load_time_ms"), "new": curr.get("load_time_ms")}
    if prev.get("content_length") != curr.get("content_length"):
        changes["performance_changes"]["content_length"] = {"old": prev.get("content_length"), "new": curr.get("content_length")}

    # content (headings counts)
    if len(prev.get("h1", [])) != len(curr.get("h1", [])):
        changes["content_changes"]["h1_count"] = {"old": len(prev.get("h1", [])), "new": len(curr.get("h1", []))}
    if len(prev.get("h2", [])) != len(curr.get("h2", [])):
        changes["content_changes"]["h2_count"] = {"old": len(prev.get("h2", [])), "new": len(curr.get("h2", []))}

    # naive issues list (examples)
    prev_issues = set()
    curr_issues = set()
    if not prev.get("title"): prev_issues.add("missing_title")
    if not prev.get("description"): prev_issues.add("missing_description")
    if not curr.get("title"): curr_issues.add("missing_title")
    if not curr.get("description"): curr_issues.add("missing_description")

    changes["new_issues"] = sorted(list(curr_issues - prev_issues))
    changes["resolved_issues"] = sorted(list(prev_issues - curr_issues))
    changes["has_changes"] = any([changes["score_change"]["difference"] != 0, changes["meta_changes"], changes["performance_changes"], changes["content_changes"], changes["new_issues"], changes["resolved_issues"]])
    return changes