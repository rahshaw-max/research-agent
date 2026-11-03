"""
Primary-source domain classifier + source-type heuristic.
Edit the sets below to tune classification.
"""
from urllib.parse import urlparse

# TLD/suffix rules — any domain ending in these counts as primary
PRIMARY_SUFFIXES = [
    ".gov",
    ".edu",
    ".mil",
]

# Exact netloc matches (after stripping leading "www.")
PRIMARY_DOMAINS = {
    # Healthcare regulators and standards bodies
    "cms.gov", "hhs.gov", "cdc.gov", "fda.gov", "nih.gov",
    "ncqa.org", "ahip.org", "aha.org", "aafp.org", "acp.org",
    "hl7.org", "x12.org", "ihe.net", "caqh.org",
    "urac.org", "jointcommission.org",
    # Academic journals
    "pubmed.ncbi.nlm.nih.gov", "nejm.org", "jamanetwork.com",
    "thelancet.com", "bmj.com", "nature.com", "cell.com",
    "sciencedirect.com", "arxiv.org", "acm.org", "ieee.org",
    "hbr.org",
    # SEC / financial filings
    "sec.gov", "edgar.sec.gov",
    # Vendor official documentation (docs.* subdomains handled below)
    "docs.aws.amazon.com", "learn.microsoft.com", "cloud.google.com",
    # Industry analyst / research (conservative list — only well-known)
    "gartner.com", "forrester.com", "kff.org", "rand.org",
    "mckinsey.com", "pwc.com", "deloitte.com",
}

# Subdomain patterns — if netloc contains these subdomains, it's primary
PRIMARY_SUBDOMAIN_PREFIXES = ("docs.", "developer.", "developers.", "ir.")

# ── Source-type heuristic ─────────────────────────────────────────────────
# Domains not in PRIMARY_DOMAINS that are classified as news/analyst/blog.
# "other" is the catch-all for anything not matched.

_NEWS_DOMAINS = {
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "bbc.com", "bbc.co.uk",
    "cnn.com", "nytimes.com", "theguardian.com", "forbes.com", "businessinsider.com",
    "techcrunch.com", "theverge.com", "wired.com", "arstechnica.com",
    "zdnet.com", "cnet.com", "computerworld.com", "infoworld.com",
    "healthcareitnews.com", "modernhealthcare.com", "beckershospitalreview.com",
    "fiercehealthcare.com", "hcinnovationgroup.com", "statnews.com",
    "axios.com", "politico.com", "thehill.com",
}

_ANALYST_DOMAINS = {
    "idc.com", "451research.com", "gartner.com", "forrester.com",  # gartner/forrester also in PRIMARY
    "woodmackenzie.com", "accenture.com", "capgemini.com", "kpmg.com",
    "ey.com", "bcg.com", "bain.com", "oliverwyman.com", "s&p.com", "spglobal.com",
    "everestgrp.com", "issg.com", "hfsresearch.com",
}

_BLOG_INDICATORS = {
    "medium.com", "substack.com", "wordpress.com", "blogspot.com",
    "ghost.io", "hashnode.dev", "dev.to", "beehiiv.com",
}


def _netloc(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""


def is_primary(url: str) -> bool:
    """Return True if the URL is from a primary / authoritative source."""
    netloc = _netloc(url)
    if not netloc:
        return False
    # Suffix check
    for suffix in PRIMARY_SUFFIXES:
        if netloc.endswith(suffix):
            return True
    # Exact domain match
    if netloc in PRIMARY_DOMAINS:
        return True
    # Check bare domain against set (strip one level of subdomain)
    parts = netloc.split(".")
    if len(parts) > 2:
        bare = ".".join(parts[-2:])
        if bare in PRIMARY_DOMAINS:
            return True
    # Subdomain prefix check (e.g. docs.snowflake.com)
    for prefix in PRIMARY_SUBDOMAIN_PREFIXES:
        if netloc.startswith(prefix):
            return True
    return False


def domain_of(url: str) -> str:
    return _netloc(url)


def source_type_of(url: str) -> str:
    """Classify a URL as: primary | analyst | news | blog | other."""
    if is_primary(url):
        return "primary"
    netloc = _netloc(url)
    if not netloc:
        return "other"
    parts = netloc.split(".")
    bare = ".".join(parts[-2:]) if len(parts) >= 2 else netloc
    if bare in _NEWS_DOMAINS or netloc in _NEWS_DOMAINS:
        return "news"
    if bare in _ANALYST_DOMAINS or netloc in _ANALYST_DOMAINS:
        return "analyst"
    if bare in _BLOG_INDICATORS or netloc in _BLOG_INDICATORS:
        return "blog"
    for ind in _BLOG_INDICATORS:
        if netloc.endswith("." + ind):
            return "blog"
    return "other"
