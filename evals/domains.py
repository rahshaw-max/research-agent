"""
Primary-source domain classifier.
Edit PRIMARY_SUFFIXES and PRIMARY_DOMAINS to tune the classifier.
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
