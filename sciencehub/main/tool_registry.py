import os

DOMAINS_PATH = os.path.join("sciencehub", "domains")


def get_categories():
    """Return all available science domains."""
    if not os.path.isdir(DOMAINS_PATH):
        return []

    return sorted(
        folder.replace("_", " ").title()
        for folder in os.listdir(DOMAINS_PATH)
        if os.path.isdir(os.path.join(DOMAINS_PATH, folder))
    )


def get_tools_for_category(category):
    """Return tools for a given human-readable category name."""
    folder_name = category.lower().replace(" ", "_")
    domain_path = os.path.join(DOMAINS_PATH, folder_name)

    if not os.path.isdir(domain_path):
        return ["No tools available"]

    tools = [
        os.path.splitext(f)[0]
        for f in os.listdir(domain_path)
        if os.path.isfile(os.path.join(domain_path, f))
           and not f.startswith("_")
    ]

    return sorted(tools) if tools else ["No tools available"]
