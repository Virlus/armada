import re

# def parse_message_regex(message, template):
#     """
#     Parse message using regex pattern derived from template.
#     Template should use {} for parts to extract.
#     """
#     # Convert template to regex pattern
#     pattern = template.replace("{}", "(.*?)")
#     match = re.match(pattern, message)
#     if not match:
#         raise ValueError(f"Message '{message}' does not match template '{template}'")
#     return match.groups()

def parse_message_regex(message: str, template: str) -> list:
    regex_pattern = re.escape(template).replace(r"\{\}", "(.*?)") #replace {} with capture groups (.*?)
    match = re.fullmatch(regex_pattern, message)
    if match:
        return list(match.groups())
    return []