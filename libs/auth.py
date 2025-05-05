from getpass import getpass

# Global dictionary to store credentials for session-wide access
_credentials = {}

def authenticate():
    """
    Prompt user for username and password and store them in a global _credentials dictionary.
    """
    global _credentials
    print("Note: Credentials will be hidden for security.")
    username = getpass("Enter username: ")
    password = getpass("Enter password: ")
    _credentials['username'] = username
    _credentials['password'] = password
    print("Credentials saved.")

def get_credentials():
    """
    Retrieve stored credentials. Raises an error if not set.
    """
    if not _credentials:
        raise ValueError("No credentials found. Please run authenticate() first.")
    return _credentials